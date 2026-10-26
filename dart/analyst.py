"""The LLM analyst of DART: prompts, strict JSON schemas, program checks and transports.

The analyst writes a difference memory from the observation pairs of a domain (observation prompt)
and revises it on its worst-reproduced cases (revision prompt). It sees the paired texts and program
statistics, but not the name of the attack, the generator or the domain.

Transports (config `method.analyst.backend`):
  api    direct HTTPS calls to an OpenAI-compatible /chat/completions endpoint (default);
  queue  a file queue served by relay/gpt_broker.py + relay/gpt_relay.py, for GPU servers without
         internet access (the setup of the paper's runs).
"""

import itertools
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeout
from datetime import datetime, timezone
from pathlib import Path

import text_utils as T

DIMS = ["length_structure", "style", "form", "sentence", "elaboration", "wording"]


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------------------------
# Prompt templates. Placeholders are filled in one pass (fill()).
# ---------------------------------------------------------------------------------------------

TEMPLATE_OBSERVE = """You are analyzing how AI-written texts differ from human-written texts in one text domain.
You will see {n} pairs. In each pair, Text H was written by a human and Text A was produced
by an AI system on the same topic or source. The topic is shared within a pair, so ignore
the topic and describe only how A is written differently from H.

Each pair has a tag:
- HARD: a detector separates A from H poorly for this pair; the differences may be subtle.
- RANDOM: a randomly chosen pair.
Each pair also comes with statistics computed by a program. Use them as given and do not
count anything yourself. Length ratios are handled by the program, so do not state lengths
in numbers.

Compare A with H on six dimensions:
1. length_structure: relative length, paragraphs, lists, headings
2. style: formality, emotion, person, hedging, stock phrases
3. form: how the text opens and ends, summaries, itemization
4. sentence: sentence length and variety, complex sentences, questions and imperatives
5. elaboration: concrete details, examples, personal experience versus general statements
6. wording: unusual word choices, awkward collocations, small grammar slips, odd punctuation

Then:
- Group the pairs into 2 to 6 difference patterns. A pattern is a set of differences that
  tend to occur together. Patterns found mainly in HARD pairs matter most; describe them
  precisely.
- Write each pattern so that a writer could reproduce it on any topic and in any kind of
  text. Do not mention topics, names or facts from the texts. Do not copy any phrase longer
  than 5 words.
- Assign every pair to exactly one pattern.

Return JSON only:
{
  "patterns": [
    {"id": "P1", "name": "<short label>",
     "differences": {"length_structure": "...", "style": "...", "form": "...",
                     "sentence": "...", "elaboration": "...", "wording": "..."}}
  ],
  "assignments": [{"pair_id": "<id>", "pattern": "<pattern id>"}],
  "domain_overview": {"topics": "<one sentence>", "forms": "<one sentence>"}
}

Pairs:
{pair_blocks}"""

TEMPLATE_REVISE = """You maintain a description of how AI-written texts differ from human-written texts in one
domain. The description is a list of difference patterns. Mirrors were written by following
these patterns, but some of them do not match the real AI texts well.

Below are {m} cases. In each case:
- Text H is a human-written text.
- Text A is the real AI text on the same topic.
- Text M is a mirror written from H's content by following the named pattern.
Word counts are computed by a program; use them as given.

For each case, compare M with A on the six dimensions (length_structure, style, form,
sentence, elaboration, wording): what does A have that M lacks, and what does M have that
A lacks?

Then rewrite the pattern descriptions so that future mirrors match the real AI texts better.
- Keep the same pattern ids and the same number of patterns. Only rewrite "name" and
  "differences".
- Keep the descriptions usable on any topic and in any kind of text. Do not mention topics,
  names or facts. Do not copy any phrase longer than 5 words.

Return JSON only:
{
  "case_notes": [{"case_id": "<id>", "A_has_M_lacks": "...", "M_has_A_lacks": "..."}],
  "patterns": [<every pattern, in the same format as the current patterns>]
}

Current patterns:
{patterns_json}

Cases:
{case_blocks}"""


def fill(template: str, values: dict) -> str:
    """Fill {placeholders} in one pass (braces of the JSON examples in the templates stay untouched)."""
    pattern = re.compile(r"\{(" + "|".join(re.escape(k) for k in values) + r")\}")
    return pattern.sub(lambda m: str(values[m.group(1)]), template)


def pair_block(label: str, tag: str, h: str, a: str) -> str:
    return "\n".join([f"[pair {label}] tag: {tag}", T.stats_line("H", T.stats(h)), T.stats_line("A", T.stats(a)),
                      "Text H:", "<<<", h, ">>>", "Text A:", "<<<", a, ">>>"])


def observation_prompt(blocks: list) -> str:
    return fill(TEMPLATE_OBSERVE, {"n": len(blocks), "pair_blocks": "\n\n".join(blocks)})


def case_block(label: str, pattern_id: str, h: str, a: str, m: str) -> str:
    return "\n".join([f"[case {label}] pattern: {pattern_id}",
                      f"words: H={T.words(h)}, A={T.words(a)}, M={T.words(m)}",
                      "Text H:", "<<<", h, ">>>", "Text A:", "<<<", a, ">>>", "Text M:", "<<<", m, ">>>"])


def revision_prompt(patterns: list, cases: list) -> str:
    current = [{"id": p["id"], "name": p["name"], "differences": p["differences"]} for p in patterns]
    return fill(TEMPLATE_REVISE, {"m": len(cases), "patterns_json": json.dumps(current, ensure_ascii=False, indent=2),
                                  "case_blocks": "\n\n".join(cases)})


# ---------------------------------------------------------------------------------------------
# JSON schemas (strict structured output) and program checks
# ---------------------------------------------------------------------------------------------

def _obj(props: dict) -> dict:
    return {"type": "object", "additionalProperties": False, "required": list(props), "properties": props}


_S = {"type": "string"}
_PATTERN = _obj({"id": _S, "name": _S, "differences": _obj({k: _S for k in DIMS})})
OBSERVE_SCHEMA = _obj({"patterns": {"type": "array", "items": _PATTERN},
                       "assignments": {"type": "array", "items": _obj({"pair_id": _S, "pattern": _S})},
                       "domain_overview": _obj({"topics": _S, "forms": _S})})
REVISE_SCHEMA = _obj({"case_notes": {"type": "array", "items": _obj({"case_id": _S, "A_has_M_lacks": _S,
                                                                     "M_has_A_lacks": _S})},
                      "patterns": {"type": "array", "items": _PATTERN}})


def _nonempty(x) -> bool:
    return isinstance(x, str) and bool(x.strip())


def _check_patterns(patterns, reference_ngrams) -> str | None:
    ids = []
    for p in patterns:
        if not isinstance(p, dict) or not _nonempty(p.get("id")) or not _nonempty(p.get("name")):
            return "pattern without id/name"
        diffs = p.get("differences")
        if not isinstance(diffs, dict) or not all(_nonempty(diffs.get(k)) for k in DIMS):
            return f"pattern {p.get('id')} has empty differences"
        ids.append(p["id"].strip())
        for text in [p["name"], *(diffs[k] for k in DIMS)]:
            hit = T.shared_ngram(text, reference_ngrams)
            if hit:
                return f"8-gram overlap with training text in {p['id']}: {hit!r}"
    if len(set(ids)) != len(ids):
        return "duplicate pattern ids"
    return None


def validate_observation(obj, pair_labels: set, reference_ngrams: set) -> str | None:
    """2-6 patterns; every pair assigned to exactly one pattern; descriptions pass the 8-gram check."""
    patterns = obj.get("patterns") if isinstance(obj, dict) else None
    if not isinstance(patterns, list) or not 2 <= len(patterns) <= 6:
        return f"pattern count {len(patterns) if isinstance(patterns, list) else None} not in 2-6"
    err = _check_patterns(patterns, reference_ngrams)
    if err:
        return err
    ids = {p["id"].strip() for p in patterns}
    seen = {}
    for a in obj.get("assignments") or []:
        pid, pat = str(a.get("pair_id", "")).strip(), str(a.get("pattern", "")).strip()
        if pid not in pair_labels:
            return f"unknown pair id {pid!r}"
        if pat not in ids:
            return f"pair {pid} assigned to unknown pattern {pat!r}"
        if pid in seen:
            return f"pair {pid} assigned more than once"
        seen[pid] = pat
    if set(seen) != pair_labels:
        return f"{len(pair_labels - set(seen))} pairs not assigned"
    overview = obj.get("domain_overview")
    if not isinstance(overview, dict) or not all(_nonempty(overview.get(k)) for k in ("topics", "forms")):
        return "incomplete domain_overview"
    return None


def validate_revision(obj, current_patterns: list, reference_ngrams: set) -> str | None:
    """Same pattern ids and count; descriptions pass the 8-gram check."""
    patterns = obj.get("patterns") if isinstance(obj, dict) else None
    if not isinstance(patterns, list) or len(patterns) != len(current_patterns):
        return "pattern count changed"
    err = _check_patterns(patterns, reference_ngrams)
    if err:
        return err
    if {p["id"].strip() for p in patterns} != {p["id"] for p in current_patterns}:
        return "pattern ids changed"
    return None


# ---------------------------------------------------------------------------------------------
# Transports. Both return, for every request, a message
#   {"ok", "http_status", "body" | "error", "attempts", "latency_s", ...}
# ---------------------------------------------------------------------------------------------

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}


def call_chat_api(base_url: str, api_key: str, payload: dict, timeout: float = 300, max_wait: float = 1800,
                  user_agent: str = "dart-analyst/1.0") -> dict:
    """POST {base_url}/chat/completions; retries network errors and retryable HTTP codes with exponential
    backoff (5 s up to 60 s) for at most `max_wait` seconds. Never raises."""
    data = json.dumps(payload).encode("utf-8")
    start, delay, attempt, err = time.time(), 5, 0, None
    while True:
        attempt += 1
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=data, method="POST",
                                     headers={"Authorization": f"Bearer {api_key}",
                                              "Content-Type": "application/json", "User-Agent": user_agent})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return {"ok": True, "http_status": r.status, "body": json.loads(r.read().decode("utf-8")),
                        "attempts": attempt}
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")[:2000]
            if e.code not in RETRY_STATUS:
                return {"ok": False, "http_status": e.code, "error": text, "attempts": attempt}
            err = f"HTTP {e.code}: {text[:300]}"
        except Exception as e:  # noqa: BLE001  network errors, timeouts, bad JSON body
            err = f"{type(e).__name__}: {e}"
        if time.time() - start > max_wait:
            return {"ok": False, "http_status": None, "error": err, "attempts": attempt}
        time.sleep(delay)
        delay = min(delay * 2, 60)


class APIClient:
    """Direct calls to an OpenAI-compatible endpoint from the training process (`backend: api`).
    The key is read from the environment variable named by `api_key_env` (default OPENAI_API_KEY);
    the base URL from `base_url` or OPENAI_BASE_URL (default https://api.openai.com/v1)."""

    def __init__(self, cfg: dict, run_id: str):
        key_env = cfg.get("api_key_env", "OPENAI_API_KEY")
        self.api_key = os.environ.get(key_env, "").strip()
        if not self.api_key:
            raise RuntimeError(f"Set the analyst API key in the environment variable {key_env}")
        self.base_url = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        self.timeout, self.max_wait = float(cfg.get("timeout", 300)), float(cfg.get("max_wait", 1800))
        self.run_id = run_id
        self.seq = itertools.count(1)
        self.lock = threading.Lock()
        self.submitted = self.completed = 0
        self.pool = ThreadPoolExecutor(max_workers=int(cfg.get("concurrency", 8)), thread_name_prefix="analyst-api")

    def _call(self, payload):
        t0 = time.time()
        msg = call_chat_api(self.base_url, self.api_key, payload, self.timeout, self.max_wait)
        msg.update({"latency_s": round(time.time() - t0, 3), "relay": "direct"})
        with self.lock:
            self.completed += 1
        return msg

    def submit(self, payload: dict, meta: dict):
        with self.lock:
            rid = f"{self.run_id}__{next(self.seq):07d}"
            self.submitted += 1
        return rid, self.pool.submit(self._call, payload)

    def relay_age(self):
        return 0.0

    def stats(self) -> dict:
        with self.lock:
            return {"submitted": self.submitted, "completed": self.completed,
                    "pending": self.submitted - self.completed}

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)


class QueueClient:
    """File queue for GPU servers without internet access (`backend: queue`). Requests are written to
    <queue_root>/req/; relay/gpt_broker.py (on the server) and relay/gpt_relay.py (on a machine with
    internet access) answer them into <queue_root>/resp/. The API key never reaches the server."""

    def __init__(self, root, run_id: str, poll: float = 0.3):
        self.root = Path(root)
        self.req, self.resp, self.done = self.root / "req", self.root / "resp", self.root / "done"
        for d in (self.req, self.resp, self.done):
            d.mkdir(parents=True, exist_ok=True)
        self.run_id, self.poll = run_id, poll
        self.seq = itertools.count(1)
        self.lock = threading.Lock()
        self.pending = {}
        self.submitted = self.completed = 0
        self._stop = False
        threading.Thread(target=self._loop, name="analyst-queue-poller", daemon=True).start()

    def submit(self, payload: dict, meta: dict):
        with self.lock:
            rid = f"{self.run_id}__{next(self.seq):07d}"
            future = Future()
            self.pending[rid] = future
            self.submitted += 1
        body = {"id": rid, "created_utc": utc(), "payload": payload, "meta": meta}
        tmp = self.req / f".{rid}.tmp"
        tmp.write_text(json.dumps(body, ensure_ascii=True), encoding="ascii")
        os.replace(tmp, self.req / f"{rid}.json")
        return rid, future

    def _loop(self):
        prefix = self.run_id + "__"
        while not self._stop:
            try:
                with os.scandir(self.resp) as entries:
                    names = [e.name for e in entries if e.name.startswith(prefix) and e.name.endswith(".json")]
            except FileNotFoundError:
                names = []
            for name in names:
                rid, path = name[:-5], self.resp / name
                try:
                    msg = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                os.replace(path, self.done / f"{rid}.resp.json")
                with self.lock:
                    future = self.pending.pop(rid, None)
                    self.completed += 1
                if future is not None and not future.done():
                    future.set_result(msg)
            time.sleep(self.poll)

    def relay_age(self):
        """Seconds since the relay was last heard from (None if never)."""
        try:
            beat = json.loads((self.root / "relay_heartbeat.json").read_text(encoding="utf-8"))
            return time.time() - float(beat["last_rx_epoch"])
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def stats(self) -> dict:
        with self.lock:
            return {"submitted": self.submitted, "completed": self.completed, "pending": len(self.pending)}

    def close(self):
        self._stop = True


def make_client(cfg: dict, run_id: str, queue_root=None):
    backend = cfg.get("backend", "api")
    if backend == "api":
        return APIClient(cfg, run_id)
    if backend == "queue":
        return QueueClient(queue_root, run_id)
    raise ValueError(f"analyst.backend must be 'api' or 'queue', got {backend!r}")


# ---------------------------------------------------------------------------------------------
# The analyst: chat calls with full logging, JSON calls with format retries
# ---------------------------------------------------------------------------------------------

class Analyst:
    """Chat calls through a client, JSON calls with at most `json_retries` format retries, full logging
    of every attempt (prompt, raw output, usage, checks) to the stage's llm_calls.jsonl. The method never
    reads these logs back."""

    def __init__(self, client, cfg: dict, on_wait=None):
        self.client, self.cfg, self.on_wait = client, cfg, on_wait
        self.log_lock = threading.Lock()
        self.log_path = None
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def set_log(self, path):
        self.log_path = Path(path)

    def log(self, record: dict):
        record = {"utc": utc(), **record}
        with self.log_lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _wait(self, future, purpose):
        while True:
            try:
                return future.result(timeout=60)
            except FutureTimeout:
                if self.on_wait is not None:
                    self.on_wait(purpose, self.client.relay_age(), self.client.stats())

    def chat(self, purpose: str, prompt: str, *, temperature: float, max_tokens: int, schema=None, meta=None) -> dict:
        payload = {"model": self.cfg["model"], "messages": [{"role": "user", "content": prompt}],
                   "temperature": temperature, self.cfg.get("max_tokens_field", "max_tokens"): max_tokens}
        if schema is not None:
            payload["response_format"] = {"type": "json_schema",
                                          "json_schema": {"name": purpose, "strict": True, "schema": schema}}
        resubmits = 0
        while True:
            rid, future = self.client.submit(payload, {"purpose": purpose, **(meta or {})})
            msg = self._wait(future, purpose)
            relay = {k: msg.get(k) for k in ("http_status", "attempts", "latency_s", "relay_t0", "relay_t1", "relay")}
            body = msg.get("body")
            if msg.get("ok") and isinstance(body, dict):
                choice = (body.get("choices") or [{}])[0]
                usage = body.get("usage") or {}
                with self.log_lock:
                    self.usage["calls"] += 1
                    self.usage["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                    self.usage["completion_tokens"] += int(usage.get("completion_tokens") or 0)
                return {"rid": rid, "content": (choice.get("message") or {}).get("content"),
                        "finish_reason": choice.get("finish_reason"), "usage": usage, "model": body.get("model"),
                        "relay": relay, "payload": payload, "resubmits": resubmits, "error": None}
            status = msg.get("http_status")
            self.log({"purpose": purpose, "rid": rid, "meta": meta, "transport_error": msg.get("error"),
                      "relay": relay})
            if status is not None and 400 <= int(status) < 500 and int(status) not in (408, 409, 425, 429):
                return {"rid": rid, "content": None, "finish_reason": None, "usage": {}, "relay": relay,
                        "payload": payload, "resubmits": resubmits, "error": f"HTTP {status}: {msg.get('error')}"}
            resubmits += 1
            if resubmits > int(self.cfg.get("max_resubmits", 20)):
                raise RuntimeError(f"Analyst request kept failing after {resubmits} resubmissions: {msg.get('error')}")
            time.sleep(60)

    def json_call(self, purpose: str, prompt: str, schema: dict, validate, *, max_tokens: int, meta=None):
        """Returns (obj, n_attempts, last_error). validate(obj) -> error string or None."""
        last = None
        for attempt in range(1, 2 + int(self.cfg["json_retries"])):
            r = self.chat(purpose, prompt, temperature=self.cfg["temperature"],
                          max_tokens=max_tokens, schema=schema, meta=meta)
            obj, err = None, r["error"]
            if err is None:
                try:
                    obj = json.loads(r["content"] or "")
                except ValueError as exc:
                    err = f"json parse: {exc}"
                if err is None:
                    err = validate(obj)
            self.log({"purpose": purpose, "attempt": attempt, "meta": meta, "rid": r["rid"],
                      "params": {k: v for k, v in r["payload"].items() if k != "messages"},
                      "prompt": prompt, "output": r["content"], "finish_reason": r["finish_reason"],
                      "usage": r["usage"], "relay": r["relay"], "resubmits": r["resubmits"], "check_error": err})
            if err is None:
                return obj, attempt, None
            last = err
        return None, 1 + int(self.cfg["json_retries"]), last
