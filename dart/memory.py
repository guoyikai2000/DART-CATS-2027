"""Replay-text writing (Section 4.3) and the verified difference memory (Section 4.2).

write_mirror: the writer G continues the opening h_{1:m} of a human text under one pattern of a memory;
  the output is checked, cleaned like the CATS AI texts and cut to the human text's visible length.
write_memory: observation pairs -> analyst memory M_t -> R revision rounds on the worst-reproduced cases;
  the version with the highest F = harmonic mean of Cov and Prec (SimCSE space) is frozen.

Model access is injected: detector_scores(texts) -> s(x) = logit(AI) - logit(HWT) of the current detector;
embed(texts) -> frozen SimCSE embeddings; visible_fn(text) -> detector-visible text.
"""

import json
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import analyst as A
import text_utils as T
import textproc as P


def draw_pattern(patterns: list, rng, exclude=None):
    """Draw a pattern id with probability w_k (patterns with w=0 are never drawn)."""
    cands = [p for p in patterns if p["q"] > 0 and p["id"] != exclude]
    if not cands:
        return None
    w = np.array([p["q"] for p in cands], dtype=np.float64)
    return cands[int(rng.choice(len(cands), p=w / w.sum()))]["id"]


TEMPLATE_CONT = """Continue the text below. It is the opening of a text; write how it goes on.

Opening:
<<<
{opening}
>>>

The continuation differs from human writing as follows:
- Length and structure: {length_structure}
- Style: {style}
- Form: {form_diff}
- Sentences: {sentence}
- Elaboration: {elaboration}
- Wording: {wording}

Length of the continuation: about {N} words (between {lo} and {hi}).

Output only the continuation itself, starting right after the opening. Do not repeat the opening. Do not add a
preface such as "Sure" or "Here is", a closing remark or notes."""


def continuation_prompt(opening, pattern, n, lo, hi):
    d = pattern["differences"]
    return A.fill(TEMPLATE_CONT, {"opening": opening, "length_structure": d["length_structure"], "style": d["style"],
                                  "form_diff": d["form"], "sentence": d["sentence"], "elaboration": d["elaboration"],
                                  "wording": d["wording"], "N": n, "lo": lo, "hi": hi})


# a leading label ("Continuation (...):") and leading ellipsis/dashes that the writer sometimes adds
_CONT_LABEL = re.compile(r"^\s*[^\n:]{0,60}continuation[^\n:]{0,60}:\s*", re.I)
_CONT_LEAD = re.compile(r"^[\s.…\-–—]+")


def clean_continuation(text: str) -> str:
    text = _CONT_LABEL.sub("", text)
    return _CONT_LEAD.sub("", text).strip()


def write_mirror(writer, log, mcfg: dict, anchor_text: str, anchor_visible_words: int, anchor_ngrams: set,
                 patterns: list, pattern_id: str, switch_rng, meta: dict) -> dict:
    """Write -> check; the same pattern up to `mirror_rewrites` more times, then one write with a different
    pattern drawn by w (`mirror_switch_pattern`). Returns text=None when every attempt failed."""
    by_id = {p["id"]: p for p in patterns}
    wcfg = mcfg["writer"]
    n_same = 1 + int(mcfg["mirror_rewrites"])
    attempts = []
    for i in range(n_same + int(mcfg["mirror_switch_pattern"])):
        if i < n_same:
            pid = pattern_id
        else:
            pid = draw_pattern(patterns, switch_rng, exclude=pattern_id)
            if pid is None:
                break
        pattern = by_id[pid]
        # write a bit longer than the anchor, then cut to exactly the anchor's visible word count
        ratio = max(float(pattern["length_ratio"]), 1.0) * float(mcfg["mirror_overshoot"])
        n, lo, hi = T.target_length(anchor_visible_words, ratio)
        opening = " ".join(anchor_text.split()[:int(mcfg["opening_words"])])
        prompt = continuation_prompt(opening, pattern, n, lo, hi)
        r = writer.chat(prompt, float(wcfg["temperature"]), int(wcfg["max_new_tokens"]))
        text = clean_continuation((r["content"] or "").strip())
        failed = {}
        if not text:
            failed["empty_output"] = r.get("error") or "empty"
        else:
            hit = T.boilerplate(text)
            if hit:
                failed["opening_boilerplate"] = hit
            hit = T.shared_ngram(text, anchor_ngrams)
            if hit:
                failed["ngram8_with_anchor"] = hit
            # "lower": accept 0.8N-3N words, since the text is cut to the anchor length below anyway
            lower_only = mcfg.get("length_check", "lower") == "lower" and 0.8 * n <= T.words(text) <= 3 * n
            if not T.length_ok(text, n) and not lower_only:
                failed["length"] = {"words": T.words(text), "target": n}
        record = {"attempt": i + 1, "pattern": pid, "target_words": n, "range": [lo, hi],
                  "words": T.words(text), "failed_checks": failed, "rid": r["rid"]}
        log({"purpose": "mirror", "attempt": i + 1, "meta": meta, "rid": r["rid"],
             "params": {"temperature": float(wcfg["temperature"]), "max_new_tokens": int(wcfg["max_new_tokens"])},
             "pattern": pid, "target_words": n, "prompt": prompt, "output": r["content"],
             "finish_reason": r["finish_reason"], "usage": r["usage"], "latency_s": r.get("latency_s"),
             "failed_checks": failed})
        attempts.append(record)
        if not failed:
            # same processing as the CATS AI texts: format unification, AI-side delete-only cleanup,
            # typography, then the cut to the anchor's visible word count (both texts end mid-text)
            t2 = P.normalize(P.clean_ai(P.fmt(text)))
            if T.words(t2) <= anchor_visible_words:
                failed["too_short_for_trim"] = {"words": T.words(t2), "anchor_words": anchor_visible_words}
                record["failed_checks"] = failed
            else:
                record["trimmed_from_words"] = T.words(t2)
                text = P.cut(t2, anchor_visible_words)
        if not failed:
            return {"text": text, "pattern": pid, "attempts": attempts, "failure": None,
                    "switched_pattern": pid != pattern_id}
    return {"text": None, "pattern": None, "attempts": attempts, "failure": "checks"}


def unit(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def knn_radius(X: np.ndarray, k: int) -> np.ndarray:
    """epsilon_k(x): cosine distance from x to its k-th nearest other point of X (self excluded)."""
    D = 1.0 - X @ X.T
    np.fill_diagonal(D, np.inf)
    return np.sort(D, axis=1)[:, k - 1]


def coverage_precision(S: np.ndarray, X: np.ndarray, hard: np.ndarray, r: np.ndarray):
    """Eq. (4): Cov over the hard real AI texts (rows `hard` of X), Prec over the generated texts S,
    F = harmonic mean."""
    if len(S) == 0:
        return 0.0, 0.0, 0.0
    within = (1.0 - S @ X.T) <= r[None, :]
    cov = float(within[:, hard].any(axis=0).mean())
    prec = float(within.any(axis=1).mean())
    f = 0.0 if cov + prec == 0 else 2 * cov * prec / (cov + prec)
    return cov, prec, f


def _check_memory_ngrams(patterns, train_ngrams):
    for p in patterns:  # the frozen memory keeps no 8-gram of the training texts
        for text in [p["name"], *p["differences"].values()]:
            hit = T.shared_ngram(text, train_ngrams)
            if hit:
                raise RuntimeError(f"selected memory overlaps training text: {hit!r}")


def write_memory(*, analyst, writer, mcfg: dict, domain: str, pairs: list, detector_scores, embed, visible_fn,
                 train_ngrams: set, stage_seed: int, out_dir: Path, workers: int, on_phase=None) -> dict:
    """Write M_t while D_t is still available. Raises if the observation call fails after its retries."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    phase = on_phase or (lambda *_: None)
    verify = mcfg.get("verification", True) is not False
    n = len(pairs)
    # observation pairs: the n_hard pairs with the smallest margin under theta_t plus n_random random pairs
    phase("s5_scoring_train")
    s_h = detector_scores([p["h_text"] for p in pairs])
    s_a = detector_scores([p["a_text"] for p in pairs])
    f_a = embed([p["a_text"] for p in pairs]) if verify else None
    delta = s_a - s_h
    order = np.argsort(delta, kind="stable")
    hard = [int(i) for i in order[:int(mcfg["observe_hard"])]]
    rest = sorted(set(range(n)) - set(hard))
    rng = np.random.default_rng([stage_seed, 2])
    rand = sorted(int(i) for i in rng.choice(rest, size=int(mcfg["observe_random"]), replace=False))
    observed = hard + rand
    scoring = sorted(set(range(n)) - set(observed))  # held-out reference set D_t^ref
    shown = [int(i) for i in rng.permutation(observed)]
    label = {i: f"p{j + 1:02d}" for j, i in enumerate(shown)}
    hard_set = set(hard)
    blocks = [A.pair_block(label[i], "HARD" if i in hard_set else "RANDOM", pairs[i]["h_vis"], pairs[i]["a_vis"])
              for i in shown]
    phase("s5_observe")
    obj, attempts, err = analyst.json_call("observe", A.observation_prompt(blocks), A.OBSERVE_SCHEMA,
                                           lambda o: A.validate_observation(o, set(label.values()), train_ngrams),
                                           max_tokens=int(analyst.cfg["max_tokens_observe"]),
                                           meta={"domain_step": "S5.1"})
    if obj is None:
        raise RuntimeError(f"S5 observation failed after {attempts} attempts: {err}")
    assigned = {str(a["pair_id"]).strip(): str(a["pattern"]).strip() for a in obj["assignments"]}
    patterns = []
    for p in obj["patterns"]:
        pid = p["id"].strip()
        members = [i for i in observed if assigned[label[i]] == pid]
        ratios = [pairs[i]["a_words"] / max(1, pairs[i]["h_words"]) for i in members]
        patterns.append({"id": pid, "name": p["name"].strip(),
                         "differences": {k: p["differences"][k].strip() for k in A.DIMS},
                         "q": len(members) / len(observed),
                         "length_ratio": float(statistics.median(ratios)) if ratios else None,
                         "n_pairs": len(members), "n_hard": sum(1 for i in members if i in hard_set)})
    overview = obj["domain_overview"]
    (out_dir / "observation.json").write_text(json.dumps({
        "hard_pairs": [pairs[i]["pair_id"] for i in hard], "random_pairs": [pairs[i]["pair_id"] for i in rand],
        "shown_order": [[label[i], pairs[i]["pair_id"], "HARD" if i in hard_set else "RANDOM"] for i in shown],
        "assignments": assigned, "patterns": patterns, "domain_overview": overview, "attempts": attempts},
        ensure_ascii=False, indent=2), encoding="utf-8")

    def frozen(selected):
        return [{"id": p["id"], "name": p["name"], "differences": p["differences"], "q": p["q"],
                 "length_ratio": p["length_ratio"]} for p in selected]

    if not verify:  # ablation: the initial memory is kept, with no regeneration, Cov/Prec or revision
        _check_memory_ngrams(patterns, train_ngrams)
        rounds = [{"round": 0, "Cov": None, "Prec": None, "F": None,
                   "n_mirrors": 0, "n_failed": 0, "verification": "disabled"}]
        (out_dir / "revisions.json").write_text(json.dumps({
            "rounds": rounds, "selected_round": 0, "versions": [patterns], "diagnostics": [],
            "verification": "disabled"}, ensure_ascii=False, indent=2), encoding="utf-8")
        return {"domain": domain, "patterns": frozen(patterns), "domain_overview": overview, "rounds": rounds,
                "selected_round": 0, "verification": "disabled"}

    # reference: X = real AI texts of D_t^ref, hard = the hard_ai of them with the smallest margin
    X = unit(f_a[scoring])
    hard_ai = np.argsort(delta[scoring], kind="stable")[:int(mcfg["hard_ai"])]
    radius = knn_radius(X, int(mcfg["knn_k"]))
    A_obs = {i: unit(f_a[[i]])[0] for i in observed}
    # pattern draws: one per (anchor, slot), reused in every round so that versions are compared fairly
    draw_rng = np.random.default_rng([stage_seed, 3])
    draws = [(i, j, draw_pattern(patterns, draw_rng)) for i in observed for j in range(int(mcfg["mirrors_per_anchor"]))]

    versions = [patterns]
    rounds, diagnostics = [], []
    for rr in range(int(mcfg["revisions"]) + 1):
        current = versions[rr]
        phase(f"s5_round{rr}_mirrors")

        def job(d):
            i, j, pid = d
            return write_mirror(writer, analyst.log, mcfg, pairs[i]["h_vis"], pairs[i]["h_words"],
                                pairs[i]["h_ngrams"], current, pid, np.random.default_rng([stage_seed, 4, i, j]),
                                {"domain_step": f"S5.2 round {rr}", "anchor_pair": pairs[i]["pair_id"], "slot": j})

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(job, draws))
        ok = [(d, res) for d, res in zip(draws, results) if res["text"]]
        phase(f"s5_round{rr}_scoring")
        if ok:
            S = unit(embed([res["text"] for _, res in ok]))
            cov, prec, f = coverage_precision(S, X, hard_ai, radius)
            own = np.array([1.0 - float(S[k] @ A_obs[d[0]]) for k, (d, _) in enumerate(ok)])
        else:
            S, own = np.zeros((0, X.shape[1])), np.zeros(0)
            cov = prec = f = 0.0
        rounds.append({"round": rr, "Cov": cov, "Prec": prec, "F": f, "n_mirrors": len(ok),
                       "n_failed": len(results) - len(ok)})
        with (out_dir / f"round{rr}_mirrors.jsonl").open("w", encoding="utf-8") as handle:
            for d, res in zip(draws, results):
                handle.write(json.dumps({"anchor_pair": pairs[d[0]]["pair_id"], "slot": d[1], "drawn_pattern": d[2],
                                         **res}, ensure_ascii=False) + "\n")
        if rr < int(mcfg["revisions"]):
            phase(f"s5_revise{rr + 1}")
            new = current
            if ok:  # the generated texts farthest from the real AI text of their own pair
                worst = np.argsort(-own, kind="stable")[:int(mcfg["revision_cases"])]
                cases = []
                for c, k in enumerate(worst):
                    d, res = ok[int(k)]
                    i = d[0]
                    cases.append(A.case_block(f"c{c + 1:02d}", res["pattern"], pairs[i]["h_vis"], pairs[i]["a_vis"],
                                              visible_fn(res["text"])))
                obj, attempts, err = analyst.json_call(
                    "revise", A.revision_prompt(current, cases), A.REVISE_SCHEMA,
                    lambda o: A.validate_revision(o, current, train_ngrams),
                    max_tokens=int(analyst.cfg["max_tokens_revise"]), meta={"domain_step": f"S5.4 revision {rr + 1}"})
                if obj is not None:
                    by_id = {p["id"].strip(): p for p in obj["patterns"]}
                    new = [{**p, "name": by_id[p["id"]]["name"].strip(),
                            "differences": {k: by_id[p["id"]]["differences"][k].strip() for k in A.DIMS}}
                           for p in current]
                    note = {"case_notes": obj.get("case_notes"), "attempts": attempts}
                else:
                    note = {"kept_previous": True, "attempts": attempts, "error": err}
                note["cases"] = [{"anchor_pair": pairs[ok[int(k)][0][0]]["pair_id"], "distance": float(own[int(k)]),
                                  "pattern": ok[int(k)][1]["pattern"]} for k in worst]
            else:
                note = {"kept_previous": True, "error": "no mirrors in this round"}
            diagnostics.append({"revision": rr + 1, **note})
            versions.append(new)
    best = max(range(len(rounds)), key=lambda k: (rounds[k]["F"], -k))  # ties -> earliest round
    selected = versions[best]
    _check_memory_ngrams(selected, train_ngrams)
    (out_dir / "revisions.json").write_text(json.dumps({"rounds": rounds, "selected_round": best,
                                                        "versions": versions, "diagnostics": diagnostics},
                                                       ensure_ascii=False, indent=2), encoding="utf-8")
    return {"domain": domain, "patterns": frozen(selected), "domain_overview": overview, "rounds": rounds,
            "selected_round": best}
