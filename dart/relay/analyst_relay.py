#!/usr/bin/env python3
r"""Client end of the analyst relay: runs on a machine WITH internet access (e.g. a workstation).

It keeps one SSH session to the training server, receives the queued analyst requests from
analyst_broker.py, calls the API (N at a time) and sends each response back. If the connection drops,
it reconnects; requests that were not answered are sent again by the server.

    export OPENAI_API_KEY=...            # read here only; never sent to the server
    python analyst_relay.py --host <ssh-alias-of-the-server> \
        --remote-python <python on the server> \
        --remote-broker <repo on the server>/dart/relay/analyst_broker.py \
        --remote-queue <repo on the server>/outputs/analyst_queue \
        [--base-url https://api.openai.com/v1]

The SSH alias must log in without a password prompt (key-based authentication). A one-line record per
call (no texts) is appended to relay_logs/relay_YYYYMMDD.jsonl next to this file; the full prompts and
outputs are logged on the server.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def say(msg):
    print(f"[{now()}] {msg}", flush=True)


def load_key(env_name):
    key = os.environ.get(env_name, "").strip()
    if not key:
        sys.exit(f"No API key: set the environment variable {env_name}")
    return key


class Relay:
    def __init__(self, args, key):
        self.args, self.key = args, key
        self.counts = {"done": 0, "failed": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self.lock = threading.Lock()
        os.makedirs(os.path.join(HERE, "relay_logs"), exist_ok=True)

    def log(self, rec):
        path = os.path.join(HERE, "relay_logs", f"relay_{datetime.now():%Y%m%d}.jsonl")
        with self.lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def call_api(self, payload):
        data = json.dumps(payload).encode("utf-8")
        start, delay, attempt, err = time.time(), 5, 0, None
        while True:
            attempt += 1
            req = urllib.request.Request(self.args.base_url.rstrip("/") + "/chat/completions", data=data,
                                         method="POST", headers={"Authorization": f"Bearer {self.key}",
                                                                 "Content-Type": "application/json",
                                                                 "User-Agent": "dart-relay/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=self.args.timeout) as r:
                    return {"ok": True, "http_status": r.status, "body": json.loads(r.read().decode("utf-8")),
                            "attempts": attempt}
            except urllib.error.HTTPError as e:
                text = e.read().decode("utf-8", "replace")[:2000]
                if e.code not in RETRY_STATUS:
                    return {"ok": False, "http_status": e.code, "error": text, "attempts": attempt}
                err = f"HTTP {e.code}: {text[:300]}"
            except Exception as e:  # noqa: BLE001  network errors, timeouts, bad JSON body
                err = f"{type(e).__name__}: {e}"
            if time.time() - start > self.args.max_wait:
                return {"ok": False, "http_status": None, "error": err, "attempts": attempt}
            time.sleep(delay)
            delay = min(delay * 2, 60)

    def session(self, stop_at):
        remote = f"{self.args.remote_python} -u {self.args.remote_broker} --root {self.args.remote_queue}"
        cmd = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4",
               self.args.host, remote]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wlock, inflight, alive = threading.Lock(), set(), {"ok": True}

        def send(obj):
            line = (json.dumps(obj, ensure_ascii=True) + "\n").encode("ascii")
            with wlock:
                try:
                    proc.stdin.write(line)
                    proc.stdin.flush()
                    return True
                except (OSError, ValueError):
                    return False

        def stderr_pump():
            for raw in proc.stderr:
                say("ssh: " + raw.decode("utf-8", "replace").rstrip())

        def keepalive():
            while alive["ok"]:
                if not send({"type": "alive"}):
                    break
                if stop_at and time.time() > stop_at:
                    proc.kill()
                    break
                time.sleep(20)

        def handle(msg):
            t0 = time.time()
            res = self.call_api(msg["payload"])
            res.update({"type": "response", "id": msg["id"], "latency_s": round(time.time() - t0, 3),
                        "relay_t0": datetime.fromtimestamp(t0).isoformat(), "relay_t1": datetime.now().isoformat(),
                        "relay": platform.node()})
            delivered = send(res)
            usage = (res.get("body") or {}).get("usage") or {} if res["ok"] else {}
            with self.lock:
                self.counts["done" if res["ok"] else "failed"] += 1
                self.counts["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
                self.counts["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            self.log({"t": now(), "id": msg["id"], "purpose": (msg.get("meta") or {}).get("purpose"),
                      "ok": res["ok"], "http_status": res.get("http_status"), "attempts": res.get("attempts"),
                      "latency_s": res["latency_s"], "usage": usage, "delivered": delivered,
                      "error": None if res["ok"] else str(res.get("error"))[:300]})
            inflight.discard(msg["id"])

        threading.Thread(target=stderr_pump, daemon=True).start()
        threading.Thread(target=keepalive, daemon=True).start()
        pool = ThreadPoolExecutor(max_workers=self.args.concurrency)
        last_report, connected = 0.0, False
        for raw in proc.stdout:
            try:
                msg = json.loads(raw)
            except ValueError:
                say("unexpected line from server: " + raw[:200].decode("ascii", "replace"))
                continue
            kind = msg.get("type")
            if kind == "hello":
                connected = True
                say(f"connected to {self.args.host} (broker pid {msg.get('pid')})")
            elif kind == "request" and msg["id"] not in inflight:
                inflight.add(msg["id"])
                pool.submit(handle, msg)
            elif kind == "ping" and time.time() - last_report > 60:
                last_report = time.time()
                c = self.counts
                say(f"waiting on server: {msg.get('pending')} | running here: {len(inflight)} | done {c['done']} "
                    f"failed {c['failed']} | tokens in {c['prompt_tokens']} out {c['completion_tokens']}")
        alive["ok"] = False
        pool.shutdown(wait=False, cancel_futures=True)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        return connected

    def run(self):
        stop_at = time.time() + self.args.max_seconds if self.args.max_seconds else 0
        say(f"relay started: host={self.args.host} concurrency={self.args.concurrency}")
        while True:
            connected = self.session(stop_at)
            if stop_at and time.time() > stop_at:
                say("max-seconds reached, exiting")
                return
            say("connection closed" + ("" if connected else " before the server answered (check SSH)")
                + "; reconnecting in 10 s")
            time.sleep(10)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", required=True, help="SSH alias of the training server")
    ap.add_argument("--remote-python", required=True, help="python executable on the server")
    ap.add_argument("--remote-broker", required=True, help="path of analyst_broker.py on the server")
    ap.add_argument("--remote-queue", required=True, help="queue_root of the training config, as a server path")
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY", help="environment variable holding the API key")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=300, help="seconds per HTTP request")
    ap.add_argument("--max-wait", type=int, default=1800, help="seconds of retrying network errors per request")
    ap.add_argument("--max-seconds", type=int, default=0, help="stop after this many seconds (0 = run forever)")
    args = ap.parse_args()
    Relay(args, load_key(args.api_key_env)).run()


if __name__ == "__main__":
    main()
