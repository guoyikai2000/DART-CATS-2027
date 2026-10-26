#!/usr/bin/env python3
"""Server end of the analyst relay (standard library only), for GPU servers without internet access.

The training process (method.analyst.backend: queue) writes each request as a JSON file into
<queue_root>/req/. analyst_relay.py, running on a machine with internet access, starts this script
over SSH:

    ssh <server> "<python> -u dart/relay/analyst_broker.py --root <queue_root>"

stdout -> relay: one ASCII JSON line per pending request ({"type": "request", ...}) and a ping every 20 s.
stdin  <- relay: one JSON line per response ({"type": "response", "id": ...}) or {"type": "alive"}.
Responses are written atomically to <root>/resp/<id>.json; the answered request moves to <root>/done/.
The server itself never opens a network connection, and the API key never reaches it.
"""

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    req, resp, done = root / "req", root / "resp", root / "done"
    for d in (req, resp, done):
        d.mkdir(parents=True, exist_ok=True)
    state = {"last_rx": time.time(), "responses": 0}
    out_lock = threading.Lock()

    def emit(obj):
        line = json.dumps(obj, ensure_ascii=True) + "\n"
        with out_lock:
            sys.stdout.write(line)
            sys.stdout.flush()

    def beat():
        tmp = root / f".relay_heartbeat.{os.getpid()}.tmp"
        tmp.write_text(json.dumps({"utc": utc(), "pid": os.getpid(), "last_rx_epoch": state["last_rx"],
                                   "responses_this_session": state["responses"]}), encoding="ascii")
        os.replace(tmp, root / "relay_heartbeat.json")

    def reader():
        for raw in sys.stdin:
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            state["last_rx"] = time.time()
            if msg.get("type") != "response":
                continue
            rid = str(msg.get("id", ""))
            source = req / f"{rid}.json"
            if not rid or "/" in rid or not source.exists():
                continue  # unknown or already answered (duplicate delivery)
            tmp = resp / f".{rid}.tmp"
            tmp.write_text(json.dumps(msg, ensure_ascii=True), encoding="ascii")
            os.replace(tmp, resp / f"{rid}.json")
            try:
                os.replace(source, done / f"{rid}.req.json")
            except FileNotFoundError:
                pass
            state["responses"] += 1
        os._exit(0)  # relay closed the session

    threading.Thread(target=reader, daemon=True).start()
    emit({"type": "hello", "pid": os.getpid(), "root": str(root), "utc": utc()})
    sent, last_ping = set(), 0.0
    while True:
        try:
            names = sorted(e.name for e in os.scandir(req) if e.name.endswith(".json") and not e.name.startswith("."))
        except FileNotFoundError:
            names = []
        for name in names:
            rid = name[:-5]
            if rid in sent:
                continue
            try:
                body = json.loads((req / name).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            emit({"type": "request", "id": rid, "payload": body["payload"], "meta": body.get("meta", {})})
            sent.add(rid)
        now = time.time()
        if now - last_ping >= 20:
            emit({"type": "ping", "pending": len(names), "utc": utc()})
            beat()
            last_ping = now
        if now - state["last_rx"] > 180:  # relay silent for 3 min: this session is stale
            os._exit(3)
        time.sleep(0.3)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        os._exit(0)
