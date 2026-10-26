#!/usr/bin/env python3
"""CPU-only checks that need no model weights, GPU or API key (python tests/test_offline.py):

1. the CATS files match data/CATS/manifest.json (sizes, hashes, ids, splits, content fingerprint);
2. the analyst client against a local mock of an OpenAI-compatible endpoint, both transports
   (direct API and file queue + analyst_broker.py, with this script acting as the relay);
3. one complete memory write (observation, 2 revision rounds, Cov/Prec/F, selection) with a fake writer,
   fake detector scores and fake embeddings.
"""

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(REPO / "dart"))

import data  # noqa: E402
import analyst as A  # noqa: E402
import memory as M  # noqa: E402
import text_utils as T  # noqa: E402


def fake_observation(prompt: str) -> dict:
    labels = sorted({line.split("]")[0].split()[-1] for line in prompt.splitlines() if line.startswith("[pair p")})
    patterns = [{"id": f"P{k}", "name": f"pattern {k}",
                 "differences": {d: f"generic description {k} of {d}" for d in A.DIMS}} for k in (1, 2)]
    return {"patterns": patterns,
            "assignments": [{"pair_id": lab, "pattern": "P1" if j % 3 else "P2"} for j, lab in enumerate(labels)],
            "domain_overview": {"topics": "various topics", "forms": "short texts"}}


def fake_revision(prompt: str) -> dict:
    current = json.loads(prompt.split("Current patterns:\n", 1)[1].split("\n\nCases:", 1)[0])
    return {"case_notes": [{"case_id": "c01", "A_has_M_lacks": "x", "M_has_A_lacks": "y"}],
            "patterns": [{**p, "name": p["name"] + " (revised)"} for p in current]}


class MockAPI(BaseHTTPRequestHandler):
    calls = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        MockAPI.calls.append(body)
        assert self.headers["Authorization"] == "Bearer test-key"
        prompt = body["messages"][0]["content"]
        purpose = body.get("response_format", {}).get("json_schema", {}).get("name")
        content = json.dumps(fake_observation(prompt) if purpose == "observe" else fake_revision(prompt))
        out = json.dumps({"model": body["model"], "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 10, "completion_tokens": 5}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


class FakeWriter:
    """Returns a continuation of the requested length built from a fixed vocabulary."""

    def __init__(self):
        self.calls = 0
        self.rng = np.random.default_rng(0)
        self.lock = threading.Lock()

    def chat(self, prompt, temperature, max_tokens=None):
        n = int(prompt.split("Length of the continuation: about ")[1].split(" ")[0])
        with self.lock:
            self.calls += 1
            words = self.rng.choice(["alpha", "beta", "gamma", "delta", "omega", "sigma", "kappa"], size=n + 5)
        return {"content": " ".join(words) + ".", "finish_reason": "stop", "usage": {}, "rid": f"w{self.calls}"}


class OfflineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), MockAPI)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}/v1"
        cls.prepared, cls.audit = data.inspect_data(REPO / "data" / "CATS")

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def analyst_cfg(self, backend):
        return {"backend": backend, "model": "mock", "temperature": 0.2, "json_retries": 2, "max_resubmits": 2,
                "max_tokens_observe": 100, "max_tokens_revise": 100, "base_url": self.base_url,
                "api_key_env": "DART_TEST_KEY", "concurrency": 2, "timeout": 10, "max_wait": 10}

    def test_1_data(self):
        self.assertEqual(self.audit["pair_count"], 8100)
        self.assertEqual(self.audit["content_fingerprint"], data.CONTENT_FINGERPRINT)
        d04 = self.prepared["D04"]["train"]
        self.assertEqual(len(d04), 1000)
        self.assertEqual([r["labels"] for r in d04[:4]], [0, 1, 0, 1])

    def test_2_api_client(self):
        os.environ["DART_TEST_KEY"] = "test-key"
        client = A.make_client(self.analyst_cfg("api"), "test_api")
        llm = A.Analyst(client, self.analyst_cfg("api"))
        with tempfile.TemporaryDirectory() as tmp:
            llm.set_log(Path(tmp) / "calls.jsonl")
            blocks = [A.pair_block(f"p{j:02d}", "HARD", "human text here", "ai text here") for j in (1, 2, 3)]
            obj, attempts, err = llm.json_call("observe", A.observation_prompt(blocks), A.OBSERVE_SCHEMA,
                                               lambda o: A.validate_observation(o, {"p01", "p02", "p03"}, set()),
                                               max_tokens=100)
        client.close()
        self.assertIsNone(err)
        self.assertEqual(attempts, 1)
        self.assertEqual(len(obj["patterns"]), 2)
        self.assertEqual(MockAPI.calls[-1]["response_format"]["json_schema"]["strict"], True)

    def test_3_queue_and_broker(self):
        """QueueClient <-> analyst_broker.py <-> (this test as the relay) <-> mock API."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "queue"
            client = A.make_client(self.analyst_cfg("queue"), "test_queue", root)
            broker = subprocess.Popen([sys.executable, "-u", str(REPO / "dart" / "relay" / "analyst_broker.py"),
                                       "--root", str(root)], stdin=subprocess.PIPE, stdout=subprocess.PIPE)

            def relay():
                for raw in broker.stdout:
                    msg = json.loads(raw)
                    if msg.get("type") == "request":
                        res = A.call_chat_api(self.base_url, "test-key", msg["payload"], 10, 10)
                        res.update({"type": "response", "id": msg["id"], "relay": "test"})
                        broker.stdin.write((json.dumps(res) + "\n").encode())
                        broker.stdin.flush()

            threading.Thread(target=relay, daemon=True).start()
            llm = A.Analyst(client, self.analyst_cfg("queue"))
            llm.set_log(Path(tmp) / "calls.jsonl")
            blocks = [A.pair_block(f"p{j:02d}", "RANDOM", "human text", "ai text") for j in (1, 2)]
            obj, attempts, err = llm.json_call("observe", A.observation_prompt(blocks), A.OBSERVE_SCHEMA,
                                               lambda o: A.validate_observation(o, {"p01", "p02"}, set()),
                                               max_tokens=100)
            client.close()
            broker.kill()
            broker.wait()
            broker.stdin.close()
            broker.stdout.close()
            self.assertIsNone(err)
            self.assertEqual(len(obj["patterns"]), 2)
            self.assertEqual(len(list((root / "done").glob("*.resp.json"))), 1)

    def test_4_memory_write(self):
        os.environ["DART_TEST_KEY"] = "test-key"
        rows = self.prepared["D04"]["train"][:2 * 30]
        pairs = []
        for p in range(30):
            h, a = rows[2 * p]["text"], rows[2 * p + 1]["text"]
            pairs.append({"pair_id": p + 1, "h_text": h, "a_text": a, "h_vis": h.strip(), "a_vis": a.strip(),
                          "h_words": T.words(h), "a_words": T.words(a), "h_ngrams": T.ngrams(h)})
        train_ngrams = set().union(*(T.ngrams(r["text"]) for r in rows))
        rng = np.random.default_rng(1)
        scores = {}

        def detector_scores(texts):
            return np.array([scores.setdefault(t, float(rng.normal())) for t in texts])

        def embed(texts):
            return np.stack([np.random.default_rng(abs(hash(t)) % (2 ** 32)).normal(size=16) for t in texts])

        mcfg = {"observe_hard": 3, "observe_random": 3, "hard_ai": 6, "knn_k": 3, "mirrors_per_anchor": 2,
                "revisions": 2, "revision_cases": 3, "verification": True, "mirror_rewrites": 2,
                "mirror_switch_pattern": 1, "mirror_overshoot": 1.15, "length_check": "lower",
                "opening_words": 30, "writer": {"temperature": 0.9, "max_new_tokens": 1024}}
        client = A.make_client(self.analyst_cfg("api"), "test_memory")
        llm = A.Analyst(client, self.analyst_cfg("api"))
        writer = FakeWriter()
        with tempfile.TemporaryDirectory() as tmp:
            llm.set_log(Path(tmp) / "calls.jsonl")
            memory = M.write_memory(analyst=llm, writer=writer, mcfg=mcfg, domain="D04", pairs=pairs,
                                    detector_scores=detector_scores, embed=embed, visible_fn=lambda t: t,
                                    train_ngrams=train_ngrams, stage_seed=2026, out_dir=Path(tmp) / "memory",
                                    workers=4)
            revisions = json.loads((Path(tmp) / "memory" / "revisions.json").read_text())
            self.assertEqual(len(revisions["versions"]), 3)
            self.assertTrue(revisions["versions"][1][0]["name"].endswith("(revised)"))
            # the replay side: one anchor of a later stage, written from this memory
            res = M.write_mirror(writer, llm.log, mcfg, pairs[0]["h_vis"], pairs[0]["h_words"], pairs[0]["h_ngrams"],
                                 memory["patterns"], memory["patterns"][0]["id"], np.random.default_rng(0), {})
        client.close()
        self.assertEqual(len(memory["rounds"]), 3)
        self.assertIn(memory["selected_round"], (0, 1, 2))
        self.assertAlmostEqual(sum(p["q"] for p in memory["patterns"]), 1.0)
        self.assertEqual(writer.calls >= 3 * 6 * 2, True)
        self.assertIsNotNone(res["text"])
        self.assertEqual(T.words(res["text"]), pairs[0]["h_words"])  # cut to the anchor's visible length


if __name__ == "__main__":
    unittest.main(verbosity=2)
