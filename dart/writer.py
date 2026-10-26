"""The frozen writer G and the frozen sentence encoder phi of DART.

LocalWriter: the base checkpoint of the detector backbone (Ministral-3-8B-Instruct, no LoRA) loaded a
second time, frozen, in bf16. Requests from the worker threads are batched by one generation thread;
only that thread touches the model. Sampling: temperature from the request, top-p 1.0.
SimCSE: supervised SimCSE ([CLS] with the MLP pooler), used for the write-time fidelity measure.
"""

import itertools
import queue
import threading
import time

import numpy as np
import torch

_BYTE_DEC = None


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def fix_bytelevel(text):
    """The tokenizer (fix_mistral_regex) can decode to byte-level symbols ('Ġ'); map them back to UTF-8."""
    global _BYTE_DEC
    if "Ġ" not in text and "Ċ" not in text:
        return text
    if _BYTE_DEC is None:
        _BYTE_DEC = {v: k for k, v in _bytes_to_unicode().items()}
    try:
        return bytes([_BYTE_DEC[c] for c in text if c in _BYTE_DEC]).decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001
        return text.replace("Ġ", " ").replace("Ċ", "\n")


class LocalWriter:
    def __init__(self, model_path, max_new_tokens=1024, batch=16, wait=0.5, device="cuda"):
        from transformers import AutoModelForImageTextToText, AutoTokenizer
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True, local_files_only=True)
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16,
                                                                 local_files_only=True).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.max_new, self.batch, self.wait = int(max_new_tokens), int(batch), float(wait)
        self.q = queue.Queue()
        self.seq = itertools.count(1)
        self.calls = 0
        threading.Thread(target=self._loop, name="local-writer", daemon=True).start()

    def chat(self, prompt, temperature, max_tokens=None):
        """Returns {"content", "finish_reason", "usage", "rid", "latency_s"}."""
        t0 = time.time()
        box = {"ev": threading.Event()}
        limit = self.max_new if max_tokens is None else min(int(max_tokens), self.max_new)
        self.q.put((prompt, float(temperature), limit, box))
        box["ev"].wait()
        if "exc" in box:
            raise box["exc"]
        return {**box["out"], "rid": f"local_{next(self.seq):07d}", "latency_s": round(time.time() - t0, 2)}

    @torch.no_grad()
    def _run(self, reqs):
        msgs = [[{"role": "user", "content": r[0]}] for r in reqs]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", padding=True,
                                           return_dict=True).to(self.device)
        mx = max(r[2] for r in reqs)
        o = self.model.generate(**enc, max_new_tokens=mx, do_sample=True, temperature=reqs[0][1], top_p=1.0,
                                pad_token_id=self.tok.pad_token_id)
        L = enc["input_ids"].shape[1]
        for k, r in enumerate(reqs):
            gen = o[k, L:]
            n_out = int((gen != self.tok.pad_token_id).sum())
            text = fix_bytelevel(self.tok.decode(gen, skip_special_tokens=True)).strip()
            finish = "length" if n_out >= mx else "stop"
            r[3]["out"] = {"content": text, "finish_reason": finish,
                           "usage": {"prompt_tokens": int(enc["attention_mask"][k].sum()), "completion_tokens": n_out}}

    def _loop(self):
        while True:
            reqs = [self.q.get()]
            t0 = time.time()
            while len(reqs) < self.batch:
                try:
                    reqs.append(self.q.get(timeout=max(0.0, self.wait - (time.time() - t0))))
                except queue.Empty:
                    break
            groups = {}
            for r in reqs:
                groups.setdefault(r[1], []).append(r)
            for g in groups.values():
                try:
                    self._run(g)
                except Exception as exc:  # noqa: BLE001
                    for r in g:
                        r[3]["exc"] = exc
                finally:
                    for r in g:
                        r[3]["ev"].set()
                self.calls += len(g)


class SimCSE:
    def __init__(self, path, device="cuda"):
        from transformers import AutoModel, AutoTokenizer
        self.device = torch.device(device)
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.model = AutoModel.from_pretrained(path, local_files_only=True).to(self.device).eval()

    @torch.no_grad()
    def embed(self, texts):
        out = []
        texts = list(texts)
        for s in range(0, len(texts), 64):
            e = self.tok(texts[s:s + 64], padding=True, truncation=True, max_length=512,
                         return_tensors="pt").to(self.device)
            out.append(self.model(**e).pooler_output.float().cpu())  # sup-SimCSE: [CLS] with the MLP pooler
        return torch.cat(out).numpy() if out else np.zeros((0, 768), dtype=np.float32)
