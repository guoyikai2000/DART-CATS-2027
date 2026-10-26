"""Generative-replay baselines (no real old data kept): PseudoReplay and SSR, adapted to detection.

pseudo : EasyCL Pseudo Replay (LAMOL-style): the frozen base LLM writes pseudo samples from few-shot
         examples of the task just learned; the label is part of the generation (label-conditioned prompts,
         num_shots=5 examples of one class -> one new text of that class), temperature 1.0.
ssr    : EasyCL SSR (Huang et al., ACL 2024): in-context generation of inputs by the frozen base LLM
         (num_shots=2 mixed examples, temperature 0.9) -> labels from the fine-tuned detector -> KMeans
         (n_clusters=20) diversity selection of pseudo_sample_memory samples (round-robin over clusters by
         distance to the centroid until the memory is full) on SimCSE embeddings.
Both: the pseudo samples of domain t are generated at the end of stage t while D_t still exists (examples
from D_t train), then D_t is deleted; at every later stage a fixed replay budget (200 samples) is drawn
evenly from the stored pseudo memories and added to the current training set. Generator = the base
checkpoint of the detector backbone (Ministral-3-8B-Instruct, no LoRA), chat template, top_p 1.0.
"""
from __future__ import annotations
import json, random, re
import numpy as np
import torch

_BYTE_DEC = None


def _bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]; n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def fix_bytelevel(text: str) -> str:
    """the local tokenizer (fix_mistral_regex) decodes to byte-level symbols (e.g. 'Ġ'); map them back to UTF-8"""
    global _BYTE_DEC
    if "Ġ" not in text and "Ċ" not in text:
        return text
    if _BYTE_DEC is None:
        _BYTE_DEC = {v: k for k, v in _bytes_to_unicode().items()}
    try:
        return bytes([_BYTE_DEC[c] for c in text if c in _BYTE_DEC]).decode("utf-8", errors="ignore")
    except Exception:
        return text.replace("Ġ", " ").replace("Ċ", "\n")


def _clip(t: str, words: int) -> str:
    w = t.split()
    return " ".join(w[:words])


class GenReplayState:
    def __init__(self, method: str, mcfg: dict):
        self.method = method
        self.budget = int(mcfg.get("replay_per_stage", 200))
        self.memory = int(mcfg.get("pseudo_sample_memory", 200))
        self.shots = int(mcfg.get("num_shots", 5 if method == "pseudo" else 2))
        self.temperature = float(mcfg.get("generation_temperature", 1.0 if method == "pseudo" else 0.9))
        self.n_clusters = int(mcfg.get("n_clusters", 20))
        self.candidate_factor = float(mcfg.get("candidate_factor", 2.0))  # ssr: candidates generated before selection
        self.max_new_tokens = int(mcfg.get("max_new_tokens", 400))
        self.demo_words = int(mcfg.get("demo_words", 220))
        self.gen_batch = int(mcfg.get("gen_batch", 16))
        self.memories: dict[str, list[dict]] = {}
        # generator = "base": the base checkpoint, as in the original papers
        self.generator = str(mcfg.get("generator", "base"))
        if self.generator != "base":
            raise ValueError("Only generator: base is supported")
        # SSR selection embeddings: "simcse" (sup-simcse-bert-base-uncased, as in SSR) or "detector"
        self.select_embed = str(mcfg.get("select_embed", "detector"))
        self.simcse_path = mcfg.get("simcse_path", "models/sup-simcse-bert-base-uncased")

    # ------------------------------------------------------------------ replay
    def replay_rows(self, history, seed):
        if not history or not self.memories:
            return []
        rng = random.Random(seed)
        n = len(history); base, extra = divmod(self.budget, n)
        rows = []
        for j, d in enumerate(history):
            pool = self.memories.get(d, [])
            k = min(len(pool), base + (1 if j < extra else 0))
            rows += rng.sample(pool, k)
        return rows

    # ------------------------------------------------------------------ generation
    def _prompts(self, rows, rng, n):
        prompts, labels = [], []
        human = [r for r in rows if int(r["labels"]) == 0]; ai = [r for r in rows if int(r["labels"]) == 1]
        for i in range(n):
            if self.method == "pseudo":
                lab = i % 2
                pool = ai if lab else human
                demos = rng.sample(pool, min(self.shots, len(pool)))
                kind = "AI-generated" if lab else "human-written"
                body = "\n\n".join(f"Text {k + 1}:\n{_clip(d['text'], self.demo_words)}" for k, d in enumerate(demos))
                msg = (f"Below are examples of {kind} texts from one source.\n\n{body}\n\n"
                       f"Write one new {kind} text of the same kind, on a new topic, with similar length and style. "
                       f"Output only the text.")
                labels.append(lab)
            else:
                demos = rng.sample(rows, min(self.shots, len(rows)))
                body = "\n\n".join(f"Text {k + 1}:\n{_clip(d['text'], self.demo_words)}" for k, d in enumerate(demos))
                msg = (f"Create a text sample following the examples below.\n\n{body}\n\n"
                       f"Write one new text of the same kind. Output only the text.")
                labels.append(None)
            prompts.append([{"role": "user", "content": msg}])
        return prompts, labels

    @torch.no_grad()
    def generate(self, model_path, rows, seed, log=print):
        from transformers import AutoTokenizer, AutoModelForImageTextToText
        n = self.memory if self.method == "pseudo" else int(round(self.memory * self.candidate_factor))
        rng = random.Random(seed)
        prompts, labels = self._prompts(rows, rng, n)
        tok = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True, local_files_only=True)
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        gen = AutoModelForImageTextToText.from_pretrained(model_path, dtype=torch.bfloat16, local_files_only=True).cuda().eval()
        torch.manual_seed(seed)
        out = []
        for s in range(0, n, self.gen_batch):
            enc = tok.apply_chat_template(prompts[s:s + self.gen_batch], add_generation_prompt=True, return_tensors="pt",
                                          padding=True, return_dict=True).to("cuda")
            o = gen.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=True, temperature=self.temperature,
                             top_p=1.0, pad_token_id=tok.pad_token_id)
            for seq in o:
                out.append(fix_bytelevel(tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)).strip())
            log(f"generated {len(out)}/{n}")
        del gen
        torch.cuda.empty_cache()
        samples = [{"text": t, "labels": l} for t, l in zip(out, labels) if len(t.split()) >= 20]
        return samples

    def ssr_refine_select(self, samples, score_fn, seed):
        """labels from the fine-tuned detector (s > 0 -> AI); KMeans diversity selection on detector features"""
        from sklearn.cluster import KMeans
        texts = [s["text"] for s in samples]
        scores, feats = score_fn(texts)
        if self.select_embed == "simcse":
            feats = self._simcse(texts)
        for s, v in zip(samples, scores):
            s["labels"] = int(v > 0); s["refine_score"] = float(v)
        X = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-12)
        k = min(self.n_clusters, len(samples), self.memory)
        km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
        dist = np.linalg.norm(X - km.cluster_centers_[km.labels_], axis=1)
        per = {c: sorted(np.where(km.labels_ == c)[0].tolist(), key=lambda i: dist[i]) for c in range(k)}
        chosen, r = [], 0
        while len(chosen) < min(self.memory, len(samples)):
            added = False
            for c in range(k):
                if r < len(per[c]):
                    chosen.append(per[c][r]); added = True
                    if len(chosen) >= self.memory:
                        break
            if not added:
                break
            r += 1
        return [samples[i] for i in chosen]

    @torch.no_grad()
    def _simcse(self, texts):
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained(self.simcse_path, local_files_only=True)
        m = AutoModel.from_pretrained(self.simcse_path, local_files_only=True).cuda().eval()
        embs = []
        for s in range(0, len(texts), 64):
            enc = tok(texts[s:s + 64], padding=True, truncation=True, max_length=256, return_tensors="pt").to("cuda")
            embs.append(m(**enc).pooler_output.float().cpu())  # sup-SimCSE: [CLS] with the MLP pooler
        del m; torch.cuda.empty_cache()
        return torch.cat(embs).numpy()
