#!/usr/bin/env python3
"""Fixed (non-continual) detectors on CATS: one frozen scoring pass over the valid and test splits of all
nine domains, then AA and pooled TPR at 1% FPR with the same metric code as the continual methods.

  python baselines/fixed_detectors/score_fixed.py --config baselines/fixed_detectors/configs/radar.yaml

Detectors and their native scores (higher = more likely AI):
  fast_detect_gpt  GPT-J-6B (fp16, 2048 tokens), analytic sampling discrepancy        [author function]
  binoculars       Falcon-7B observer + Falcon-7B-Instruct performer (bf16, 512), -score [author functions]
  lastde           GPT-J-6B (fp16, 2048), Lastde; window cap for texts below 8 tokens   [author functions]
  radar            RADAR-Vicuna-7B RoBERTa-large (fp32, 512), P(class 0 = AI)
  imbd             ImBD scoring adapter on GPT-Neo-2.7B (fp32, 2048), analytic Style-CPC criterion
  taste            TASTE mBERT classifier detector_ep3.pt (fp16 autocast, 512), P(class 1 = AI)
  osm_det          OSM-Det Longformer (fp32, 4096), P(class 1 = AI)
The author functions of the first three are loaded from a clone of the Lastde repository
(scripts/fetch_third_party.sh) and checked against the SHA-256 of the files used in the paper.
"""

import argparse
import ast
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "common"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from cl_common import load_config, resolve_path, utc  # noqa: E402
from data import DOMAINS, inspect_data, save_json  # noqa: E402
import metrics  # noqa: E402

# Files of https://github.com/TrustMedia-zju/Lastde_Detector used in the paper (relative to py_scripts/baselines)
THIRD_PARTY_SHA256 = {
    "fast_detect_gpt.py": "6ed45ce05bbd6a5dfa971f9806613da71378a81bf256bc46cc7d3795769e9c46",
    "binoculars.py": "fdcef8e6a7359fbfa6d1b55e9e4467657821e5d5d0c9bde1ed0e74849ebcbb83",
    "scoring_methods/based_scoring_baselines.py": "1ac0fdf71a88108cbcdf49b9e241d3adf8870a2f871a839c97f1addc244d9de9",
    "scoring_methods/fastMDE.py": "14b6549616f3a2080d1c1029e0fe0b7e13faf86f2cdc7e641befc094670c155f",
}
METHODS = ("fast_detect_gpt", "binoculars", "lastde", "radar", "imbd", "taste", "osm_det")


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_all(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def check_third_party(root: Path, names, strict: bool):
    report = {}
    for name in names:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"{path} not found; run scripts/fetch_third_party.sh first")
        got = sha256_file(path)
        report[name] = {"sha256": got, "matches_paper": got == THIRD_PARTY_SHA256[name]}
        if got != THIRD_PARTY_SHA256[name]:
            msg = f"{path} differs from the version used in the paper"
            if strict:
                raise ValueError(msg + " (pass --allow_source_mismatch to continue)")
            print("WARNING:", msg, flush=True)
    return report


def extract_functions(path: Path, names: set, env: dict):
    """Execute only the named top-level functions of an author script (avoids its CLI imports)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {f.name for f in functions} != names:
        raise ValueError(f"{path} does not define {sorted(names)}")
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), env)
    return env


def dtype_kwargs(dtype) -> dict:
    """The `dtype` keyword of transformers >= 4.56 (main environment), `torch_dtype` before (native one)."""
    import transformers
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) else {"torch_dtype": dtype}


def causal_tokenizer(path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    tok.padding_side = tok.truncation_side = "right"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


class ProbabilityScorer:
    """fast_detect_gpt / binoculars / lastde: one text at a time with the authors' scoring functions."""

    def __init__(self, cfg, sources: Path, strict: bool):
        from transformers import AutoModelForCausalLM
        self.cfg, self.method = cfg, cfg["method"]
        self.device = torch.device(cfg.get("device", "cuda:0"))
        models = {k: str(resolve_path(v)) for k, v in cfg["models"].items()}
        if self.method == "binoculars":
            import transformers
            self.provenance = check_third_party(sources, ["binoculars.py"], strict)
            self.tok = causal_tokenizer(models["observer"])
            other = causal_tokenizer(models["performer"])
            assert self.tok.get_vocab() == other.get_vocab(), "Falcon tokenizer mismatch"
            self.observer = AutoModelForCausalLM.from_pretrained(
                models["observer"], **dtype_kwargs(torch.bfloat16), local_files_only=True).to(self.device).eval()
            self.performer_device = torch.device(cfg.get("performer_device", "cuda:1"))
            self.performer = AutoModelForCausalLM.from_pretrained(
                models["performer"], **dtype_kwargs(torch.bfloat16), local_files_only=True).to(self.performer_device).eval()
            self.observer.requires_grad_(False)
            self.performer.requires_grad_(False)
            env = {"torch": torch, "np": np, "transformers": transformers, "device": str(self.device),
                   "ce_loss_fn": torch.nn.CrossEntropyLoss(reduction="none"), "softmax_fn": torch.nn.Softmax(dim=-1)}
            self.native = extract_functions(sources / "binoculars.py",
                                            {"perplexity", "entropy", "get_binoculars_score"}, env)["get_binoculars_score"]
            self.max_length = int(cfg["max_length"])
        else:
            self.tok = causal_tokenizer(models["scoring"])
            self.model = AutoModelForCausalLM.from_pretrained(
                models["scoring"], **dtype_kwargs(torch.float16 if self.device.type == "cuda" else torch.float32),
                local_files_only=True).to(self.device).eval()
            self.model.requires_grad_(False)
            self.max_length = int(self.model.config.n_positions)
            if self.max_length != int(cfg["max_length"]):
                raise ValueError(f"expected a {cfg['max_length']}-token scoring model")
            if self.method == "lastde":
                self.provenance = check_third_party(
                    sources, ["scoring_methods/based_scoring_baselines.py", "scoring_methods/fastMDE.py"], strict)
                sys.path.insert(0, str(sources))
                from scoring_methods.based_scoring_baselines import get_lastde
                from scoring_methods import fastMDE
                self.native, self.fast_mde = get_lastde, fastMDE
            else:
                self.provenance = check_third_party(sources, ["fast_detect_gpt.py"], strict)
                env = extract_functions(sources / "fast_detect_gpt.py", {"get_sampling_discrepancy_analytic"},
                                        {"torch": torch})
                self.native = env["get_sampling_discrepancy_analytic"]

    def score(self, rows, progress):
        values = []
        for index, row in enumerate(rows):
            if index % 50 == 0:
                progress(rows_done=index, rows_total=len(rows))
            inputs = self.tok(row["text"], return_tensors="pt", truncation=True,
                              max_length=self.max_length, return_token_type_ids=False).to(self.device)
            with torch.no_grad():
                if self.method == "binoculars":
                    observer = self.observer(**inputs).logits
                    performer_inputs = {k: v.to(self.performer_device) for k, v in inputs.items()}
                    performer = self.performer(**performer_inputs).logits.to(self.device)
                    value = -float(self.native(self.tok.pad_token_id, inputs, observer, performer))  # higher = AI
                else:
                    logits = self.model(**inputs).logits[:, :-1]
                    labels = inputs.input_ids[:, 1:]
                    if self.method == "lastde":
                        from lastde_short import lastde_score
                        value = lastde_score(self.native, self.fast_mde, logits, labels)
                    else:
                        value = self.native(logits, logits, labels)
                value = float(value)
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite {self.method}: {row['sample_id']}")
            values.append(value)
        return values


def imbd_discrepancy(logits, labels):
    """ImBD's analytic criterion (Style-CPC): score and reference distributions are the same LM."""
    logp = torch.log_softmax(logits, dim=-1)
    prob = torch.softmax(logits, dim=-1)
    observed = logp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    expected = (prob * logp).sum(-1)
    variance = (prob * logp.square()).sum(-1) - expected.square()
    return (observed.sum(-1) - expected.sum(-1)) / variance.sum(-1).sqrt()


class ClassifierScorer:
    """radar / imbd / taste / osm_det: released weights, batched."""

    def __init__(self, cfg):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer
        self.cfg, self.method = cfg, cfg["method"]
        self.device = torch.device(cfg.get("device", "cuda:0"))
        path = str(resolve_path(cfg["models"]["detector"]))
        self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        self.provenance = {}
        if self.method == "imbd":
            from peft import PeftModel
            base = AutoModelForCausalLM.from_pretrained(str(resolve_path(cfg["models"]["base"])), **dtype_kwargs(torch.float32),
                                                        attn_implementation="eager", local_files_only=True)
            self.model = PeftModel.from_pretrained(base, path, is_trainable=False, local_files_only=True)
        elif self.method == "taste":
            config = AutoConfig.from_pretrained(path, local_files_only=True, num_labels=2)
            # the supplied config describes the pretraining task; all classifier tensors are in the checkpoint
            self.model = AutoModelForSequenceClassification.from_config(config, attn_implementation="eager")
            state = torch.load(Path(path) / cfg.get("checkpoint", "detector_ep3.pt"), map_location="cpu",
                               weights_only=True)
            self.model.load_state_dict(state, strict=True)
        else:
            kwargs = {**dtype_kwargs(torch.float32), "local_files_only": True, "output_loading_info": True}
            if self.method == "radar":
                kwargs["attn_implementation"] = "eager"
            self.model, info = AutoModelForSequenceClassification.from_pretrained(path, **kwargs)
            assert not info.get("missing_keys") and not info.get("mismatched_keys") and not info.get("unexpected_keys"), info
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def score(self, rows, progress):
        result, batch = [], int(self.cfg["batch_size"])
        for start in range(0, len(rows), batch):
            if start % (batch * 10) == 0:
                progress(rows_done=start, rows_total=len(rows))
            selected = rows[start:start + batch]
            enc = self.tok([r["text"] for r in selected], return_tensors="pt", padding=True,
                           truncation=True, max_length=int(self.cfg["max_length"]))
            enc = {k: v.to(self.device) for k, v in enc.items()}
            if self.method == "imbd":
                enc.pop("token_type_ids", None)
            amp = torch.autocast("cuda", dtype=torch.float16) if self.method == "taste" else contextlib.nullcontext()
            with torch.inference_mode(), amp:
                if self.method == "imbd":
                    logits = self.model(**enc, use_cache=False).logits[:, :-1]
                    scores = imbd_discrepancy(logits, enc["input_ids"][:, 1:])
                else:
                    logits = self.model(**enc).logits
                    if self.method == "radar":  # class 0 = AI
                        scores = torch.nn.functional.log_softmax(logits, -1)[:, 0].exp()
                    else:
                        scores = torch.softmax(logits, -1)[:, 1]
            values = scores.float().cpu().numpy()
            assert len(values) == len(selected) and np.isfinite(values).all()
            result.extend(float(x) for x in values)
        return result


def evaluate(prepared, scorer, run: Path, target_fpr: float, smoke: bool, status):
    """Per-domain and pooled metrics; the score enters metrics.py as the logit proxy [0, score]."""
    domain_metrics, pooled = {}, {s: [[], []] for s in ("valid", "test")}
    splits = ("valid",) if smoke else ("valid", "test")
    (run / "predictions").mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        results = {}
        for split in splits:
            rows = prepared[domain][split]
            scores = np.asarray(scorer.score(rows, lambda **kw: status(domain=domain, split=split, **kw)),
                                dtype=np.float64)
            labels = np.asarray([r["labels"] for r in rows], dtype=np.int64)
            if scores.shape != labels.shape or not np.isfinite(scores).all():
                raise ValueError(f"incomplete/non-finite scores: {domain}/{split}")
            logits = np.column_stack([np.zeros(len(scores)), scores])
            metrics.checked_labels(labels, len(scores))
            results[split] = (logits, labels)
            pooled[split][0].append(logits)
            pooled[split][1].append(labels)
            with (run / "predictions" / f"fixed_{domain}_{split}.jsonl").open("w", encoding="utf-8") as stream:
                for row, score in zip(rows, scores):
                    stream.write(json.dumps({"sample_id": row["sample_id"], "label": row["labels"],
                                             "score_ai_vs_hwt": float(score)}, allow_nan=False) + "\n")
        threshold = metrics.threshold_at_fpr(metrics.binary_scores(results["valid"][0]), results["valid"][1], target_fpr)
        domain_metrics[domain] = metrics.detection_metrics(*results["valid" if smoke else "test"],
                                                           target_fpr=target_fpr, valid_threshold=threshold)
    valid_logits, valid_labels = np.concatenate(pooled["valid"][0]), np.concatenate(pooled["valid"][1])
    threshold = metrics.threshold_at_fpr(metrics.binary_scores(valid_logits), valid_labels, target_fpr)
    split = "valid" if smoke else "test"
    domain_metrics["__overall__"] = metrics.detection_metrics(
        np.concatenate(pooled[split][0]), np.concatenate(pooled[split][1]), target_fpr, threshold)
    return {"stage": "fixed", "train_domains": [], "domain_metrics": domain_metrics}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--output_dir")
    ap.add_argument("--smoke", action="store_true", help="score 8 valid texts per domain only")
    ap.add_argument("--allow_source_mismatch", action="store_true",
                    help="continue when the third-party files differ from the versions used in the paper")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if cfg.get("method") not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    run = resolve_path(args.output_dir or cfg["output_dir"])
    if args.smoke:
        run = run.parent / f"_smoke_{run.name}"
    if (run / "fixed_metrics.json").exists() and os.environ.get("OVERWRITE_OUTPUT", "0") != "1":
        raise FileExistsError(f"Results already exist in {run}; set OVERWRITE_OUTPUT=1 to overwrite")
    run.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def status(**fields):
        save_json(run / "status.json", {"updated_utc": utc(), "method": cfg["method"], **fields})

    status(phase="loading_data")
    prepared, audit = inspect_data(resolve_path(cfg.get("data_dir", "data/CATS")))
    if args.smoke:
        prepared = {d: {s: rows[:8] for s, rows in splits.items() if s == "valid"} for d, splits in prepared.items()}
    seed_all(int(cfg.get("seed", 2026)))
    status(phase="loading_model")
    if cfg["method"] in ("fast_detect_gpt", "binoculars", "lastde"):
        scorer = ProbabilityScorer(cfg, resolve_path(cfg["third_party_dir"]), strict=not args.allow_source_mismatch)
    else:
        scorer = ClassifierScorer(cfg)
    save_json(run / "run_config.json", {**cfg, "data_audit": {k: v for k, v in audit.items() if k != "files"},
                                         "third_party": scorer.provenance, "smoke": args.smoke,
                                         "torch": torch.__version__,
                                         "transformers": __import__("transformers").__version__})
    stage = evaluate(prepared, scorer, run, float(cfg.get("target_fpr", 0.01)), args.smoke,
                     lambda **kw: status(phase="scoring", **kw))
    summary, matrices = metrics.summarize([stage], list(DOMAINS), sequential=False,
                                          target_fpr=float(cfg.get("target_fpr", 0.01)))
    save_json(run / "fixed_metrics.json", {"method": cfg["method"], "history": [stage], "summary": summary,
                                           "matrices": matrices, "smoke": args.smoke,
                                           "elapsed_seconds": round(time.time() - started, 1)})
    status(phase="completed")
    print(json.dumps({"method": cfg["method"],
                      "AA_TPR@1%FPR": round(100 * summary["final_AA_empirical_test_TPR_AI_at_FPR"], 2),
                      "pooled_TPR@1%FPR": round(100 * summary["final_global_empirical_test_TPR_AI_at_FPR"], 2),
                      "pooled_AUROC": round(100 * summary["final_global_AUROC_AI_vs_HWT"], 2)}, indent=2))


if __name__ == "__main__":
    main()
