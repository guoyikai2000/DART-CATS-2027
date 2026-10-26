#!/usr/bin/env python3
"""DART: keep the difference, not the data (main method of the paper).

One stage t on the training set D_t (Algorithm 1):
  epoch 1   fine-tune theta_{t-1} with cross-entropy on the 2N texts of D_t, one text at a time, in exactly
            the order of the Sequential baseline -> theta_t^(1).
  boundary  (t > 1) a fraction beta of the pairs become anchors, split evenly among the past domains j < t.
            For each anchor h: draw a pattern k ~ Cat(w) of M_j, let the frozen writer G continue the opening
            h_{1:m} under pattern k (|C(h)| candidates), keep the valid candidate with the lowest
            s_{theta_t^(1)} (hard selection) and replace the anchor's AI text with it.
  epoch 2   N pairs, two pairs per step, loss = CE + lambda * mean max(0, gamma - s(a) + s(h)) -> theta_t^(2);
            theta_t = theta_{t-1} + alpha (theta_t^(2) - theta_{t-1}) for t > 1.
  memory    (t < T) write and verify M_t while D_t is still available (memory.write_memory).
  discard   D_t is released from the method side; only theta_t and the memories are kept.
The shell (data, prompt, classifier, LoRA, optimizer, schedule, seeds, evaluation) is shared with the
continual baselines (common/), and the detector is evaluated on all nine domains after every stage.

Usage:
  python dart/run_dart.py --config dart/configs/dart.yaml --order O1
  python dart/run_dart.py --config dart/configs/dart.yaml --order O1 --smoke   # 2 stages, tiny subsets
"""

import argparse
import gc
import json
import logging
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "common"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset as TorchDataset  # noqa: E402
from transformers import Trainer, TrainerCallback, set_seed  # noqa: E402

from cl_common import (collator, file_binding, guard_output, load_config, prepare_config, resolve_path,  # noqa: E402
                       shell_namespace, training_args, utc, validate_shell)
from data import LABELS, fixed_prompt_encoder, inspect_data, load_data, save_json  # noqa: E402
from metrics import evaluate_domains, summarize  # noqa: E402
from model import create_model  # noqa: E402

import analyst as A  # noqa: E402
import text_utils as T  # noqa: E402
from memory import draw_pattern, write_memory, write_mirror  # noqa: E402
import synthetic_human  # noqa: E402
from writer import LocalWriter, SimCSE  # noqa: E402

LOGGER = logging.getLogger("dart")
CODE_FILES = [HERE / n for n in ("run_dart.py", "analyst.py", "memory.py", "text_utils.py", "textproc.py",
                                 "writer.py", "synthetic_human.py")] + \
             [HERE.parent / "common" / n for n in ("cl_common.py", "data.py", "metrics.py", "model.py")]


# ------------------------------------------------------------------ training-side pieces

class UnitDataset(TorchDataset):
    """Plain list of feature dicts; epoch-2 items are appended at the epoch boundary."""

    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class UnitBatchSampler:
    """Packs the current unit list into batches of `texts_per_batch` texts; units are never split.
    Epoch 1: units are single rows -> randperm(2N, seed) = the order of the Sequential baseline.
    Epoch 2: units are pairs, shuffled with the same seed."""

    def __init__(self, units, texts_per_batch: int, seed: int):
        self.units, self.k, self.seed, self.epoch = list(units), texts_per_batch, seed, 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        batch = []
        for u in torch.randperm(len(self.units), generator=generator).tolist():
            batch.extend(self.units[u])
            if len(batch) == self.k:
                yield batch
                batch = []
        if batch:
            raise RuntimeError("Units do not fill whole batches")

    def __len__(self):
        return sum(len(u) for u in self.units) // self.k


class PairCollator:
    """DataCollatorWithPadding; epoch-2 items carry pair_key >= 0 -> pair_slot / pair_key tensors."""

    def __init__(self, tokenizer):
        self.base = collator(tokenizer)

    def __call__(self, features):
        keys = [int(f.get("pair_key", -1)) for f in features]
        batch = self.base([{k: v for k, v in f.items() if k != "pair_key"} for f in features])
        if any(k >= 0 for k in keys):
            if any(k < 0 for k in keys):
                raise RuntimeError("Batch mixes paired and unpaired texts")
            slot_of = {}
            batch["pair_slot"] = torch.tensor([slot_of.setdefault(k, len(slot_of)) for k in keys])
            batch["pair_key"] = torch.tensor(keys)
        return batch


class DARTTrainer(Trainer):
    """HF Trainer as in the Sequential baseline, with the unit batch sampler and loss = CE + lambda * hinge
    on paired batches (Eq. 7)."""

    def __init__(self, *args, unit_sampler=None, lam=0.0, gamma=0.0, **kwargs):
        self.unit_sampler, self.lam, self.gamma = unit_sampler, float(lam), float(gamma)
        self.pair_kind, self.hinge_records = {}, []
        super().__init__(*args, **kwargs)

    def get_train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_sampler=self.unit_sampler, collate_fn=self.data_collator,
                            num_workers=self.args.dataloader_num_workers,
                            pin_memory=self.args.dataloader_pin_memory, persistent_workers=False)
        return self.accelerator.prepare(loader)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        slots, keys = inputs.pop("pair_slot", None), inputs.pop("pair_key", None)
        outputs = model(**inputs)
        loss = outputs.loss
        if slots is not None and model.training:
            logits = outputs.logits.float()
            score = logits[:, 1] - logits[:, 0]
            labels = inputs["labels"]
            deltas, kinds = [], []
            for slot in range(int(slots.max().item()) + 1):
                in_slot = slots == slot
                ai, hwt = in_slot & (labels == 1), in_slot & (labels == 0)
                if int(ai.sum()) != 1 or int(hwt.sum()) != 1:
                    raise RuntimeError("A pair slot must hold exactly one HWT and one AI text")
                deltas.append(score[ai][0] - score[hwt][0])
                kinds.append(self.pair_kind.get(int(keys[in_slot][0]), "?"))
            delta = torch.stack(deltas)
            hinge = torch.clamp(self.gamma - delta, min=0).mean()
            if self.lam != 0.0:
                loss = loss + self.lam * hinge
            self.hinge_records.append({"step": self.state.global_step + 1, "kinds": kinds,
                                       "delta": [round(float(x), 4) for x in delta.detach().cpu()],
                                       "hinge": float(hinge.detach()), "ce": float(outputs.loss.detach())})
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss {float(loss.detach())}")
        return (loss, outputs) if return_outputs else loss


class EpochBoundary(TrainerCallback):
    def __init__(self, fn):
        self.fn, self.fired = fn, False

    def on_epoch_end(self, args, state, control, **kwargs):
        if not self.fired and state.epoch is not None and round(state.epoch) == 1:
            self.fired = True
            self.fn()


# ------------------------------------------------------------------ model access for the method

def find_score_module(model):
    candidates = [(n, m) for n, m in model.named_modules() if n.split(".")[-1] == "score"]
    if not candidates:
        raise RuntimeError("Classifier head 'score' not found")
    return min(candidates, key=lambda x: x[0].count("."))[1]


class Scorer:
    """Eval mode, no grad, bf16 autocast, batch = eval batch size. Returns s(x) = logit(AI) - logit(HWT)."""

    def __init__(self, tokenizer, encode, batch_size: int):
        self.collate, self.encode, self.bs = collator(tokenizer), encode, batch_size

    @torch.no_grad()
    def items(self, model, items):
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()
        scores = []
        try:
            for i in range(0, len(items), self.bs):
                batch = self.collate([{"input_ids": it["input_ids"], "attention_mask": it["attention_mask"]}
                                      for it in items[i:i + self.bs]])
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=batch["input_ids"].to(device),
                                   attention_mask=batch["attention_mask"].to(device)).logits.float()
                scores.append((logits[:, 1] - logits[:, 0]).cpu())
        finally:
            if was_training:
                model.train()
        return torch.cat(scores).numpy().astype(np.float64)

    def texts(self, model, texts):
        enc = self.encode(list(texts))
        items = [{"input_ids": a, "attention_mask": b} for a, b in zip(enc["input_ids"], enc["attention_mask"])]
        return self.items(model, items)


class Status:
    def __init__(self, path: Path):
        self.path, self.lock, self.base, self.last_warn = path, threading.Lock(), {}, 0.0

    def set(self, **fields):
        with self.lock:
            self.base.update(fields)
            save_json(self.path, {**self.base, "updated_utc": utc()})

    def waiting(self, purpose, relay_age, stats):
        with self.lock:
            save_json(self.path, {**self.base, "waiting_for_analyst": purpose, "relay_age_s": relay_age,
                                  "analyst_queue": stats, "updated_utc": utc()})
        if (relay_age is None or relay_age > 120) and time.time() - self.last_warn > 300:
            self.last_warn = time.time()
            LOGGER.warning("Waiting for the analyst relay (last heard %s s ago); queue %s", relay_age, stats)


# ------------------------------------------------------------------ one stage

class Stage:
    def __init__(self, *, index, domain, history, memories, items, pairs, mcfg, analyst, writer, scorer,
                 stage_dir, stage_seed, rows_per_batch, status, workers):
        self.index, self.domain, self.history, self.memories = index, domain, history, memories
        self.base_items, self.pairs, self.mcfg, self.scorer = items, pairs, mcfg, scorer
        self.analyst, self.writer = analyst, writer
        self.stage_dir, self.stage_seed, self.status, self.workers = stage_dir, stage_seed, status, workers
        self.rows_per_batch = rows_per_batch
        self.R = int(round(float(mcfg["beta"]) * len(pairs)))  # anchors (replay pairs) per stage
        self.trainer = None
        self.prev = None  # [(param, theta_{t-1} copy on CPU)] for the interpolation of Eq. (8)

    def quotas(self):
        """Even split of the R anchors over the past domains (remainder to the earliest ones)."""
        base, extra = divmod(self.R, len(self.history))
        return [base + (1 if j < extra else 0) for j in range(len(self.history))]

    # called from on_epoch_end of epoch 1
    def boundary(self):
        t0 = time.time()
        trainer, n = self.trainer, len(self.pairs)
        record = {"stage": self.index, "R": self.R, "history": self.history}
        replay, restored = [], []
        if self.history and self.R > 0:
            # anchors: R pairs drawn at random; the other pairs keep their real AI text
            order = np.random.default_rng([self.stage_seed, 6]).permutation(n)
            keep, anchors = [int(i) for i in order[:n - self.R]], [int(i) for i in order[n - self.R:]]
            quotas = self.quotas()
            rng = np.random.default_rng([self.stage_seed, 1])
            shuffled = [int(i) for i in rng.permutation(anchors)]
            assignment, pos = [], 0
            for dom, quota in zip(self.history, quotas):
                for i in shuffled[pos:pos + quota]:
                    assignment.append((i, dom, draw_pattern(self.memories[dom]["patterns"], rng)))
                pos += quota
            self.status.set(phase="epoch_boundary_mirrors")
            K = int(self.mcfg["candidates"])

            def job(a):
                (i, dom, pid), slot = a
                return write_mirror(self.writer, self.analyst.log, self.mcfg, self.pairs[i]["h_vis"],
                                    self.pairs[i]["h_words"], self.pairs[i]["h_ngrams"],
                                    self.memories[dom]["patterns"], pid,
                                    np.random.default_rng([self.stage_seed, 5, i] + ([slot] if slot else [])),
                                    {"domain_step": "S3 replay", "anchor_pair": self.pairs[i]["pair_id"],
                                     "memory_of": dom, **({"candidate": slot} if K > 1 else {})})

            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                res_all = list(pool.map(job, [(a, k) for a in assignment for k in range(K)]))
            if K == 1:
                results = res_all
            elif self.mcfg.get("selection", "hardest") == "random":
                # ablation "w/o hard selection": a valid candidate chosen uniformly at random
                results = []
                record["candidate_selection"] = "uniform_random"
                for j, (i, dom, pid) in enumerate(assignment):
                    valid = [k for k in range(K) if res_all[j * K + k]["text"]]
                    selected = (int(np.random.default_rng([self.stage_seed, 17, i]).choice(valid))
                                if valid else None)
                    candidate = res_all[j * K + (selected if selected is not None else 0)]
                    results.append({**candidate, "candidate_selection": "uniform_random",
                                    "valid_candidate_indices": valid, "selected_candidate": selected})
            else:
                # hard selection (Eq. 6): theta_t^(1) keeps the valid candidate it finds least AI-like
                ok = [(n_, r["text"]) for n_, r in enumerate(res_all) if r["text"]]
                sc = {}
                if ok:
                    s_c = self.scorer.texts(trainer.model, [x for _, x in ok])
                    sc = {n_: float(v) for (n_, _), v in zip(ok, s_c)}
                results = []
                for j in range(len(assignment)):
                    cands = [(sc[j * K + k], res_all[j * K + k]) for k in range(K) if (j * K + k) in sc]
                    if cands:
                        best = min(cands, key=lambda c: c[0])[1]
                        results.append({**best, "candidate_scores": [round(c[0], 3) for c in cands]})
                    else:
                        results.append(res_all[j * K])
            for (i, dom, pid), res in zip(assignment, results):
                (replay if res["text"] else restored).append((i, dom, pid, res))  # no valid candidate: keep (h, a)
            current = sorted(keep + [a[0] for a in restored])
            record.update({"quotas": dict(zip(self.history, quotas)),
                           "mirrors": [{"anchor_pair": self.pairs[i]["pair_id"], "memory_of": dom,
                                        "drawn_pattern": pid, **res} for i, dom, pid, res in replay + restored]})
        else:
            current = list(range(n))
        # epoch-2 units: current pairs by pair index, then replay pairs by anchor pair index
        new_items, units, kinds = [], [], {}
        offset = len(trainer.train_dataset.items)

        def add(feature, key):
            new_items.append({**feature, "pair_key": key})
            return offset + len(new_items) - 1

        for key, i in enumerate(current):
            units.append((add(self.base_items[2 * i], key), add(self.base_items[2 * i + 1], key)))
            kinds[key] = "current"
        replay.sort(key=lambda a: a[0])
        humans = (synthetic_human.replay_humans(self, replay)
                  if self.mcfg.get("replay_human", "real") == "synthetic" else None)
        if replay:
            enc = self.scorer.encode([res["text"] for _, _, _, res in replay])
            for (i, dom, pid, res), ids, mask in zip(replay, enc["input_ids"], enc["attention_mask"]):
                key = len(units)
                mirror = {"input_ids": ids, "attention_mask": mask, "labels": LABELS["AI"]}
                human = humans[i] if humans is not None else self.base_items[2 * i]
                units.append((add(human, key), add(mirror, key)))
                kinds[key] = "replay"
        if len(units) != n or len(units) * 2 % self.rows_per_batch:
            raise RuntimeError(f"Epoch-2 plan has {len(units)} units")
        trainer.train_dataset.items.extend(new_items)
        trainer.unit_sampler.units = units
        trainer.pair_kind = kinds
        record.update({"n_current_pairs": len(current), "n_replay_pairs": len(replay),
                       "n_restored_pairs": len(restored),
                       "replay_by_domain": {d: sum(1 for a in replay if a[1] == d) for d in self.history},
                       "replay_human": self.mcfg.get("replay_human", "real"),
                       "seconds": round(time.time() - t0, 1)})
        save_json(self.stage_dir / "epoch_boundary.json", record)
        LOGGER.info("%s boundary: %d current + %d replay pairs (%d restored) in %.0fs", self.domain,
                    len(current), len(replay), len(restored), time.time() - t0)
        self.status.set(phase="training_epoch2")

    def hinge_summary(self) -> dict:
        recs = self.trainer.hinge_records
        out = {"gamma": self.trainer.gamma, "lambda": self.trainer.lam, "steps": len(recs)}
        for kind in ("current", "replay"):
            ds = [d for r in recs for d, k in zip(r["delta"], r["kinds"]) if k == kind]
            if ds:
                arr = np.asarray(ds)
                out[kind] = {"n": len(ds), "frac_delta_below_gamma": float((arr < self.trainer.gamma).mean()),
                             "mean_delta": float(arr.mean()), "median_delta": float(np.median(arr))}
        out["mean_hinge"] = float(np.mean([r["hinge"] for r in recs])) if recs else 0.0
        return out


# ------------------------------------------------------------------ run

def run(cfg: dict, tag: str, smoke: bool = False) -> None:
    if cfg.get("method_name") != "dart":
        raise ValueError("config method_name must be dart")
    validate_shell(cfg)
    shell, mcfg = cfg["shell"], dict(cfg["method"])
    full_order = list(cfg["domain_order"])
    order = full_order[:2] if smoke else full_order
    smoke_cfg = mcfg.pop("smoke", {})
    if smoke:
        mcfg.update(smoke_cfg)
    if mcfg.get("selection", "hardest") not in ("hardest", "random"):
        raise ValueError("method.selection must be 'hardest' or 'random'")
    if mcfg.get("replay_human", "real") not in ("real", "synthetic"):
        raise ValueError("method.replay_human must be 'real' or 'synthetic'")
    replay_enabled = float(mcfg["beta"]) > 0
    verify = mcfg.get("verification", True) is not False
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_id = f"{'smoke_' if smoke else ''}{tag}_{stamp}"
    output = Path(cfg["output_dir"]) / run_id
    guard_output(output)
    status = Status(output / "status.json")
    status.set(phase="starting", run_id=run_id)
    data_dir = Path(cfg["data_dir"])
    prepared, data_audit = inspect_data(data_dir)
    save_json(output / "intent.json", {"created_utc": utc(), "method": "dart", "argv": sys.argv,
                                        "data_audit": data_audit, "smoke": smoke, "run_id": run_id})

    set_seed(shell["seed"])  # seed before the classifier-head and LoRA initialisation
    model, tokenizer = create_model(shell_namespace(cfg))
    targets = model.resolved_lora_targets
    data = load_data(data_dir, full_order, tokenizer, shell["max_length"], prepared)
    encode, encoder_info = fixed_prompt_encoder(tokenizer, shell["max_length"])
    budget = encoder_info["body_token_budget"]
    texts = {d: [ex["text"] for ex in prepared[d]["train"]] for d in order}
    sample_ids = {d: [ex["sample_id"] for ex in prepared[d]["train"]] for d in order}
    if smoke:
        rows, ev = 2 * int(mcfg["train_pairs"]), int(mcfg["eval_rows"])
        for d in data:
            data[d] = {k: v.select(range(min(len(v), rows if k == "train" else ev))) for k, v in data[d].items()}
        texts = {d: t[:rows] for d, t in texts.items()}
        sample_ids = {d: t[:rows] for d, t in sample_ids.items()}
    del prepared

    acfg, wcfg = mcfg["analyst"], mcfg["writer"]
    queue_root = resolve_path(acfg.get("queue_root", "outputs/analyst_queue"))
    client = A.make_client(acfg, run_id, queue_root) if replay_enabled else None
    analyst = A.Analyst(client, acfg, on_wait=status.waiting)
    writer = None
    if replay_enabled:  # the writer G: the frozen base checkpoint of the detector backbone
        writer = LocalWriter(cfg["model_name_or_path"], max_new_tokens=int(wcfg["max_new_tokens"]),
                             batch=int(wcfg.get("batch_size", 16)), device=wcfg.get("device", "cuda"))
    encoder = SimCSE(str(resolve_path(mcfg["encoder_path"]))) if (replay_enabled and verify) else None
    scorer = Scorer(tokenizer, encode, shell["eval_batch_size"])
    save_json(output / "run_config.json", {
        **cfg, "method": mcfg, "domain_order_effective": order, "label_map": LABELS, "data_audit": data_audit,
        "code": {str(p.relative_to(HERE.parent)): file_binding(p) for p in CODE_FILES},
        "encoder": encoder_info, "resolved_lora_targets": targets, "resolved_lora_target_count": len(targets),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "optimizer_state_between_stages": "reset (new Trainer per stage, as Sequential)",
        "checkpoint_selection": "fixed epochs; final checkpoint; no valid or test selection",
        "evaluation_protocol": "metrics.evaluate_domains on all nine domains after every stage",
        "analyst_backend": acfg.get("backend", "api") if replay_enabled else None, "smoke": smoke})

    memories, history, usage_by_stage = {}, [], {}
    workers = int(mcfg["workers"])
    for index, domain in enumerate(order):
        stage = f"stage_{index:02d}_{domain}"
        stage_dir = output / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_seed = shell["seed"] + index  # seed + index per stage, as the baselines
        set_seed(stage_seed)
        status.set(phase="preparing", stage=stage)
        analyst.set_log(stage_dir / "llm_calls.jsonl")
        usage_before = dict(analyst.usage)

        # the only data the method reads: this domain's train split
        rows = data[domain]["train"]
        ids = sample_ids[domain]
        items = [{"input_ids": r["input_ids"], "attention_mask": r["attention_mask"], "labels": int(r["labels"]),
                  "pair_key": -1} for r in rows]
        pairs, inexact = [], 0
        for p in range(len(items) // 2):
            if (items[2 * p]["labels"], items[2 * p + 1]["labels"]) != (0, 1) or \
                    ids[2 * p].split(":")[0] != ids[2 * p + 1].split(":")[0] or ids[2 * p] != rows[2 * p]["sample_id"]:
                raise RuntimeError(f"Rows {2 * p},{2 * p + 1} of {domain} are not an aligned HWT/AI pair")
            h_text, a_text = texts[domain][2 * p], texts[domain][2 * p + 1]
            h_vis, ok_h = T.visible_text(h_text, tokenizer, budget)
            a_vis, ok_a = T.visible_text(a_text, tokenizer, budget)
            inexact += (not ok_h) + (not ok_a)
            pairs.append({"pair_id": int(ids[2 * p].split(":")[0]), "h_text": h_text, "a_text": a_text,
                          "h_vis": h_vis, "a_vis": a_vis, "h_words": T.words(h_vis), "a_words": T.words(a_vis),
                          "h_ngrams": T.ngrams(h_text)})
        train_ngrams = set().union(*(T.ngrams(t) for t in texts[domain]))
        st = Stage(index=index, domain=domain, history=order[:index], memories=memories, items=items, pairs=pairs,
                   mcfg=mcfg, analyst=analyst, writer=writer, scorer=scorer, stage_dir=stage_dir,
                   stage_seed=stage_seed, rows_per_batch=shell["train_batch_size"], status=status, workers=workers)
        save_json(stage_dir / "stage_setup.json", {
            "stage": stage, "stage_seed": stage_seed, "train_rows": len(items), "pairs": len(pairs), "R": st.R,
            "history": order[:index], "visible_text_inexact_reencodings": inexact,
            "visible_truncated_human": sum(1 for p in pairs if p["h_vis"] != p["h_text"].strip()),
            "visible_truncated_ai": sum(1 for p in pairs if p["a_vis"] != p["a_text"].strip())})

        if index > 0 and mcfg.get("merge_alpha") is not None:
            st.prev = [(q, q.detach().to("cpu", copy=True)) for q in model.parameters() if q.requires_grad]
        sampler = UnitBatchSampler([(i,) for i in range(len(items))], shell["train_batch_size"], stage_seed)
        trainer = DARTTrainer(model=model, train_dataset=UnitDataset(list(items)), processing_class=tokenizer,
                              data_collator=PairCollator(tokenizer), args=training_args(shell, stage_dir, stage_seed),
                              unit_sampler=sampler, lam=mcfg["lambda"], gamma=mcfg["gamma"])
        trainer.add_callback(EpochBoundary(st.boundary))
        st.trainer = trainer
        status.set(phase="training_epoch1")
        train_metrics = trainer.train().metrics
        trained = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
        if mcfg.get("merge_alpha") is not None and st.prev is not None:  # Eq. (8)
            a_ = float(mcfg["merge_alpha"])
            with torch.no_grad():
                for q, old in st.prev:
                    o = old.to(q.device, q.dtype)
                    q.data.copy_(o + a_ * (q.data - o))
        adapter = stage_dir / "adapter_final"
        trainer.save_model(str(adapter))
        save_json(stage_dir / "log_history.json", {"log_history": trainer.state.log_history})
        save_json(stage_dir / "hinge_stats.json", {**st.hinge_summary(), "records": trainer.hinge_records})

        status.set(phase="evaluating")
        domain_metrics = evaluate_domains(trainer, data, full_order, stage, output, shell["target_fpr"])
        history.append({"stage_index": index, "stage": stage, "train_domains": [domain],
                        "train_metrics": train_metrics, "domain_metrics": domain_metrics})
        result = {"method": "dart", "domain_order": order, "history": history}
        if index == len(order) - 1 and not smoke:
            result["summary"], result["performance_matrices"] = summarize(history, order, True, shell["target_fpr"])
        save_json(output / "continual_metrics.json", result)

        # write the memory of this domain while its data still exist (not needed for the last domain)
        if index < len(order) - 1 and replay_enabled:
            memory = write_memory(
                analyst=analyst, writer=writer, mcfg=mcfg, domain=domain, pairs=pairs,
                detector_scores=lambda texts_: scorer.texts(trained, texts_),
                embed=(lambda texts_: encoder.embed(texts_)) if encoder is not None else None,
                visible_fn=lambda text: T.visible_text(text, tokenizer, budget)[0], train_ngrams=train_ngrams,
                stage_seed=stage_seed, out_dir=stage_dir / "memory", workers=workers,
                on_phase=lambda name: status.set(phase=name))
            memories[domain] = memory
            save_json(output / "memories" / f"{domain}.json", memory)
            LOGGER.info("%s memory: rounds %s, selected %d", domain,
                        [None if r["F"] is None else round(r["F"], 3) for r in memory["rounds"]], memory["selected_round"])
        usage_by_stage[stage] = {k: analyst.usage[k] - usage_before.get(k, 0) for k in analyst.usage}
        usage_by_stage[stage]["writer_calls_total"] = writer.calls if writer is not None else 0
        save_json(output / "analyst_usage.json", {"by_stage": usage_by_stage, "total": analyst.usage})

        # this domain's train data are gone from the method side
        data[domain]["train"] = None
        texts.pop(domain)
        sample_ids.pop(domain)
        del st, items, pairs, train_ngrams, rows
        model = trained
        trainer.accelerator.free_memory()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    if client is not None:
        client.close()
    save_json(output / "completion.json", {"completed_utc": utc(), "method": "dart", "stage_count": len(history),
                                            "smoke": smoke, "result": file_binding(output / "continual_metrics.json"),
                                            "analyst_usage": analyst.usage})
    status.set(phase="completed")
    LOGGER.info("finished: %s", output)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="YAML config, e.g. dart/configs/dart.yaml")
    ap.add_argument("--order", choices=("O0", "O1", "O2", "O3"), help="domain order of the paper")
    ap.add_argument("--tag", default="dart", help="run name prefix; the run goes to <output_dir>/<order>/<tag>_<time>")
    ap.add_argument("--smoke", action="store_true", help="first 2 stages, tiny subsets, scaled-down memory writing")
    ap.add_argument("--data_dir")
    ap.add_argument("--model_name_or_path")
    ap.add_argument("--output_dir")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = prepare_config(load_config(args.config), args.order, args.data_dir, args.model_name_or_path,
                         args.output_dir)
    run(cfg, args.tag, args.smoke)


if __name__ == "__main__":
    main()
