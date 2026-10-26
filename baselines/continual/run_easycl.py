#!/usr/bin/env python3
"""ER / EWC / LwF / O-LoRA / SSR / PseudoReplay on CATS.

The shell (data, prompt, classifier, LoRA, optimizer, schedule, seeds, metrics) is that of the Sequential
baseline; the continual-learning mechanisms are ports of EasyCL (algorithms.py, lwf.py) and of
generative replay (genreplay.py).

  python baselines/continual/run_easycl.py --config baselines/continual/configs/er.yaml --order O1
  python baselines/continual/run_easycl.py --config baselines/continual/configs/er.yaml --order O1 --smoke 2
"""

import argparse
import gc
import json
import logging
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "common"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from datasets import Dataset, concatenate_datasets  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from transformers import Trainer, set_seed  # noqa: E402

from cl_common import (collator, file_binding, guard_output, load_config, predict_logits,  # noqa: E402
                       prepare_config, resolve_path, shell_namespace, strip_ids, training_args, utc,
                       validate_shell)
from data import LABELS, fixed_prompt_encoder, inspect_data, load_data, save_json  # noqa: E402
from metrics import evaluate_domains, summarize  # noqa: E402
from model import create_model  # noqa: E402
from algorithms import EWCState, OLoRAState  # noqa: E402
from lwf import TEACHER_COLUMN, LwFState  # noqa: E402
from genreplay import GenReplayState  # noqa: E402

LOGGER = logging.getLogger("easycl_cats")
METHODS = ("er", "ewc", "olora", "lwf", "ssr", "pseudo")


class CLTrainer(Trainer):
    """HF Trainer exactly as the Sequential baseline, plus one CL loss term."""

    def __init__(self, *args, cl_state=None, **kwargs):
        self.cl_state = cl_state
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        teacher = inputs.pop(TEACHER_COLUMN, None)
        outputs = model(**inputs)
        loss = outputs.loss
        state = self.cl_state
        if isinstance(state, (OLoRAState, EWCState)):
            loss = loss + state.penalty(model)
        elif isinstance(state, LwFState) and state.enabled and model.training:
            if teacher is None:
                raise RuntimeError("LwF enabled but batch has no teacher logits")
            loss = loss + state.loss(outputs.logits, teacher, inputs["labels"])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss {float(loss.detach())}")
        return (loss, outputs) if return_outputs else loss


class TeacherCollator:
    """DataCollatorWithPadding for the text fields; teacher logits stacked separately
    (the tokenizer's pad() drops non-model-input keys in transformers 5)."""

    def __init__(self, tokenizer):
        self.base = collator(tokenizer)

    def __call__(self, features):
        teacher = None
        if features and TEACHER_COLUMN in features[0]:
            teacher = torch.tensor([f[TEACHER_COLUMN] for f in features], dtype=torch.float32)
            features = [{k: v for k, v in f.items() if k != TEACHER_COLUMN} for f in features]
        batch = self.base(features)
        if teacher is not None:
            batch[TEACHER_COLUMN] = teacher
        return batch


class PairReplay:
    """ER: for every previous domain, add `replay_pairs` real HWT-AI pairs from its train split on top of
    the full current domain; no replacement, same epochs. data.py expands each pair as [HWT, AI]
    consecutively, so pair p -> rows 2p, 2p+1."""

    def __init__(self, replay_pairs: int):
        self.replay_pairs = int(replay_pairs)

    def select(self, dataset):
        if len(dataset) % 2:
            raise ValueError("Expected an even number of rows (HWT/AI pairs)")
        pairs = sorted(random.sample(range(len(dataset) // 2), self.replay_pairs))
        rows = [r for p in pairs for r in (2 * p, 2 * p + 1)]
        picked = dataset.select(rows)
        ids, labels = list(picked["sample_id"]), [int(x) for x in picked["labels"]]
        for i in range(0, len(rows), 2):
            if labels[i:i + 2] != [0, 1] or ids[i].split(":")[0] != ids[i + 1].split(":")[0]:
                raise RuntimeError(f"Replay rows are not an aligned HWT/AI pair: {ids[i:i + 2]}")
        return picked, pairs


def attach_fresh_lora(base_model, targets, shell):
    """Same LoraConfig as model.create_model, on the merged backbone (O-LoRA)."""
    config = LoraConfig(task_type=TaskType.SEQ_CLS, r=shell["lora_rank"],
                        lora_alpha=shell["lora_alpha"], lora_dropout=shell["lora_dropout"],
                        target_modules=targets, modules_to_save=["score"], bias="none")
    model = get_peft_model(base_model, config)
    model.resolved_lora_targets = targets
    return model


def fisher_batches(trainer, dataset, batch_size):
    from torch.utils.data import DataLoader
    loader = DataLoader(strip_ids(dataset), batch_size=batch_size, shuffle=True,
                        collate_fn=trainer.data_collator, num_workers=0)
    for batch in loader:
        yield trainer._prepare_inputs(batch)


@torch.no_grad()
def score_texts(model, encode_fn, texts, tokenizer, bs=8):
    """s(x) = logit_AI - logit_HWT and the classifier-head input (last-token feature), eval mode."""
    coll = collator(tokenizer)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    feats, scores = [], []
    hook = base.score.register_forward_pre_hook(lambda m, a: feats.append(a[0].detach().float().cpu()))
    was = model.training
    model.eval()
    try:
        enc = encode_fn(texts)
        items = [{"input_ids": a, "attention_mask": b} for a, b in zip(enc["input_ids"], enc["attention_mask"])]
        for i in range(0, len(items), bs):
            batch = coll(items[i:i + bs])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=batch["input_ids"].cuda(), attention_mask=batch["attention_mask"].cuda()).logits.float()
            scores.append((lg[:, 1] - lg[:, 0]).cpu())
    finally:
        hook.remove()
        if was:
            model.train()
    return torch.cat(scores).numpy(), torch.cat(feats).numpy()


def make_state(method, mcfg):
    if method == "er":
        return PairReplay(int(mcfg["replay_pairs"]))
    if method == "ewc":
        if int(mcfg.get("fisher_batch_size", 1)) != 1:
            raise ValueError("EWC Fisher batch_size must be 1 (EasyCL setting)")
        return EWCState(float(mcfg["ewc_lambda"]), int(mcfg["fisher_samples"]))
    if method == "olora":
        return OLoRAState(float(mcfg["orthogonal_lambda"]), float(mcfg["l2_lambda"]))
    if method in ("ssr", "pseudo"):
        mcfg = dict(mcfg)
        if mcfg.get("simcse_path"):
            mcfg["simcse_path"] = str(resolve_path(mcfg["simcse_path"]))
        return GenReplayState(method, mcfg)
    if method == "lwf":
        return LwFState(float(mcfg["lwf_temperature"]), float(mcfg["lwf_alpha"]))
    raise ValueError(method)


def run(cfg: dict, method: str, smoke: int = 0) -> None:
    if method not in METHODS or cfg.get("method_name") != method:
        raise ValueError(f"Config method_name={cfg.get('method_name')!r} does not match --method {method}")
    validate_shell(cfg, skip=("epochs",) if smoke else ())
    shell, mcfg = cfg["shell"], cfg["method"]
    order = list(cfg["domain_order"])
    output = Path(cfg["output_dir"])
    if smoke:
        output = output.parent / f"_smoke_{output.name}"
        order = order[:smoke]
    guard_output(output)
    data_dir = Path(cfg["data_dir"])
    prepared, data_audit = inspect_data(data_dir)
    save_json(output / "intent.json", {"created_utc": utc(), "method": method, "argv": sys.argv,
                                        "data_audit": data_audit, "smoke": smoke})

    set_seed(shell["seed"])  # seed before the classifier-head and LoRA initialisation
    model, tokenizer = create_model(shell_namespace(cfg))
    targets = model.resolved_lora_targets
    data = load_data(data_dir, list(cfg["domain_order"]), tokenizer, shell["max_length"], prepared)
    raw_train = {d: [{"text": r["text"], "labels": int(r.get("labels", r.get("label", 0)))} for r in prepared[d]["train"]]
                 for d in cfg["domain_order"]} if method in ("ssr", "pseudo") else {}
    encode_fn = fixed_prompt_encoder(tokenizer, shell["max_length"])[0] if method in ("ssr", "pseudo") else None
    if smoke:
        for d in data:
            data[d] = {k: v.select(range(min(len(v), 64 if k == "train" else 40))) for k, v in data[d].items()}
        shell = {**shell, "epochs": 1}
    del prepared
    state = make_state(method, mcfg)
    code_files = [HERE / n for n in ("run_easycl.py", "lwf.py", "algorithms.py", "genreplay.py")] + \
                 [REPO_ROOT / "common" / n for n in ("cl_common.py", "data.py", "metrics.py", "model.py")]
    save_json(output / "run_config.json", {
        **cfg, "domain_order_effective": order, "label_map": LABELS, "data_audit": data_audit,
        "code": {str(p.relative_to(REPO_ROOT)): file_binding(p) for p in code_files},
        "resolved_lora_targets": targets, "resolved_lora_target_count": len(targets),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "optimizer_state_between_stages": "reset (new Trainer per stage, as Sequential)",
        "checkpoint_selection": "fixed epochs; final checkpoint; no test selection",
        "evaluation_protocol": "metrics.evaluate_domains on all nine domains after every stage",
    })

    history = []
    for index, domain in enumerate(order):
        stage = f"stage_{index:02d}_{domain}"
        stage_dir = output / stage
        stage_seed = shell["seed"] + index  # seed + index per stage, as Sequential
        set_seed(stage_seed)
        save_json(output / "status.json", {"phase": "preparing", "stage": stage, "updated_utc": utc()})

        if isinstance(state, OLoRAState) and index > 0:
            model = attach_fresh_lora(model, targets, shell)

        train_dataset = strip_ids(data[domain]["train"])
        stage_extra = {}

        if isinstance(state, PairReplay) and index > 0:
            parts, selection = [], {}
            for d in order[:index]:
                picked, pairs = state.select(data[d]["train"])
                selection[d] = {"pair_indices": pairs, "sample_ids": list(picked["sample_id"])}
                parts.append(strip_ids(picked))
            stage_extra["replay_selection"] = selection
            stage_extra["replay_pairs_total"] = state.replay_pairs * index
            train_dataset = concatenate_datasets([train_dataset, *parts])

        if isinstance(state, GenReplayState) and index > 0:  # add pseudo samples of earlier domains
            rows = state.replay_rows(order[:index], stage_seed)
            if rows:
                enc = encode_fn([r["text"] for r in rows])
                cols = {k: list(v) for k, v in enc.items() if k in train_dataset.column_names}
                cols["labels"] = [int(r["labels"]) for r in rows]
                pseudo = Dataset.from_dict(cols).cast(train_dataset.features)
                train_dataset = concatenate_datasets([train_dataset, pseudo])
                stage_extra["pseudo_replay"] = {"n": len(rows), "ai": sum(int(r["labels"]) for r in rows),
                                                "by_domain": {d: sum(1 for r in rows if r.get("domain") == d) for d in order[:index]}}

        if isinstance(state, LwFState):
            state.enabled = index > 0
            if state.enabled:
                teacher = predict_logits(model, tokenizer, data[domain]["train"], shell,
                                         stage_dir / "_teacher_tmp", stage_seed)
                train_dataset = train_dataset.add_column(TEACHER_COLUMN, teacher.tolist())
                stage_extra["lwf_teacher"] = {"n": int(teacher.shape[0]),
                                              "source": "end-of-previous-stage model, eval mode"}
            set_seed(stage_seed)

        LOGGER.info("%s %s: train=%d extra=%s", method, stage, len(train_dataset),
                    {k: (len(v) if hasattr(v, "__len__") else v) for k, v in stage_extra.items()})
        save_json(stage_dir / "stage_setup.json", {"stage": stage, "stage_seed": stage_seed,
                                                    "train_size": len(train_dataset), **stage_extra})
        trainer = CLTrainer(model=model, train_dataset=train_dataset, processing_class=tokenizer,
                            data_collator=TeacherCollator(tokenizer),
                            args=training_args(shell, stage_dir, stage_seed), cl_state=state)
        save_json(output / "status.json", {"phase": "training", "stage": stage, "updated_utc": utc()})
        train_metrics = trainer.train().metrics
        trained = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)

        adapter = stage_dir / "adapter_final"
        trainer.save_model(str(adapter))

        if isinstance(state, EWCState):
            ok = state.compute_fisher(trained, fisher_batches(trainer, data[domain]["train"],
                                                              int(mcfg.get("fisher_batch_size", 1))))
            if not ok:
                raise RuntimeError(f"EWC Fisher failed at {stage}")
            torch.save(state.state_dict_cpu(), stage_dir / "ewc_state.pt")
            trained.train()
        if isinstance(state, OLoRAState):
            state.after_task(trained)  # snapshot A before the merge (EasyCL order)
            torch.save(state.state_dict_cpu(), stage_dir / "olora_history.pt")

        save_json(output / "status.json", {"phase": "evaluating", "stage": stage, "updated_utc": utc()})
        domain_metrics = evaluate_domains(trainer, data, list(cfg["domain_order"]), stage, output,
                                          shell["target_fpr"])
        history.append({"stage_index": index, "stage": stage, "train_domains": [domain],
                        "train_metrics": train_metrics, "domain_metrics": domain_metrics})
        result = {"method": method, "domain_order": order, "history": history}
        if index == len(order) - 1 and not smoke:
            result["summary"], result["performance_matrices"] = summarize(
                history, order, True, shell["target_fpr"])
        save_json(output / "continual_metrics.json", result)

        if isinstance(state, GenReplayState) and index < len(order) - 1:  # pseudo samples of D_t before deletion
            save_json(output / "status.json", {"phase": "generating_pseudo", "stage": stage, "updated_utc": utc()})
            trained.to("cpu")
            torch.cuda.empty_cache()
            samples = state.generate(cfg["model_name_or_path"], raw_train[domain], stage_seed, log=LOGGER.info)
            trained.to("cuda")
            if method == "ssr":
                samples = state.ssr_refine_select(samples, lambda texts: score_texts(trained, encode_fn, texts, tokenizer), stage_seed)
            for s_ in samples:
                s_["domain"] = domain
            state.memories[domain] = samples
            with (stage_dir / "pseudo_samples.jsonl").open("w", encoding="utf-8") as fh:
                for s_ in samples:
                    fh.write(json.dumps(s_, ensure_ascii=False) + "\n")
            save_json(stage_dir / "pseudo_summary.json", {"n": len(samples), "ai": sum(int(s_["labels"]) for s_ in samples)})
        raw_train.pop(domain, None)  # D_t raw text is gone after this stage
        if isinstance(state, OLoRAState):
            model = trained.merge_and_unload(progressbar=False, safe_merge=False)
        else:
            model = trained
        trainer.accelerator.free_memory()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()

    save_json(output / "completion.json", {"completed_utc": utc(), "method": method,
                                            "stage_count": len(history), "smoke": smoke,
                                            "result": file_binding(output / "continual_metrics.json")})
    save_json(output / "status.json", {"phase": "completed", "updated_utc": utc()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--order", choices=("O0", "O1", "O2", "O3"))
    ap.add_argument("--smoke", type=int, default=0,
                    help="N>0: tiny subsets, first N stages, 1 epoch, written next to the output as _smoke_<order>")
    ap.add_argument("--data_dir")
    ap.add_argument("--model_name_or_path")
    ap.add_argument("--output_dir")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = prepare_config(load_config(args.config), args.order, args.data_dir, args.model_name_or_path,
                         args.output_dir)
    run(cfg, cfg["method_name"], args.smoke)


if __name__ == "__main__":
    main()
