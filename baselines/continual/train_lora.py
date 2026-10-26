#!/usr/bin/env python3
"""Sequential (lower bound) and Joint (upper bound) dense LoRA on CATS.

Sequential fine-tunes on one domain after another with cross-entropy only; Joint trains once on the
union of all nine training sets (shuffled with the seed). Both share the detector, prompt, LoRA and
optimizer settings of all continual methods.

  python baselines/continual/train_lora.py sequential_lora --order O1
  python baselines/continual/train_lora.py joint_lora --order O0
Relative paths are resolved against the repository root.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "common"))

from datasets import concatenate_datasets  # noqa: E402
from transformers import (DataCollatorWithPadding, Trainer, TrainerCallback,  # noqa: E402
                          TrainingArguments, set_seed)

from cl_common import ORDERS, resolve_path  # noqa: E402
from data import DOMAINS, LABELS, inspect_data, load_data, save_json  # noqa: E402
from metrics import evaluate_domains, summarize  # noqa: E402
from model import create_model  # noqa: E402

DEFAULT_MODEL = "models/Ministral-3-8B-Instruct-2512-BF16"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_binding(path: Path) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def process_identity() -> dict:
    stat = Path(f"/proc/{os.getpid()}/stat")
    start_ticks = None
    if stat.exists():
        start_ticks = int(stat.read_text().rsplit(")", 1)[1].split()[19])
    return {"pid": os.getpid(), "start_ticks": start_ticks, "argv": sys.argv.copy()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("method", choices=("sequential_lora", "joint_lora"))
    parser.add_argument("--order", choices=sorted(ORDERS), default=None,
                        help="domain order of the paper (Joint: the concatenation order before shuffling, O0)")
    parser.add_argument("--domain_order", default=None, help="explicit comma-separated order instead of --order")
    parser.add_argument("--data_dir", default="data/CATS")
    parser.add_argument("--output_dir", default=None, help="default: outputs/<method>/<order>")
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--lora_rank", type=int, default=24)
    parser.add_argument("--lora_alpha", type=int, default=48)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", default="gate_proj,up_proj,down_proj")
    parser.add_argument("--precision", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--target_fpr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dataloader_workers", type=int, default=4)
    parser.add_argument("--smoke", type=int, default=0,
                        help="N>0: tiny subsets, first N stages, 1 epoch, written next to the output as _smoke_<name>")
    args = parser.parse_args()
    if args.domain_order:
        domains = [domain.strip() for domain in args.domain_order.split(",")]
        name = "custom"
    else:
        name = args.order or ("O0" if args.method == "joint_lora" else None)
        if name is None:
            parser.error("give --order (O0-O3) or --domain_order")
        domains = list(ORDERS[name])
    if len(domains) != len(DOMAINS) or set(domains) != set(DOMAINS):
        parser.error("domain_order must contain D00-D08 exactly once")
    if args.method == "sequential_lora" and domains[0] != "D00":
        parser.error("Sequential LoRA domain_order must start with D00")
    if (args.epochs <= 0 or args.learning_rate <= 0 or args.max_length <= 0
            or min(args.train_batch_size, args.eval_batch_size,
                   args.gradient_accumulation_steps, args.lora_rank, args.lora_alpha) <= 0
            or not 0 <= args.warmup_ratio <= 1 or not 0 <= args.lora_dropout < 1
            or not 0 < args.target_fpr < 1 or args.dataloader_workers < 0):
        parser.error("Invalid training argument range")
    args.domain_order = domains
    args.order = name
    args.data_dir = resolve_path(args.data_dir)
    args.model_name_or_path = str(resolve_path(args.model_name_or_path))
    args.output_dir = resolve_path(args.output_dir or f"outputs/{args.method}/{name}")
    if args.smoke:
        args.output_dir = args.output_dir.parent / f"_smoke_{args.output_dir.name}"
        args.epochs = 1
    return args


class Progress(TrainerCallback):
    def __init__(self, path: Path, context: dict):
        self.path, self.context = path, context

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero:
            save_json(self.path, {**self.context, "updated_utc": utc(), "phase": "training",
                                  "global_step": state.global_step, "max_steps": state.max_steps,
                                  "epoch": state.epoch, "logs": logs or {}})


def build_trainer(model, tokenizer, dataset, args, output_dir: Path, seed: int,
                  progress: Progress) -> Trainer:
    return Trainer(
        model=model, train_dataset=dataset, processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8),
        callbacks=[progress],
        args=TrainingArguments(
            output_dir=str(output_dir), num_train_epochs=args.epochs,
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            weight_decay=args.weight_decay, warmup_ratio=args.warmup_ratio,
            lr_scheduler_type="cosine", optim="adamw_torch",
            bf16=args.precision == "bf16", fp16=args.precision == "fp16",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            ddp_find_unused_parameters=False, dataloader_num_workers=args.dataloader_workers,
            label_names=["labels"], eval_strategy="no", save_strategy="no",
            logging_steps=10, logging_first_step=True, report_to="none",
            seed=seed, data_seed=seed,
        ),
    )


def main() -> None:
    args = parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This baseline uses one training process and one GPU per run")
    output = args.output_dir
    if any((output / name).exists() for name in
           ("intent.json", "continual_metrics.json", "joint_metrics.json", "completion.json")):
        if os.environ.get("OVERWRITE_OUTPUT", "0") != "1":
            raise FileExistsError(f"Results already exist in {output}; use another --output_dir "
                                  "or set OVERWRITE_OUTPUT=1")
    prepared, data_audit = inspect_data(args.data_dir)
    output.mkdir(parents=True, exist_ok=True)
    identity = process_identity()
    intent = {"created_utc": utc(), "method": args.method, "identity": identity,
              "output_dir": str(output), "data_audit": data_audit}
    save_json(output / "intent.json", intent)
    context = {"method": args.method, "identity": identity, "stage_index": None,
               "stage": None, "stage_count": 9 if args.method == "sequential_lora" else 1}
    try:
        code_files = [HERE / "train_lora.py"] + [REPO_ROOT / "common" / n for n in ("data.py", "model.py", "metrics.py")]
        execution = {
            **intent, "code": {str(p.relative_to(REPO_ROOT)): file_binding(p) for p in code_files},
            "model_assets": {name: file_binding(Path(args.model_name_or_path) / name)
                             for name in ("config.json", "tokenizer_config.json", "tokenizer.json")
                             if (Path(args.model_name_or_path) / name).is_file()},
        }
        save_json(output / "execution.json", execution)
        save_json(output / "status.json", {**context, "phase": "loading_model", "updated_utc": utc()})
        set_seed(args.seed)
        model, tokenizer = create_model(args)
        save_json(output / "status.json", {**context, "phase": "tokenizing", "updated_utc": utc()})
        data = load_data(args.data_dir, args.domain_order, tokenizer, args.max_length, prepared)
        del prepared
        sequential = args.method == "sequential_lora"
        stages = [[domain] for domain in args.domain_order] if sequential else [args.domain_order]
        if args.smoke:
            for d in data:
                data[d] = {k: v.select(range(min(len(v), 64 if k == "train" else 40))) for k, v in data[d].items()}
            stages = stages[:args.smoke]
        config = {
            **vars(args), "label_map": LABELS, "data_audit": data_audit,
            "code": execution["code"], "model_assets": execution["model_assets"],
            "resolved_lora_targets": model.resolved_lora_targets,
            "resolved_lora_target_count": len(model.resolved_lora_targets),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_selection": "fixed epochs; final checkpoint; no test selection",
            "optimizer_state_between_stages": "reset", "replay": False,
            "data_protocol": "500/100/300 pairs per domain; HWT versus AI",
            "training_text_exposures": 9000 * args.epochs,
            "expected_optimizer_steps": sum(math.ceil(
                math.ceil(1000 * len(domains) / args.train_batch_size)
                / args.gradient_accumulation_steps) * args.epochs for domains in stages),
            "evaluation_protocol": "all nine domains; valid calibration and separate empirical test ROC",
        }
        save_json(output / "run_config.json", config)
        model.print_trainable_parameters()
        history = []
        for index, train_domains in enumerate(stages):
            stage = f"stage_{index:02d}_{train_domains[0]}" if sequential else "joint_final"
            stage_dir = output / stage if sequential else output
            context = {**context, "stage_index": index, "stage": stage, "train_domains": train_domains}
            save_json(output / "status.json", {**context, "phase": "training", "updated_utc": utc()})
            dataset = concatenate_datasets([data[domain]["train"] for domain in train_domains])
            if not sequential:
                dataset = dataset.shuffle(seed=args.seed)
            trainer = build_trainer(model, tokenizer, dataset, args, stage_dir, args.seed + index,
                                    Progress(output / "status.json", context))
            train_metrics = trainer.train().metrics
            save_json(output / "status.json", {**context, "phase": "saving_adapter", "updated_utc": utc()})
            adapter = stage_dir / "adapter_final"
            trainer.save_model(str(adapter))
            if not (adapter / "adapter_config.json").is_file() or not any(
                    (adapter / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin")):
                raise RuntimeError("Final LoRA adapter and classification head were not saved")
            save_json(output / "status.json", {**context, "phase": "evaluating", "updated_utc": utc()})
            domain_metrics = evaluate_domains(trainer, data, args.domain_order, stage,
                                              output, args.target_fpr)
            history.append({"stage_index": index, "stage": stage, "train_domains": train_domains,
                            "train_metrics": train_metrics, "domain_metrics": domain_metrics})
            if sequential:
                result = {"method": args.method, "domain_order": args.domain_order, "history": history}
                filename = "continual_metrics.json"
            else:
                result = {"method": args.method, "domains": args.domain_order,
                          "train_metrics": train_metrics, "domain_metrics": domain_metrics}
                filename = "joint_metrics.json"
            if index == len(stages) - 1 and not args.smoke:
                result["summary"], matrices = summarize(history, args.domain_order,
                                                        sequential, args.target_fpr)
                if sequential:
                    result["performance_matrices"] = matrices
                print(json.dumps({k: v for k, v in result["summary"].items() if k.startswith(("final_AA", "final_global", "average_forgetting"))}, indent=2), flush=True)
            save_json(output / filename, result)
            save_json(stage_dir / "stage_completion.json", {
                **context, "completed_utc": utc(), "train_metrics": train_metrics,
                "adapter_files": {path.name: file_binding(path) for path in sorted(adapter.iterdir())
                                  if path.is_file()}, "metrics_file": file_binding(output / filename),
            })
            model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
            trainer.accelerator.free_memory()
            del trainer
        save_json(output / "completion.json", {
            "completed_utc": utc(), "method": args.method, "stage_count": len(history),
            "data_content_fingerprint": data_audit["content_fingerprint"],
            "result": {"file": filename, **file_binding(output / filename)},
        })
        save_json(output / "status.json", {**context, "phase": "completed", "updated_utc": utc(),
                                           "completed_stages": len(history)})
    except BaseException as error:
        save_json(output / "status.json", {**context, "phase": "failed", "updated_utc": utc(),
                                           "error_type": type(error).__name__})
        raise


if __name__ == "__main__":
    main()
