"""Shared continual-learning shell for DART and the continual baselines.

Everything that is not the continual-learning mechanism (data, detector prompt, classifier, LoRA
configuration, optimizer, scheduler, seeds, evaluation, metrics) is shared by all methods. A method
may only change (a) the training set of a stage, (b) an extra loss term, and (c) what happens to the
model between stages.

Paths: every relative path in a config file or on the command line is resolved against the
repository root, so the scripts can be started from any working directory.
"""

import copy
import hashlib
import os
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from data import DOMAINS  # noqa: E402

# The four domain orders of the paper (Appendix A). All start from the unattacked D00.
ORDERS = {
    "O0": ["D00", "D08", "D03", "D02", "D06", "D04", "D05", "D01", "D07"],
    "O1": ["D00", "D07", "D01", "D05", "D04", "D06", "D02", "D03", "D08"],
    "O2": ["D00", "D05", "D08", "D06", "D03", "D01", "D04", "D07", "D02"],
    "O3": ["D00", "D06", "D02", "D03", "D08", "D07", "D01", "D05", "D04"],
}

# Shared training shell of all continual methods in the paper (Section 5.1).
PAPER_SHELL = {
    "max_length": 512, "epochs": 2, "learning_rate": 2e-4, "train_batch_size": 8,
    "eval_batch_size": 8, "gradient_accumulation_steps": 1, "weight_decay": 0.01,
    "warmup_ratio": 0.05, "lora_rank": 24, "lora_alpha": 48, "lora_dropout": 0.05,
    "lora_targets": "gate_proj,up_proj,down_proj", "precision": "bf16",
    "target_fpr": 0.01, "seed": 2026, "dataloader_workers": 4,
}


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_binding(path: Path) -> dict:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"bytes": Path(path).stat().st_size, "sha256": digest.hexdigest()}


def resolve_path(value) -> Path:
    path = Path(os.path.expanduser(str(value)))
    return path if path.is_absolute() else REPO_ROOT / path


def _deep_merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path) -> dict:
    """YAML (or JSON) config. A `base: other.yaml` key (relative to this file) is loaded first and
    the remaining keys are merged into it recursively."""
    import yaml
    path = resolve_path(path)
    with open(path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    base = cfg.pop("base", None)
    if base:
        cfg = _deep_merge(load_config(path.parent / base), cfg)
    return cfg


def prepare_config(cfg: dict, order: str | None = None, data_dir=None, model_name_or_path=None,
                   output_dir=None) -> dict:
    """Apply command-line choices and resolve all paths. The run writes to output_dir/<order>."""
    cfg = copy.deepcopy(cfg)
    if order is not None:
        if order not in ORDERS:
            raise ValueError(f"Unknown order {order!r}; choose one of {sorted(ORDERS)}")
        cfg["domain_order"] = list(ORDERS[order])
        cfg["order"] = order
    if "domain_order" not in cfg:
        raise ValueError("Give --order (O0-O3) or a domain_order in the config")
    for key, value in (("data_dir", data_dir), ("model_name_or_path", model_name_or_path),
                       ("output_dir", output_dir)):
        if value is not None:
            cfg[key] = value
    cfg["data_dir"] = str(resolve_path(cfg["data_dir"]))
    cfg["model_name_or_path"] = str(resolve_path(cfg["model_name_or_path"]))
    out = resolve_path(cfg["output_dir"])
    cfg["output_dir"] = str(out / cfg["order"]) if cfg.get("order") else str(out)
    return cfg


def validate_shell(cfg: dict, skip=()) -> None:
    """With `protocol: paper` (default), the shell must equal the paper's shared settings.
    Use `protocol: custom` in a config to run other values."""
    shell = cfg["shell"]
    missing = [key for key in PAPER_SHELL if key not in shell]
    if missing:
        raise ValueError(f"Missing shell fields: {missing}")
    if cfg.get("protocol", "paper") == "paper":
        for key, expected in PAPER_SHELL.items():
            if key in skip:
                continue
            got = shell.get(key)
            same = (abs(float(got) - expected) < 1e-12) if isinstance(expected, float) else got == expected
            if not same:
                raise ValueError(f"Shell field {key}: the paper uses {expected!r}, got {got!r} "
                                 "(set `protocol: custom` to allow other values)")
    order = list(cfg["domain_order"])
    if sorted(order) != list(DOMAINS) or order[0] != "D00":
        raise ValueError("domain_order must contain D00-D08 exactly once and start with D00")


def shell_namespace(cfg) -> Namespace:
    s = cfg["shell"]
    return Namespace(model_name_or_path=cfg["model_name_or_path"], precision=s["precision"],
                     lora_targets=s["lora_targets"], lora_rank=s["lora_rank"],
                     lora_alpha=s["lora_alpha"], lora_dropout=s["lora_dropout"])


def training_args(shell: dict, output_dir: Path, seed: int) -> TrainingArguments:
    """The TrainingArguments of the Sequential baseline (baselines/continual/train_lora.py).

    Only remove_unused_columns=False is added; callers drop the string sample_id column themselves,
    so the batches are identical to the default behaviour while method-specific tensor columns
    (LwF teacher logits) can pass through.
    """
    return TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=shell["epochs"],
        learning_rate=shell["learning_rate"],
        per_device_train_batch_size=shell["train_batch_size"],
        per_device_eval_batch_size=shell["eval_batch_size"],
        gradient_accumulation_steps=shell["gradient_accumulation_steps"],
        weight_decay=shell["weight_decay"], warmup_ratio=shell["warmup_ratio"],
        lr_scheduler_type="cosine", optim="adamw_torch",
        bf16=shell["precision"] == "bf16", fp16=shell["precision"] == "fp16",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        ddp_find_unused_parameters=False, dataloader_num_workers=shell["dataloader_workers"],
        label_names=["labels"], eval_strategy="no", save_strategy="no",
        logging_steps=10, logging_first_step=True, report_to="none",
        seed=seed, data_seed=seed, remove_unused_columns=False,
    )


def collator(tokenizer):
    return DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)


def strip_ids(dataset):
    return dataset.remove_columns([c for c in ("sample_id",) if c in dataset.column_names])


def guard_output(output: Path) -> None:
    """Refuse to overwrite finished results (set OVERWRITE_OUTPUT=1 to allow it)."""
    output = Path(output)
    if any((output / name).exists() for name in ("intent.json", "continual_metrics.json", "completion.json")):
        if os.environ.get("OVERWRITE_OUTPUT", "0") != "1":
            raise FileExistsError(f"Results already exist in {output}; use another output_dir "
                                  "or set OVERWRITE_OUTPUT=1")
    output.mkdir(parents=True, exist_ok=True)


def predict_logits(model, tokenizer, dataset, shell: dict, tmp_dir: Path, seed: int):
    trainer = Trainer(model=model, processing_class=tokenizer, data_collator=collator(tokenizer),
                      args=training_args(shell, tmp_dir, seed))
    out = trainer.predict(strip_ids(dataset))
    logits = out.predictions[0] if isinstance(out.predictions, tuple) else out.predictions
    del trainer
    return torch.as_tensor(logits, dtype=torch.float32)
