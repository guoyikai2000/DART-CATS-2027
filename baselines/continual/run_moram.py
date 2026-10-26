#!/usr/bin/env python3
"""MoRAM on CATS.

The shell (data, prompt, last-token classifier with fp32 score head, HF Trainer optimizer/schedule,
seeds, metrics) is that of the Sequential baseline. Instead of dense LoRA, the same gate/up/down text
projections are wrapped with moram.MoRAMLinear (original MoRAM rank/top-k/temperature/alpha), and after
every stage the current ranks are promoted to frozen memory.

  python baselines/continual/run_moram.py --config baselines/continual/configs/moram.yaml --order O1
"""

import argparse
import gc
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "common"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from torch import nn  # noqa: E402
from transformers import AutoTokenizer, Trainer, set_seed  # noqa: E402

from cl_common import (collator, file_binding, guard_output, load_config, prepare_config, strip_ids,  # noqa: E402
                       training_args, utc, validate_shell)
from data import LABELS, inspect_data, load_data, save_json  # noqa: E402
from metrics import evaluate_domains, summarize  # noqa: E402
from model import load_classifier  # noqa: E402
from moram import MoRAMLinear  # noqa: E402

LOGGER = logging.getLogger("moram_cats")


_ORIGINAL_MORAM_FORWARD = MoRAMLinear.forward


def _moram_forward_bf16(self, x):
    """Training already runs under the Trainer's bf16 autocast. The original MoRAM also evaluates under
    bf16 autocast; apply that only inside MoRAM layers so the classifier head keeps its fp32 evaluation
    path exactly like the other baselines."""
    if torch.is_autocast_enabled():
        return _ORIGINAL_MORAM_FORWARD(self, x)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        return _ORIGINAL_MORAM_FORWARD(self, x).to(x.dtype)


MoRAMLinear.forward = _moram_forward_bf16


def build(cfg):
    shell, m = cfg["shell"], cfg["method"]
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name_or_path"], fix_mistral_regex=True,
                                              local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = tokenizer.truncation_side = "right"
    model = load_classifier(cfg["model_name_or_path"], shell["precision"])  # the shared classifier
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.get_text_config().pad_token_id = tokenizer.pad_token_id
    suffixes = tuple(shell["lora_targets"].split(","))
    targets = [name for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and name.endswith(suffixes) and "vision" not in name]
    if not targets:
        raise ValueError("No text-backbone MoRAM targets were found")
    for p in model.parameters():
        p.requires_grad = False
    for name in targets:
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child, MoRAMLinear(getattr(parent, child), rank=int(m["moram_rank"]),
                                           alpha=float(m["moram_alpha"]), dropout=float(m["moram_dropout"]),
                                           topk=int(m["moram_topk"]),
                                           temperature=float(m["moram_temperature"])))
    for p in model.score.parameters():
        p.requires_grad = True
    return model, tokenizer, targets


def moram_state(model):
    state = {}
    for name, module in model.named_modules():
        if isinstance(module, MoRAMLinear):
            for suffix in ("old_A", "old_B", "new_A", "new_B"):
                state[f"{name}.{suffix}"] = getattr(module, suffix).detach().cpu()
    state["score.weight"] = model.score.weight.detach().cpu()
    return state


def run(cfg, smoke=0):
    if cfg.get("method_name") != "moram":
        raise ValueError("method_name must be moram")
    validate_shell(cfg, skip=("lora_rank", "lora_alpha", "lora_dropout") + (("epochs",) if smoke else ()))
    shell = cfg["shell"]
    order = list(cfg["domain_order"])
    output = Path(cfg["output_dir"])
    if smoke:
        output = output.parent / f"_smoke_{output.name}"
        order = order[:smoke]
    guard_output(output)
    data_dir = Path(cfg["data_dir"])
    prepared, data_audit = inspect_data(data_dir)
    save_json(output / "intent.json", {"created_utc": utc(), "method": "moram", "argv": sys.argv,
                                        "data_audit": data_audit, "smoke": smoke})
    set_seed(shell["seed"])
    model, tokenizer, targets = build(cfg)
    data = load_data(data_dir, list(cfg["domain_order"]), tokenizer, shell["max_length"], prepared)
    if smoke:
        for d in data:
            data[d] = {k: v.select(range(min(len(v), 64 if k == "train" else 40))) for k, v in data[d].items()}
        shell = {**shell, "epochs": 1}
    del prepared
    code_files = [HERE / n for n in ("run_moram.py", "moram.py")] + \
                 [REPO_ROOT / "common" / n for n in ("cl_common.py", "data.py", "metrics.py", "model.py")]
    save_json(output / "run_config.json", {
        **cfg, "domain_order_effective": order, "label_map": LABELS, "data_audit": data_audit,
        "code": {str(p.relative_to(REPO_ROOT)): file_binding(p) for p in code_files},
        "resolved_moram_targets": targets, "resolved_moram_target_count": len(targets),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "note": "lora_rank/alpha/dropout in shell are unused; MoRAM uses method.moram_* (original MoRAM values)",
        "optimizer_state_between_stages": "reset (new Trainer per stage)",
        "checkpoint_selection": "fixed epochs; final checkpoint; no test selection",
    })
    history = []
    for index, domain in enumerate(order):
        stage = f"stage_{index:02d}_{domain}"
        stage_dir = output / stage
        stage_seed = shell["seed"] + index
        set_seed(stage_seed)
        trainer = Trainer(model=model, train_dataset=strip_ids(data[domain]["train"]),
                          processing_class=tokenizer, data_collator=collator(tokenizer),
                          args=training_args(shell, stage_dir, stage_seed))
        save_json(output / "status.json", {"phase": "training", "stage": stage, "updated_utc": utc()})
        train_metrics = trainer.train().metrics
        model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
        (stage_dir / "checkpoint").mkdir(parents=True, exist_ok=True)
        torch.save({"state": moram_state(model), "meta": {"stage": stage, "targets": targets}},
                   stage_dir / "checkpoint" / "moram_adapter.pt")
        save_json(output / "status.json", {"phase": "evaluating", "stage": stage, "updated_utc": utc()})
        domain_metrics = evaluate_domains(trainer, data, list(cfg["domain_order"]), stage, output,
                                          shell["target_fpr"])
        history.append({"stage_index": index, "stage": stage, "train_domains": [domain],
                        "train_metrics": train_metrics, "domain_metrics": domain_metrics})
        result = {"method": "moram", "domain_order": order, "history": history}
        if index == len(order) - 1 and not smoke:
            result["summary"], result["performance_matrices"] = summarize(history, order, True, shell["target_fpr"])
        save_json(output / "continual_metrics.json", result)
        if index < len(order) - 1:
            for module in model.modules():
                if isinstance(module, MoRAMLinear):
                    module.promote_current_to_old()
        trainer.accelerator.free_memory()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()
    save_json(output / "completion.json", {"completed_utc": utc(), "method": "moram",
                                            "stage_count": len(history), "smoke": smoke,
                                            "result": file_binding(output / "continual_metrics.json")})
    save_json(output / "status.json", {"phase": "completed", "updated_utc": utc()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--order", choices=("O0", "O1", "O2", "O3"))
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--data_dir")
    ap.add_argument("--model_name_or_path")
    ap.add_argument("--output_dir")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = prepare_config(load_config(args.config), args.order, args.data_dir, args.model_name_or_path,
                         args.output_dir)
    run(cfg, args.smoke)


if __name__ == "__main__":
    main()
