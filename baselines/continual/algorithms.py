"""EWC and O-LoRA, ported from EasyCL to the binary detector (the shell is shared with Sequential)."""
from __future__ import annotations
from typing import Any
import torch


def trainable_lora_params(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad and ".lora_" in n]


def trainable_lora_a_params(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad and ".lora_A." in n]


class OLoRAState:
    """Orthogonal constraint against all previous LoRA-A snapshots."""
    def __init__(self, orthogonal_lambda: float = 0.1, l2_lambda: float = 0.01):
        self.orthogonal_lambda = float(orthogonal_lambda)
        self.l2_lambda = float(l2_lambda)
        self.history: list[dict[str, torch.Tensor]] = []

    def penalty(self, model) -> torch.Tensor:
        device = next(model.parameters()).device
        current = dict(trainable_lora_a_params(model))
        orth = torch.zeros((), device=device)
        for old in self.history:
            for name, p in current.items():
                if name not in old or p.ndim != 2 or old[name].ndim != 2 or p.shape[1] != old[name].shape[1]:
                    continue
                historical = old[name]
                # Match upstream EasyCL numerics: do not force FP32 here.
                orth = orth + torch.abs(torch.mm(p, historical.transpose(0, 1).to(device=p.device))).sum()
        l2 = torch.zeros((), device=device)
        if self.l2_lambda:
            # Upstream EasyCL L2 regularization covers both LoRA A and B.
            for _, p in trainable_lora_params(model):
                l2 = l2 + p.pow(2).sum()
        return self.orthogonal_lambda * orth + self.l2_lambda * l2

    @torch.no_grad()
    def after_task(self, model) -> None:
        # Keep the historical A tensors in their original parameter dtype.
        # The upstream implementation loads these adapter tensors without a
        # dtype conversion before computing the orthogonality term.
        device = next(model.parameters()).device
        self.history.append({n: p.detach().to(device).clone() for n, p in trainable_lora_a_params(model)})

    def state_dict_cpu(self) -> list[dict[str, torch.Tensor]]:
        return [{n: v.detach().cpu() for n, v in snapshot.items()} for snapshot in self.history]


class EWCState:
    """Fisher-weighted quadratic penalty over trainable LoRA parameters."""
    def __init__(
        self,
        lambda_ewc: float = 0.5,
        fisher_samples: int = 100
    ):
        self.lambda_ewc = float(lambda_ewc)
        self.fisher_samples = int(fisher_samples)

        self.fisher: dict[str, torch.Tensor] = {}
        self.anchor: dict[str, torch.Tensor] = {}

        self.samples_processed = 0

        # Runtime-only GPU cache.
        self._runtime_device: torch.device | None = None
        self._runtime_fisher: dict[str, torch.Tensor] = {}
        self._runtime_anchor: dict[str, torch.Tensor] = {}
        self._runtime_params: list[tuple[str, torch.Tensor]] = []
    def _clear_runtime_cache(self) -> None:
        self._runtime_device = None
        self._runtime_fisher = {}
        self._runtime_anchor = {}
        self._runtime_params = []
    def _prepare_runtime_cache(self, model) -> None:
        device = next(model.parameters()).device

        if (
            self._runtime_device == device
            and self._runtime_params
        ):
            return

        self._runtime_params = [
            (name, param)
            for name, param in model.named_parameters()
            if name in self.fisher
            and name in self.anchor
        ]

        self._runtime_fisher = {
            name: fisher.to(device=device)
            for name, fisher in self.fisher.items()
            if name in self.anchor
        }

        self._runtime_anchor = {
            name: anchor.to(device=device)
            for name, anchor in self.anchor.items()
            if name in self.fisher
        }

        self._runtime_device = device
    def compute_fisher(self, model, batches) -> bool:
        self._clear_runtime_cache()
        device = next(model.parameters()).device
        lora_params = [
            (name, param)
            for name, param in model.named_parameters()
            if "lora_" in name
        ]

        fisher = {
            name: torch.zeros_like(
                param.data,
                device=device
            )
            for name, param in lora_params
        }
        model.eval()
        samples_processed = 0

        for batch in batches:
            # stop after fisher_samples texts (otherwise the whole training set would be used)
            if samples_processed >= self.fisher_samples:
                break

            model.zero_grad()

            outputs = model(**batch)

            loss = outputs.loss
            if loss is None:
                continue

            loss.backward()

            for name, param in lora_params:
                if param.grad is not None:
                    grad_data = param.grad.data.detach().clone()
                    fisher[name] += grad_data.pow(2)

            samples_processed += int(
                batch["input_ids"].shape[0]
            )
        if samples_processed <= 0:
            model.train()
            return False
        for name in fisher:
            fisher[name].div_(float(samples_processed))
        self.fisher = {
            n: tensor.cpu()
            for n, tensor in fisher.items()
        }
        self.anchor = {
            name: param.data.detach().clone().cpu()
            for name, param in lora_params
        }
        self.samples_processed = samples_processed
        model.train()
        return True

    def penalty(self, model) -> torch.Tensor:
        device = next(model.parameters()).device

        if not self.fisher or not self.anchor:
            return torch.zeros((), device=device)

        self._prepare_runtime_cache(model)

        total = torch.zeros((), device=device)

        for name, param in self._runtime_params:
            fisher = self._runtime_fisher[name]
            param_star = self._runtime_anchor[name]

            # EWC penalty on the parameter itself (the EasyCL source uses param.data, which blocks the gradient)
            total = total + (
                fisher * (param - param_star).pow(2)
            ).sum()

        return 0.5 * self.lambda_ewc * total

    def state_dict_cpu(self) -> dict[str, Any]:
        return {
            "fisher": {
                n: v.detach().cpu()
                for n, v in self.fisher.items()
            },
            "anchor": {
                n: v.detach().cpu()
                for n, v in self.anchor.items()
            },
        }
