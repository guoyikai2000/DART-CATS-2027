from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoRAMLinear(nn.Module):
    """MoRAM-style linear: frozen base + accumulated rank-1 memories + trainable current ranks."""
    def __init__(self, base: nn.Linear, rank: int, alpha: float = 32.0, dropout: float = 0.0, topk: int | None = None, temperature: float = 0.03):
        super().__init__()
        self.base = base
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / max(self.rank, 1)
        self.dropout = nn.Dropout(dropout)
        self.topk = topk if topk is not None else self.rank
        self.temperature = max(float(temperature), 1e-6)
        self.infer_threshold = 0.0
        # These are the original MoRAM implementation defaults. They are exposed
        # in the aligned config for auditability, not changed algorithmically.
        for p in self.base.parameters(): p.requires_grad = False
        self.old_A = nn.Parameter(torch.empty(0, self.in_features), requires_grad=False)
        self.old_B = nn.Parameter(torch.empty(self.out_features, 0), requires_grad=False)
        self.new_A = nn.Parameter(torch.empty(self.rank, self.in_features), requires_grad=True)
        self.new_B = nn.Parameter(torch.zeros(self.out_features, self.rank), requires_grad=True)
        self._reset_new()

    def _reset_new(self):
        nn.init.kaiming_uniform_(self.new_A, a=math.sqrt(5))
        nn.init.zeros_(self.new_B)

    @property
    def old_rank(self): return self.old_A.shape[0]
    @property
    def total_rank(self): return self.old_rank + self.rank

    def forward(self, x):
        result = F.linear(x, self.base.weight, self.base.bias)
        dropped = self.dropout(x)
        A = torch.cat([self.old_A, self.new_A], dim=0)
        proj = F.linear(dropped, A)
        normed = F.normalize(proj, p=2, dim=-1, eps=1e-8)
        k = min(max(int(self.topk), 2), normed.shape[-1])
        vals, idx = torch.topk(normed, k=k, dim=-1)
        logits = normed.new_full(normed.shape, float("-inf"))
        logits.scatter_(-1, idx, vals)
        weights = torch.softmax(logits / self.temperature, dim=-1)
        if not self.training and self.infer_threshold > 0:
            weights = weights * (normed >= self.infer_threshold).to(weights.dtype)
        weighted = proj * weights
        B = torch.cat([self.old_B, self.new_B * self.scale], dim=1)
        return result + F.linear(weighted, B)

    @torch.no_grad()
    def promote_current_to_old(self):
        old_r = self.old_rank
        new_r = self.rank
        A = torch.cat([self.old_A.detach(), self.new_A.detach()], dim=0)
        B = torch.cat([self.old_B.detach(), self.new_B.detach()], dim=1)
        self.old_A = nn.Parameter(A, requires_grad=False)
        self.old_B = nn.Parameter(B, requires_grad=False)
        self.new_A = nn.Parameter(torch.empty(new_r, self.in_features, device=A.device, dtype=A.dtype), requires_grad=True)
        self.new_B = nn.Parameter(torch.zeros(self.out_features, new_r, device=B.device, dtype=B.dtype), requires_grad=True)
        self._reset_new()


def replace_target_modules(root: nn.Module, target_suffixes: list[str], rank: int, topk: int, temperature: float) -> list[str]:
    replaced = []
    for name, module in list(root.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(name.endswith(s) or name.split(".")[-1] == s for s in target_suffixes):
            continue
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = root.get_submodule(parent_name) if parent_name else root
        setattr(parent, child_name, MoRAMLinear(module, rank=rank, topk=topk, temperature=temperature))
        replaced.append(name)
    if not replaced:
        raise RuntimeError(f"No target linear modules found for {target_suffixes}")
    return replaced


def iter_moram(root: nn.Module):
    for m in root.modules():
        if isinstance(m, MoRAMLinear):
            yield m
