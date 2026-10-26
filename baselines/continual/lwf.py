"""Learning without Forgetting, ported from EasyCL (src/easycl/cl/lwf/lwf.py).

Upstream semantics kept:
  * teacher = the model as it was at the end of the previous task, in eval mode;
    its logits on the CURRENT task's training data are pre-computed once before
    training (EasyCL `precompute_logits`, memory cache keyed by sample index);
  * lwf_total = (1 - alpha) * CE(current) + alpha * T^2 * KL(softmax(teacher/T) || softmax(current/T)),
    KL with reduction='batchmean';
  * the trainer ADDS lwf_total to the ordinary task loss (`loss += lwf_loss` in
    LWFTrainer.compute_loss), so CE effectively has weight 2 - alpha;
  * LwF is disabled on the first task (no previous model);
  * defaults temperature=2.0, alpha=0.5 (CLFinetuningArguments).
Only adaptation: the upstream per-token LM logits become the binary
classifier logits (one "position" per sample).
"""

import torch
from torch.nn import functional as F

TEACHER_COLUMN = "lwf_teacher_logits"


class LwFState:
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        self.temperature = float(temperature)
        self.alpha = float(alpha)
        self.enabled = False

    def loss(self, logits: torch.Tensor, teacher: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return logits.new_zeros(())
        t = self.temperature
        current = logits.float()
        teacher = teacher.to(device=current.device, dtype=torch.float32)
        soft_previous = F.softmax(teacher / t, dim=-1)
        log_soft_current = F.log_softmax(current / t, dim=-1)
        distillation = F.kl_div(log_soft_current, soft_previous, reduction="batchmean",
                                log_target=False) * (t ** 2)
        ce = F.cross_entropy(current, labels)
        return (1 - self.alpha) * ce + self.alpha * distillation
