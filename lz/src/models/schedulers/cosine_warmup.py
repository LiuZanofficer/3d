from __future__ import annotations

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if warmup_steps >= total_steps:
            warmup_steps = max(total_steps - 1, 0)
        if not (0.0 <= min_lr_ratio <= 1.0):
            raise ValueError(f"min_lr_ratio must be in [0,1], got {min_lr_ratio}")

        self.total_steps = int(total_steps)
        self.warmup_steps = int(warmup_steps)
        self.min_lr_ratio = float(min_lr_ratio)
        super().__init__(optimizer, last_epoch)

    def _lr_factor(self, step: int) -> float:
        step = max(int(step), 0)
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return float(step + 1) / float(self.warmup_steps)

        if self.total_steps <= self.warmup_steps:
            return 1.0

        progress = float(step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine

    def get_lr(self) -> List[float]:
        factor = self._lr_factor(self.last_epoch)
        return [base_lr * factor for base_lr in self.base_lrs]
