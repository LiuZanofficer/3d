from __future__ import annotations

from typing import Dict, Optional


class PMTLScheduler:
    """Progressive multi-task learning scheduler."""

    def __init__(
        self,
        stage1_epochs: int = 20,
        stage2_epochs: int = 20,
        stage3_epochs: int = 30,
        weights_stage1: Optional[Dict[str, float]] = None,
        weights_stage2: Optional[Dict[str, float]] = None,
        weights_stage3: Optional[Dict[str, float]] = None,
    ) -> None:
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage3_epochs = stage3_epochs
        self.weights_stage1 = weights_stage1 or {}
        self.weights_stage2 = weights_stage2 or {}
        self.weights_stage3 = weights_stage3 or {}

    def get_stage(self, epoch: int) -> int:
        if epoch < self.stage1_epochs:
            return 1
        if epoch < self.stage1_epochs + self.stage2_epochs:
            return 2
        return 3

    def get_weights(self, epoch: int) -> Dict[str, float]:
        stage = self.get_stage(epoch)
        if stage == 1:
            return self.weights_stage1
        if stage == 2:
            return self.weights_stage2
        return self.weights_stage3
