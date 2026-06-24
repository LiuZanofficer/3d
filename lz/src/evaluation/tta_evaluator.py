from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


class TTAEvaluator:
    """Test-time augmentation with rotations/flips and temperature scaling."""

    def __init__(
        self,
        rotations: List[int],
        flips: List[bool],
        temperature: float = 1.0,
        use_adapter_tta: bool = True,
        adapter_steps: int = 1,
        adapter_lr: float = 1e-3,
        adapter_weight_decay: float = 0.0,
        entropy_weight: float = 1.0,
    ) -> None:
        self.rotations = rotations
        self.flips = flips
        self.temperature = max(temperature, 1e-6)
        self.use_adapter_tta = use_adapter_tta
        self.adapter_steps = max(int(adapter_steps), 0)
        self.adapter_lr = float(adapter_lr)
        self.adapter_weight_decay = float(adapter_weight_decay)
        self.entropy_weight = float(entropy_weight)

    def _rotate(self, coord: torch.Tensor, angle: int) -> torch.Tensor:
        if angle % 360 == 0:
            return coord
        theta = torch.tensor(angle * 3.14159265 / 180.0, device=coord.device)
        c, s = torch.cos(theta), torch.sin(theta)
        rot = torch.tensor([[c, -s], [s, c]], device=coord.device)
        xy = coord[:, :2] @ rot.T
        coord = coord.clone()
        coord[:, :2] = xy
        return coord

    def _flip(self, coord: torch.Tensor) -> torch.Tensor:
        coord = coord.clone()
        coord[:, 0] = -coord[:, 0]
        return coord

    @staticmethod
    def _entropy_loss(logits: torch.Tensor) -> torch.Tensor:
        prob = F.softmax(logits, dim=-1)
        entropy = -(prob * torch.log(prob + 1e-8)).sum(dim=-1)
        return entropy.mean()

    def _adapt_adapter(self, model, batch) -> dict:
        if not hasattr(model.net, "tta_adapter"):
            return {}
        adapter = model.net.tta_adapter
        state = {k: v.detach().clone() for k, v in adapter.state_dict().items()}
        if (not self.use_adapter_tta) or self.adapter_steps <= 0:
            return state

        # Freeze full model, then enable gradients only on adapter.
        param_trainable = [p.requires_grad for p in model.parameters()]
        for p in model.parameters():
            p.requires_grad = False
        for p in adapter.parameters():
            p.requires_grad = True

        was_training = adapter.training
        try:
            adapter.train(True)
            optimizer = torch.optim.Adam(
                adapter.parameters(),
                lr=self.adapter_lr,
                weight_decay=self.adapter_weight_decay,
            )
            with torch.inference_mode(False):
                with torch.enable_grad():
                    for _ in range(self.adapter_steps):
                        optimizer.zero_grad(set_to_none=True)
                        logits = model(batch, apply_tta_adapter=True)["seg_logits"]
                        loss = self.entropy_weight * self._entropy_loss(
                            logits / self.temperature
                        )
                        loss.backward()
                        optimizer.step()
            adapter.train(was_training)
        finally:
            for p, old_flag in zip(model.parameters(), param_trainable):
                p.requires_grad = old_flag
        return state

    def __call__(self, model, batch):
        adapter_state = self._adapt_adapter(model, batch)
        logits_sum = None
        n = 0
        coord_orig = batch["coord"]
        for rot in self.rotations:
            coord_rot = self._rotate(coord_orig, rot)
            for flip in self.flips:
                coord_aug = self._flip(coord_rot) if flip else coord_rot
                batch_aug = dict(batch)
                batch_aug["coord"] = coord_aug
                with torch.no_grad():
                    logits = model(batch_aug, apply_tta_adapter=True)["seg_logits"]
                    logits = logits / self.temperature
                logits_sum = logits if logits_sum is None else logits_sum + logits
                n += 1
        if adapter_state and hasattr(model.net, "tta_adapter"):
            model.net.tta_adapter.load_state_dict(adapter_state)
        return logits_sum / max(n, 1)
