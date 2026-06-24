from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class HPZALoss(nn.Module):
    """Hierarchical Prototype Zero-shot Alignment loss (lightweight)."""

    def __init__(
        self,
        temperature: float = 0.07,
        weight_caption: float = 1.0,
        weight_class: float = 1.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self.weight_caption = weight_caption
        self.weight_class = weight_class

    @staticmethod
    def cosine_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        return 1.0 - (a * b).sum(dim=-1)

    def caption_alignment(
        self, mask_feats: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        if mask_feats.numel() == 0 or text_embeds.numel() == 0:
            return mask_feats.sum() * 0.0
        loss = self.cosine_loss(mask_feats, text_embeds).mean()
        return loss

    def class_alignment(
        self,
        class_feats: torch.Tensor,
        class_text_embeds: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if class_text_embeds is None or class_feats.numel() == 0:
            return class_feats.sum() * 0.0
        if class_text_embeds.shape[0] != class_feats.shape[0]:
            return class_feats.sum() * 0.0
        loss = self.cosine_loss(class_feats, class_text_embeds).mean()
        return loss
