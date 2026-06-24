from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class LZModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        decoder: nn.Module,
        text_dim: int = 768,
        use_tta_adapter: bool = True,
        adapter_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.text_dim = text_dim
        self.text_proj = nn.Linear(backbone.out_dim, text_dim)
        self.use_tta_adapter = use_tta_adapter
        self.tta_adapter = nn.Sequential(
            nn.Linear(backbone.out_dim, backbone.out_dim),
            nn.ReLU(),
            nn.Dropout(adapter_dropout),
            nn.Linear(backbone.out_dim, backbone.out_dim),
        )
        # Start from near-identity behavior.
        last = self.tta_adapter[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        text_guidance: Optional[torch.Tensor] = None,
        apply_tta_adapter: bool = False,
        return_text_feats: bool = True,
    ) -> Dict[str, torch.Tensor]:
        backbone_out = self.backbone(batch, text_guidance=text_guidance)
        point_feats = backbone_out["point_feats"]
        if apply_tta_adapter and self.use_tta_adapter:
            point_feats = point_feats + self.tta_adapter(point_feats)
        decoded = self.decoder(point_feats)
        decoded["point_feats"] = point_feats
        if return_text_feats:
            decoded["text_feats"] = self.text_proj(point_feats)
        for key, value in backbone_out.items():
            if key != "point_feats":
                decoded[key] = value
        return decoded
