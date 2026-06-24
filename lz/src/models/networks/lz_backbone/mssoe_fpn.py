from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class MSSOEBackbone(nn.Module):
    """Multi-scale small-object enhancement backbone (lightweight)."""

    def __init__(
        self,
        in_dim: int = 6,
        hidden_dim: int = 64,
        out_dim: int = 128,
        text_dim: int = 768,
        dropout: float = 0.1,
        num_queries: int = 128,
        use_lgg: bool = True,
        ablate_coord: bool = False,
        ablate_color: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_queries = num_queries
        self.use_lgg = use_lgg
        self.ablate_coord = bool(ablate_coord)
        self.ablate_color = bool(ablate_color)

        self.scale1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.scale2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.scale3 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

        self.fuse = nn.Sequential(
            nn.Linear(hidden_dim * 3, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Language-Guided Gate (LGG): one gate per scale branch.
        if self.use_lgg:
            self.lgg_feat_1 = nn.Linear(hidden_dim, hidden_dim)
            self.lgg_feat_2 = nn.Linear(hidden_dim, hidden_dim)
            self.lgg_feat_3 = nn.Linear(hidden_dim, hidden_dim)
            self.lgg_text_1 = nn.Linear(text_dim, hidden_dim)
            self.lgg_text_2 = nn.Linear(text_dim, hidden_dim)
            self.lgg_text_3 = nn.Linear(text_dim, hidden_dim)

        # Scale-aware query embeddings
        self.query_embed = nn.Parameter(torch.randn(num_queries, out_dim))

    def _apply_lgg(
        self,
        feat: torch.Tensor,
        text_guidance: Optional[torch.Tensor],
        feat_proj: nn.Linear,
        text_proj: nn.Linear,
    ) -> torch.Tensor:
        if (not self.use_lgg) or (text_guidance is None):
            return feat
        if text_guidance.dim() == 2:
            text_guidance = text_guidance.mean(dim=0)
        text_guidance = text_guidance.to(device=feat.device, dtype=feat.dtype)
        gate = torch.sigmoid(feat_proj(feat) + text_proj(text_guidance).unsqueeze(0))
        return feat * gate

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        text_guidance: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        coord = batch["coord"].float()
        color = batch.get("color", None)
        if color is None:
            color = torch.zeros_like(coord)
        else:
            color = color.float()

        if self.ablate_coord:
            coord = torch.zeros_like(coord)
        if self.ablate_color:
            color = torch.zeros_like(color)

        x = torch.cat([coord, color], dim=1)

        f1 = self.scale1(x)
        f2 = self.scale2(x)
        f3 = self.scale3(x)
        f1 = self._apply_lgg(f1, text_guidance, self.lgg_feat_1, self.lgg_text_1) if self.use_lgg else f1
        f2 = self._apply_lgg(f2, text_guidance, self.lgg_feat_2, self.lgg_text_2) if self.use_lgg else f2
        f3 = self._apply_lgg(f3, text_guidance, self.lgg_feat_3, self.lgg_text_3) if self.use_lgg else f3
        feats = self.fuse(torch.cat([f1, f2, f3], dim=1))

        return {
            "point_feats": feats,
            "debug_input_coord_mean": coord.mean(),
            "debug_input_coord_std": coord.std(unbiased=False),
            "debug_input_color_mean": color.mean(),
            "debug_input_color_std": color.std(unbiased=False),
        }

    def get_queries(self, num_queries: Optional[int] = None) -> torch.Tensor:
        if num_queries is None or num_queries >= self.num_queries:
            return self.query_embed
        return self.query_embed[:num_queries]
