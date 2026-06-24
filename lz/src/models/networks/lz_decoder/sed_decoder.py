from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class SEDDecoder(nn.Module):
    """Sparse Efficient Decoder (lightweight)."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        instance_dim: int = 32,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.instance_dim = instance_dim
        self.top_k = top_k

        self.seg_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_classes),
        )
        self.inst_head = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, instance_dim),
        )

    def forward(self, point_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.top_k is not None and self.top_k > 0:
            scores = torch.norm(point_feats, dim=1)
            k = min(self.top_k, point_feats.shape[0])
            topk_idx = torch.topk(scores, k=k, dim=0).indices
            feats_sparse = point_feats[topk_idx]
            seg_logits = self.seg_head(point_feats)
            inst_embeddings = self.inst_head(point_feats)
            return {
                "seg_logits": seg_logits,
                "inst_embeddings": inst_embeddings,
                "sparse_idx": topk_idx,
                "sparse_feats": feats_sparse,
            }

        seg_logits = self.seg_head(point_feats)
        inst_embeddings = self.inst_head(point_feats)
        return {
            "seg_logits": seg_logits,
            "inst_embeddings": inst_embeddings,
        }
