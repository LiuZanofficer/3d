"""Tail-Decoupled Prototype (TDP) Decoder.

Replaces the simple ``SEDDecoder``. Two prediction branches:

* **Generic branch**: a standard linear classifier producing logits over all
  ``num_classes`` classes (``z_g``).
* **Tail branch**: a lightweight projection to a unit-norm embedding space
  followed by cosine similarity against ``|C_t|`` learnable prototypes. The
  resulting tail logits ``z_t_tail \\in R^{|C_t|}`` are scattered into the
  full ``num_classes`` logit space with ``-inf`` on every non-tail position
  (so they contribute exactly zero probability after softmax outside the
  tail set).

A per-point soft gate ``alpha = sigmoid(MLP(f))`` blends the two branches:
``z = (1 - alpha) * z_g + alpha * z_t``. Gating is **not** directly
supervised; it is shaped indirectly because the cross-entropy on tail
labels is reachable only through the tail branch.

Prototypes are stored as a non-trainable buffer and updated with EMA in
``ema_update`` (called from ``LZLitModule.training_step`` after the loss has
been computed). The first ``warmup_steps`` updates use ``warmup_momentum``
(softer) to avoid being dominated by noisy initial features.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class TDPDecoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        tail_class_indices: Sequence[int],
        instance_dim: int = 32,
        proto_dim: int = 128,
        tau: float = 0.1,
        ema_momentum: float = 0.99,
        warmup_steps: int = 500,
        warmup_momentum: float = 0.9,
        hidden_dim: int = 128,
        top_k: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.instance_dim = int(instance_dim)
        self.proto_dim = int(proto_dim)
        self.tau = float(tau)
        self.ema_momentum = float(ema_momentum)
        self.warmup_steps = int(warmup_steps)
        self.warmup_momentum = float(warmup_momentum)
        self.top_k = top_k

        if len(tail_class_indices) == 0:
            raise ValueError("TDPDecoder requires at least one tail class.")
        tail_idx_tensor = torch.as_tensor(sorted(set(int(c) for c in tail_class_indices)), dtype=torch.long)
        if int(tail_idx_tensor.min()) < 0 or int(tail_idx_tensor.max()) >= self.num_classes:
            raise ValueError(
                f"tail_class_indices out of range [0, {self.num_classes}): {tail_idx_tensor.tolist()}"
            )
        self.register_buffer("tail_class_indices", tail_idx_tensor, persistent=True)
        self.num_tail_classes = int(tail_idx_tensor.numel())

        # Generic full-class linear classifier.
        self.generic_head = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_dim, self.num_classes),
        )

        # Tail projection to the prototype embedding space.
        self.tail_proj = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.proto_dim),
        )

        # Soft gate: per-point scalar in [0, 1] biased toward the tail branch.
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )

        # Instance head kept for compatibility with LZLitModule.
        self.inst_head = nn.Sequential(
            nn.Linear(self.in_dim, self.in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_dim, self.instance_dim),
        )

        # Prototype bank: not a Parameter (we do not back-prop through it; it is
        # updated via EMA from the live tail features).
        prototypes = torch.zeros(self.num_tail_classes, self.proto_dim)
        nn.init.normal_(prototypes, std=0.02)
        prototypes = F.normalize(prototypes, dim=-1)
        self.register_buffer("prototypes", prototypes, persistent=True)
        self.register_buffer("proto_initialized", torch.zeros(self.num_tail_classes, dtype=torch.bool), persistent=True)
        self.register_buffer("ema_step", torch.zeros((), dtype=torch.long), persistent=True)

        # Cache for global-id -> tail-rank mapping (filled lazily on first forward).
        global_to_tail_rank = torch.full((self.num_classes,), -1, dtype=torch.long)
        global_to_tail_rank[tail_idx_tensor] = torch.arange(self.num_tail_classes, dtype=torch.long)
        self.register_buffer("global_to_tail_rank", global_to_tail_rank, persistent=True)

    @property
    def tail_class_set(self) -> List[int]:
        return self.tail_class_indices.detach().cpu().tolist()

    def forward(self, point_feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_g = self.generic_head(point_feats)  # [N, num_classes]

        # Tail branch: project, normalize, cosine against prototypes.
        tail_feat = self.tail_proj(point_feats)  # [N, proto_dim]
        tail_feat_norm = F.normalize(tail_feat, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        tail_logits_subset = (tail_feat_norm @ proto_norm.t()) / self.tau  # [N, |C_t|]

        # Per-point soft gate.
        alpha_logit = self.gate_mlp(point_feats)
        alpha = torch.sigmoid(alpha_logit)  # [N, 1]

        # Build seg_logits in-place: non-tail class columns are exactly z_g;
        # tail class columns are blended ``(1-alpha) * z_g_tail + alpha * z_t_tail``.
        # We avoid any -inf scatter so 0 * -inf = NaN cannot occur for non-tail
        # columns where alpha is conceptually 0.
        seg_logits = z_g.clone()
        z_g_tail = z_g.index_select(dim=1, index=self.tail_class_indices)  # [N, |C_t|]
        blended_tail = (1.0 - alpha) * z_g_tail + alpha * tail_logits_subset
        seg_logits[:, self.tail_class_indices] = blended_tail

        inst_embeddings = self.inst_head(point_feats)

        out: Dict[str, torch.Tensor] = {
            "seg_logits": seg_logits,
            "inst_embeddings": inst_embeddings,
            "tail_feat": tail_feat_norm,  # [N, proto_dim], unit-norm
            "alpha": alpha.squeeze(-1),  # [N]
            "prototypes": self.prototypes,  # [|C_t|, proto_dim]
            "tail_class_indices": self.tail_class_indices,
        }

        if self.top_k is not None and self.top_k > 0:
            scores = torch.norm(point_feats, dim=1)
            k = min(self.top_k, point_feats.shape[0])
            topk_idx = torch.topk(scores, k=k, dim=0).indices
            out["sparse_idx"] = topk_idx
            out["sparse_feats"] = point_feats[topk_idx]
        return out

    @torch.no_grad()
    def ema_update(
        self,
        tail_feat: torch.Tensor,
        labels: torch.Tensor,
        ignore_label: int = -100,
    ) -> int:
        """Update tail prototypes using the live tail features.

        Args:
            tail_feat: ``[N, proto_dim]`` unit-norm features (returned as
                ``tail_feat`` from ``forward``).
            labels: ``[N]`` ground-truth class ids.
            ignore_label: class id to ignore.

        Returns:
            The number of tail prototypes that received an update this step.
        """
        if tail_feat.numel() == 0 or labels.numel() == 0:
            return 0

        labels_flat = labels.view(-1).long()
        valid_mask = labels_flat != int(ignore_label)
        if not bool(valid_mask.any()):
            return 0

        labels_valid = labels_flat[valid_mask]
        feat_valid = tail_feat[valid_mask]

        is_tail = self.global_to_tail_rank[labels_valid] >= 0
        if not bool(is_tail.any()):
            return 0

        feat_tail = feat_valid[is_tail]
        rank_tail = self.global_to_tail_rank[labels_valid[is_tail]]

        # During warmup use a softer momentum so the prototype escapes its
        # random initialisation faster.
        if int(self.ema_step.item()) < self.warmup_steps:
            momentum = self.warmup_momentum
        else:
            momentum = self.ema_momentum

        updated = 0
        # Sort by rank so we can do a per-class mean efficiently.
        unique_ranks = torch.unique(rank_tail)
        for r in unique_ranks.tolist():
            mask = rank_tail == r
            if not bool(mask.any()):
                continue
            mean_feat = feat_tail[mask].mean(dim=0)
            mean_feat = F.normalize(mean_feat, dim=-1)

            if not bool(self.proto_initialized[r]):
                self.prototypes[r] = mean_feat
                self.proto_initialized[r] = True
            else:
                self.prototypes[r] = momentum * self.prototypes[r] + (1.0 - momentum) * mean_feat
                self.prototypes[r] = F.normalize(self.prototypes[r], dim=-1)
            updated += 1

        self.ema_step += 1
        return updated
