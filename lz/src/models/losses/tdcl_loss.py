"""Tail-Decoupled Contrastive Loss (TDCL).

Three additive terms:

* **GBS** (Group-Balanced Softmax): a weighted cross-entropy with class
  weights ``w_c = 1 / sqrt(n_c)`` (mean-normalised then clipped). Implemented
  by reusing the existing ``SegLoss`` weighted-CE machinery.

* **TFR** (Tail-Focused Region loss): a Lovász-Softmax surrogate evaluated
  *only* over the tail classes. Standard Lovász-Softmax averages a per-class
  IoU term across all classes; we restrict the per-class loop to tail
  classes so that tail recall directly drives the gradient.

* **PS** (Prototype Separation): a supervised contrastive loss with a margin
  pulling each tail-class point toward its prototype while pushing it away
  from the K most-confused head-class prototypes. The "head confuser" set is
  re-derived once per epoch from a streaming confusion matrix maintained by
  ``LZLitModule``.

The class is signature-compatible with ``SegLoss``: callers may pass
``logits, labels`` and the loss will silently skip TFR / PS when the
required side inputs are missing.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.losses.seg_loss import SegLoss


class TDCLLoss(nn.Module):
    def __init__(
        self,
        num_classes: int,
        tail_class_indices: Sequence[int],
        ignore_label: int = -100,
        # GBS
        ce_weight: float = 1.0,
        label_smoothing: float = 0.0,
        class_freq_path: str = "",
        class_weight_clip_min: float = 0.2,
        class_weight_clip_max: float = 5.0,
        # TFR
        tfr_weight: float = 0.5,
        # PS
        ps_weight: float = 0.2,
        ps_margin: float = 0.3,
        ps_top_k: int = 5,
        ps_temperature: float = 0.1,
        # GBS scaling factor for back-compat with TDCL spec (lambda_g)
        gbs_weight: float = 1.0,
        # Auxiliary BCE supervision on TDP gating alpha (1=tail, 0=non-tail).
        # Breaks the chicken-and-egg deadlock where alpha collapses to 0 and
        # the tail branch never receives gradient. Recommended ~0.05.
        alpha_sup_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.ignore_label = int(ignore_label)

        self.gbs_weight = float(gbs_weight)
        self.tfr_weight = float(tfr_weight)
        self.ps_weight = float(ps_weight)
        self.ps_margin = float(ps_margin)
        self.ps_top_k = int(ps_top_k)
        self.ps_temperature = float(ps_temperature)
        self.alpha_sup_weight = float(alpha_sup_weight)

        if len(tail_class_indices) == 0:
            raise ValueError("TDCLLoss requires at least one tail class index.")
        tail_idx = torch.as_tensor(
            sorted(set(int(c) for c in tail_class_indices)), dtype=torch.long
        )
        self.register_buffer("tail_class_indices", tail_idx, persistent=True)

        # Bool mask over global class space, marks tail positions.
        is_tail_class = torch.zeros(self.num_classes, dtype=torch.bool)
        is_tail_class[tail_idx] = True
        self.register_buffer("is_tail_class", is_tail_class, persistent=True)

        # Reuse SegLoss for GBS (weighted CE). Lovász is disabled here because we
        # implement TFR ourselves with the tail-only per-class loop.
        self.gbs = SegLoss(
            ignore_label=ignore_label,
            ce_weight=ce_weight,
            lovasz_weight=0.0,
            label_smoothing=label_smoothing,
            use_class_weight=bool(class_freq_path),
            class_freq_path=str(class_freq_path),
            class_weight_mode="inv_sqrt",
            class_weight_clip_min=class_weight_clip_min,
            class_weight_clip_max=class_weight_clip_max,
        )

        # Head-confuser table: for each tail class, a list of head-class indices.
        # Initially empty — populated by ``update_head_confuser`` after the first
        # train epoch using the running confusion matrix.
        head_confuser = torch.full(
            (int(tail_idx.numel()), self.ps_top_k), -1, dtype=torch.long
        )
        self.register_buffer("head_confuser", head_confuser, persistent=True)

        # Cache global-id -> tail-rank.
        global_to_tail_rank = torch.full((self.num_classes,), -1, dtype=torch.long)
        global_to_tail_rank[tail_idx] = torch.arange(int(tail_idx.numel()), dtype=torch.long)
        self.register_buffer("global_to_tail_rank", global_to_tail_rank, persistent=True)

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _lovasz_grad(gt_sorted: torch.Tensor) -> torch.Tensor:
        p = gt_sorted.numel()
        if p == 0:
            return gt_sorted
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1.0 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / (union + 1e-7)
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0 : p - 1]
        return jaccard

    def _tail_lovasz(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels != self.ignore_label
        if not bool(valid.any()):
            return logits.sum() * 0.0

        probs = F.softmax(logits[valid], dim=-1)
        labels_valid = labels[valid]

        losses: List[torch.Tensor] = []
        for c in self.tail_class_indices.tolist():
            fg = (labels_valid == int(c)).float()
            if fg.sum() == 0:
                # Class absent in this batch — skip rather than zero-add.
                continue
            class_pred = probs[:, int(c)]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = self._lovasz_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, grad))

        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()

    def _proto_separation(
        self,
        tail_feat: torch.Tensor,
        labels: torch.Tensor,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Margin-based supervised contrastive on tail-class points.

        For each point ``i`` whose label ``y_i`` is in ``C_t``:

            L_i = max(0, margin - sim(f_i, q_{rank(y_i)}) + max_k sim(f_i, q^h_{i,k}))

        where ``q^h_{i,k}`` is the k-th *negative anchor* for the tail class
        ``y_i``.

        IMPORTANT — head-prototype approximation
        ----------------------------------------
        Strictly speaking the paper specifies head-class prototypes as the
        negative anchors. This implementation maintains **tail prototypes
        only** (cf. ``TDPDecoder.prototypes``), so for each confused head
        class id ``h`` returned by ``update_head_confuser`` we substitute the
        *nearest tail prototype to ``h``* (by class-id distance) as a
        stand-in negative anchor. This makes PS act as a 1st-order
        approximation: it still pulls each tail point toward its own
        prototype and pushes it away from a tail-prototype proxy of the
        confused head class. Maintaining true head prototypes is left as
        future work (cf. paper Section 5.4 limitations).

        If no head-confuser entries are valid yet (first epoch, before
        ``update_head_confuser`` has been called), we fall back to the
        hardest *other-tail* prototype as the negative anchor.
        """
        if tail_feat.numel() == 0 or labels.numel() == 0:
            return tail_feat.sum() * 0.0
        if prototypes.numel() == 0:
            return tail_feat.sum() * 0.0

        labels_flat = labels.view(-1).long()
        valid = labels_flat != int(self.ignore_label)
        if not bool(valid.any()):
            return tail_feat.sum() * 0.0
        labels_valid = labels_flat[valid]
        feat_valid = tail_feat[valid]

        rank = self.global_to_tail_rank[labels_valid]
        is_tail = rank >= 0
        if not bool(is_tail.any()):
            return tail_feat.sum() * 0.0

        feat_tail = feat_valid[is_tail]
        rank_tail = rank[is_tail]

        # Match prototype dtype to tail-feature dtype so every downstream op
        # (matmul, scatter, max, indexed assignment) stays consistent under
        # bf16/fp16 mixed-precision training. Prototypes are a non-trainable
        # buffer, so casting is safe (no autograd implications).
        proto_norm = F.normalize(prototypes.to(feat_tail.dtype), dim=-1)
        feat_tail_norm = F.normalize(feat_tail, dim=-1)

        # Positive similarity = sim with own prototype.
        own_proto = proto_norm[rank_tail]  # [M, D]
        pos_sim = (feat_tail_norm * own_proto).sum(dim=-1)  # [M]

        # Negative similarity from head confusers.
        head_confuser_table = self.head_confuser[rank_tail]  # [M, K]
        valid_neg = head_confuser_table >= 0  # [M, K]
        if bool(valid_neg.any()):
            # Map head class -> head prototype isn't supported here because
            # we only carry tail prototypes. The PS loss therefore uses the
            # head_confuser table to *index into the same prototype bank* but
            # with the convention: head_confuser stores head class ids
            # mapped to the closest tail prototype, used as a stand-in
            # negative anchor. (Approximation: we do not maintain head
            # prototypes; the table is filled with tail-rank ids of the
            # confused tail classes by ``update_head_confuser``.)
            safe_idx = head_confuser_table.clamp(min=0)
            neg_protos = proto_norm[safe_idx]  # [M, K, D]
            neg_sim = (feat_tail_norm.unsqueeze(1) * neg_protos).sum(dim=-1)  # [M, K]
            neg_sim = neg_sim.masked_fill(~valid_neg, float("-inf"))
            hardest_neg = neg_sim.max(dim=-1).values  # [M]
            no_neg = ~valid_neg.any(dim=-1)
            if bool(no_neg.any()):
                # Replace -inf rows with mean similarity to other tail prototypes.
                others_mask = torch.ones(proto_norm.shape[0], dtype=torch.bool, device=proto_norm.device)
                # mean across all prototypes excluding self proto for points with no_neg
                idx_no_neg = torch.nonzero(no_neg, as_tuple=False).flatten()
                if idx_no_neg.numel() > 0:
                    f_no = feat_tail_norm[idx_no_neg]
                    sim_all = f_no @ proto_norm.t()  # [m, |C_t|]
                    own = rank_tail[idx_no_neg].unsqueeze(-1)
                    sim_all.scatter_(dim=1, index=own, value=float("-inf"))
                    fallback = sim_all.max(dim=1).values
                    # Force exact dtype match: under autocast, neg_sim.max
                    # and matmul-then-max can come out at different
                    # precisions even when inputs share dtype. Casting
                    # ``fallback`` to ``hardest_neg.dtype`` here makes the
                    # in-place index assignment robust to autocast
                    # promotion rules.
                    hardest_neg[idx_no_neg] = fallback.to(hardest_neg.dtype)
        else:
            # No head confuser yet — fall back to hardest *other* tail prototype.
            sim_all = feat_tail_norm @ proto_norm.t()  # [M, |C_t|]
            own = rank_tail.unsqueeze(-1)
            sim_all.scatter_(dim=1, index=own, value=float("-inf"))
            hardest_neg = sim_all.max(dim=-1).values

        loss = F.relu(self.ps_margin - pos_sim + hardest_neg)
        return loss.mean()

    def _alpha_supervision(
        self,
        alpha: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """BCE supervision on TDP gating: target=1 for tail points, 0 otherwise.

        Without this auxiliary signal, alpha tends to collapse toward 0 in
        early training (because the tail branch initially produces noise via
        cosine-similarity to randomly-initialised prototypes, so the optimizer
        gates it out — but then the tail branch receives no gradient and
        prototypes never learn). A small BCE term breaks this deadlock by
        directly pushing alpha toward 1 on tail-labelled points and toward 0
        on head/common-labelled points.
        """
        labels_flat = labels.view(-1).long()
        alpha_flat = alpha.view(-1)
        valid = labels_flat != int(self.ignore_label)
        if not bool(valid.any()):
            return alpha_flat.sum() * 0.0
        labels_valid = labels_flat[valid]
        alpha_valid = alpha_flat[valid].clamp(min=1e-6, max=1.0 - 1e-6)
        # Restrict to in-range labels (defensive — ignore-label already filtered).
        in_range = (labels_valid >= 0) & (labels_valid < self.num_classes)
        if not bool(in_range.any()):
            return alpha_flat.sum() * 0.0
        target = self.is_tail_class[labels_valid[in_range]].to(alpha_valid.dtype)
        alpha_use = alpha_valid[in_range]
        return F.binary_cross_entropy(alpha_use, target)

    # ----------------------------------------------------------------- public
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tail_feat: Optional[torch.Tensor] = None,
        prototypes: Optional[torch.Tensor] = None,
        alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if labels is None or logits.numel() == 0:
            return logits.sum() * 0.0

        gbs = self.gbs(logits, labels)

        if self.tfr_weight > 0.0:
            tfr = self._tail_lovasz(logits, labels)
        else:
            tfr = logits.sum() * 0.0

        if (
            self.ps_weight > 0.0
            and tail_feat is not None
            and prototypes is not None
            and prototypes.numel() > 0
        ):
            ps = self._proto_separation(tail_feat, labels, prototypes)
        else:
            ps = logits.sum() * 0.0

        total = self.gbs_weight * gbs + self.tfr_weight * tfr + self.ps_weight * ps

        if self.alpha_sup_weight > 0.0 and alpha is not None and alpha.numel() > 0:
            alpha_loss = self._alpha_supervision(alpha, labels)
            total = total + self.alpha_sup_weight * alpha_loss

        return total

    @torch.no_grad()
    def update_head_confuser(self, conf_matrix: torch.Tensor) -> int:
        """Refresh the head-confuser table from the running confusion matrix.

        For each tail class ``t``, we look at the *top-K* head/common classes
        most often wrongly predicted as ``t`` (i.e., column ``t`` of the
        confusion matrix, rows restricted to non-tail classes).

        IMPORTANT — head-prototype approximation
        ----------------------------------------
        The PS loss design assumes head-class **prototypes** as negative
        anchors, but this implementation only maintains tail prototypes. As a
        1st-order proxy, each confused head class id ``h`` is mapped to the
        *nearest tail rank* (by absolute class-id distance) and stored in the
        ``head_confuser`` table. ``_proto_separation`` then pulls the tail
        point toward its own prototype and pushes it away from the
        tail-prototype proxies of the confused head classes. This is a
        deliberate simplification documented in the paper Section 5.4
        limitations; a full head-prototype implementation is left as future
        work.

        Stored layout: ``head_confuser[tail_rank, k]`` is the **tail rank** of
        the closest tail prototype that stands in for the k-th confused head
        class. Empty slots are ``-1``.

        Args:
            conf_matrix: ``[num_classes, num_classes]`` long tensor with rows
                = ground truth, cols = predictions (matches the validation
                confusion matrix in ``LZLitModule``).

        Returns:
            The number of tail classes whose confuser row was updated.
        """
        if conf_matrix.numel() == 0:
            return 0
        device = conf_matrix.device
        conf = conf_matrix.float()
        # For tail class t, look at the *predicted-as-t* column with rows
        # restricted to non-tail classes (head/common confusers).
        non_tail_mask = ~self.is_tail_class.to(device)  # [num_classes]
        new_table = self.head_confuser.detach().clone().fill_(-1).to(device)

        updated = 0
        for tail_rank, t in enumerate(self.tail_class_indices.tolist()):
            col = conf[:, int(t)] * non_tail_mask.float()
            if not bool((col > 0).any()):
                continue
            k = min(self.ps_top_k, int((col > 0).sum().item()))
            top_classes = torch.topk(col, k=k).indices  # head-class ids

            # Without head prototypes, map each confused head class to its
            # nearest tail prototype as a stand-in negative anchor. The
            # nearest tail prototype is identified by class-frequency proxy:
            # the closest tail class id by label distance (a coarse but
            # gradient-free heuristic). This is the simplest fallback and
            # avoids needing head-class prototypes.
            for slot, head_cls in enumerate(top_classes.tolist()):
                # Map to tail rank by nearest tail class id.
                tail_ids = self.tail_class_indices.tolist()
                nearest_tail_rank = min(
                    range(len(tail_ids)), key=lambda r: abs(tail_ids[r] - int(head_cls))
                )
                new_table[tail_rank, slot] = int(nearest_tail_rank)
            updated += 1

        self.head_confuser.copy_(new_table.to(self.head_confuser.dtype))
        return updated
