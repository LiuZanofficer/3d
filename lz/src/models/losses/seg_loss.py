from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegLoss(nn.Module):
    def __init__(
        self,
        ignore_label: int = -100,
        ce_weight: float = 1.0,
        lovasz_weight: float = 0.0,
        label_smoothing: float = 0.0,
        use_class_weight: bool = False,
        class_freq_path: str = "",
        class_weight_mode: str = "inv_sqrt",
        class_weight_clip_min: float = 0.2,
        class_weight_clip_max: float = 5.0,
    ) -> None:
        super().__init__()
        self.ignore_label = ignore_label
        self.ce_weight = float(ce_weight)
        self.lovasz_weight = float(lovasz_weight)
        self.label_smoothing = float(label_smoothing)
        self.use_class_weight = bool(use_class_weight)
        self.class_freq_path = str(class_freq_path)
        self.class_weight_mode = str(class_weight_mode)
        self.class_weight_clip_min = float(class_weight_clip_min)
        self.class_weight_clip_max = float(class_weight_clip_max)
        self.register_buffer("ce_class_weight", torch.empty(0), persistent=False)
        if self.use_class_weight:
            self.ce_class_weight = self._build_class_weight()

    @staticmethod
    def _parse_freq_payload(freq_payload):
        if isinstance(freq_payload, dict):
            for nested_key in ("class_freq", "freq", "class_frequency"):
                nested = freq_payload.get(nested_key, None)
                if isinstance(nested, (dict, list)):
                    return SegLoss._parse_freq_payload(nested)
            parsed = {}
            for key, value in freq_payload.items():
                try:
                    class_idx = int(key)
                except (TypeError, ValueError):
                    continue
                if isinstance(value, dict):
                    value = value.get("count", value.get("freq", value.get("value", 0)))
                parsed[class_idx] = max(float(value), 1.0)
            if parsed:
                max_idx = max(parsed.keys())
                freq = torch.ones(max_idx + 1, dtype=torch.float32)
                for class_idx, value in parsed.items():
                    freq[class_idx] = float(max(value, 1.0))
                return freq
            raise ValueError("Failed to parse class-frequency dict payload.")
        if isinstance(freq_payload, list):
            if len(freq_payload) == 0:
                raise ValueError("Empty class-frequency list payload.")
            freq = torch.ones(len(freq_payload), dtype=torch.float32)
            for idx, value in enumerate(freq_payload):
                freq[idx] = float(max(float(value), 1.0))
            return freq
        raise ValueError(f"Unsupported class-frequency payload type: {type(freq_payload)}")

    def _build_class_weight(self) -> torch.Tensor:
        if not self.class_freq_path:
            raise ValueError("class_freq_path must be provided when use_class_weight=true")
        freq_path = Path(self.class_freq_path)
        if not freq_path.exists():
            raise FileNotFoundError(f"class_freq_path not found: {freq_path}")

        with freq_path.open("r", encoding="utf-8") as f:
            freq_payload = json.load(f)
        freq = self._parse_freq_payload(freq_payload).float().clamp_min(1.0)

        if self.class_weight_mode != "inv_sqrt":
            raise ValueError(f"Unsupported class_weight_mode={self.class_weight_mode}")
        weight = torch.rsqrt(freq)
        weight = weight / weight.mean().clamp_min(1e-12)
        weight = weight.clamp(min=self.class_weight_clip_min, max=self.class_weight_clip_max)
        return weight

    def _get_ce_class_weight(self, num_classes: int, device: torch.device, dtype: torch.dtype):
        if (not self.use_class_weight) or self.ce_class_weight.numel() == 0:
            return None
        weight = self.ce_class_weight
        if weight.numel() < num_classes:
            pad = torch.ones(num_classes - weight.numel(), dtype=weight.dtype, device=weight.device)
            weight = torch.cat([weight, pad], dim=0)
        elif weight.numel() > num_classes:
            weight = weight[:num_classes]
        return weight.to(device=device, dtype=dtype)

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

    def _lovasz_softmax(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        valid = labels != self.ignore_label
        if valid.sum() == 0:
            return logits.sum() * 0.0

        probs = F.softmax(logits[valid], dim=-1)
        labels = labels[valid]
        num_classes = probs.shape[1]

        losses: List[torch.Tensor] = []
        for c in range(num_classes):
            fg = (labels == c).float()
            if fg.sum() == 0:
                continue
            class_pred = probs[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, descending=True)
            fg_sorted = fg[perm]
            grad = self._lovasz_grad(fg_sorted)
            losses.append(torch.dot(errors_sorted, grad))

        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if labels is None or logits.numel() == 0:
            return logits.sum() * 0.0
        if labels.shape[0] != logits.shape[0]:
            raise ValueError(
                f"SegLoss shape mismatch: logits={tuple(logits.shape)}, labels={tuple(labels.shape)}"
            )
        valid_mask = labels != self.ignore_label
        if int(valid_mask.sum()) == 0:
            return logits.sum() * 0.0

        total_loss = logits.sum() * 0.0

        if self.ce_weight > 0.0:
            ce_class_weight = self._get_ce_class_weight(
                num_classes=int(logits.shape[1]),
                device=logits.device,
                dtype=logits.dtype,
            )
            ce_loss = F.cross_entropy(
                logits,
                labels,
                ignore_index=self.ignore_label,
                label_smoothing=max(self.label_smoothing, 0.0),
                weight=ce_class_weight,
            )
            total_loss = total_loss + self.ce_weight * ce_loss

        if self.lovasz_weight > 0.0:
            lovasz_loss = self._lovasz_softmax(logits, labels)
            total_loss = total_loss + self.lovasz_weight * lovasz_loss

        return total_loss
