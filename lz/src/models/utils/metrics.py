from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch


def update_confusion_matrix(
    conf: torch.Tensor,
    preds: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_label: int,
) -> torch.Tensor:
    """Update confusion matrix with a batch."""
    preds = preds.view(-1).long()
    labels = labels.view(-1).long()
    mask = labels != ignore_label
    mask = mask & (preds >= 0) & (preds < num_classes)
    preds = preds[mask]
    labels = labels[mask]
    if preds.numel() == 0:
        return conf
    idx = labels * num_classes + preds
    bincount = torch.bincount(idx, minlength=num_classes * num_classes)
    conf += bincount.reshape(num_classes, num_classes)
    return conf


def compute_iou_from_conf(conf: torch.Tensor) -> torch.Tensor:
    """Compute IoU per class from confusion matrix."""
    tp = torch.diag(conf)
    fp = conf.sum(dim=0) - tp
    fn = conf.sum(dim=1) - tp
    denom = tp + fp + fn
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(denom, dtype=torch.float32))
    return iou


def compute_subset_miou(
    iou: torch.Tensor,
    class_names: List[str],
    subset_mapper: Optional[Dict],
) -> Dict[str, float]:
    if subset_mapper is None or "subset_names" not in subset_mapper:
        return {}
    subset_names = subset_mapper["subset_names"]
    values = {}
    for subset in subset_names:
        idxs = [i for i, name in enumerate(class_names) if subset_mapper.get(name) == subset]
        if len(idxs) == 0:
            continue
        values[f"{subset}_miou"] = float(iou[idxs].mean().item())
    return values
