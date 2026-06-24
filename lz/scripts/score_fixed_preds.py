from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.data.scannet.dataset import ScanNet200Dataset
from src.models.utils.metrics import compute_iou_from_conf, update_confusion_matrix


def _read_split(split_file: Path) -> list[str]:
    with split_file.open("r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]


def _check_files(pred_dir: Path, scenes: list[str]) -> None:
    pred_scenes = {p.name[: -len(".pred.npy")] for p in pred_dir.glob("*.pred.npy")}
    gt_scenes = {p.name[: -len(".gt.npy")] for p in pred_dir.glob("*.gt.npy")}
    expected = set(scenes)
    miss_pred = sorted(expected - pred_scenes)
    miss_gt = sorted(expected - gt_scenes)
    extra_pred = sorted(pred_scenes - expected)
    extra_gt = sorted(gt_scenes - expected)
    if miss_pred or miss_gt or extra_pred or extra_gt:
        raise RuntimeError(
            "Prediction files do not match split.\n"
            f"miss_pred={len(miss_pred)} miss_gt={len(miss_gt)} "
            f"extra_pred={len(extra_pred)} extra_gt={len(extra_gt)}"
        )


def score_fixed_preds(pred_dir: Path, split_file: Path, scannet_root: Path, output_json: Path) -> dict:
    scenes = _read_split(split_file)
    _check_files(pred_dir, scenes)

    dataset = ScanNet200Dataset(
        data_dir=str(scannet_root),
        split="val",
        transforms=None,
    )
    fg_idx = list(dataset.fg_class_idx)
    bg_idx = list(dataset.bg_class_idx)
    ignore_label = int(dataset.ignore_label)
    num_classes = len(dataset.CLASS_LABELS)

    conf_fg = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    conf_all = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    total_points = 0
    for scene in scenes:
        pred = np.load(pred_dir / f"{scene}.pred.npy")
        gt = np.load(pred_dir / f"{scene}.gt.npy")
        if pred.shape != gt.shape:
            raise RuntimeError(
                f"Shape mismatch in {scene}: pred={pred.shape} gt={gt.shape}"
            )
        pred_t = torch.from_numpy(pred.astype(np.int64))
        gt_t = torch.from_numpy(gt.astype(np.int64))
        total_points += int(pred_t.numel())

        conf_all = update_confusion_matrix(
            conf_all,
            pred_t,
            gt_t,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )

        gt_fg = gt_t.clone()
        for cls_idx in bg_idx:
            gt_fg[gt_fg == cls_idx] = ignore_label
        conf_fg = update_confusion_matrix(
            conf_fg,
            pred_t,
            gt_fg,
            num_classes=num_classes,
            ignore_label=ignore_label,
        )

    iou_fg = compute_iou_from_conf(conf_fg.float())
    iou_all = compute_iou_from_conf(conf_all.float())
    miou_fg_mosaic = float(iou_fg[fg_idx].mean().item())
    miou_all_lz = float(iou_all.mean().item())

    out = {
        "split_file": str(split_file),
        "pred_dir": str(pred_dir),
        "scannet_root": str(scannet_root),
        "num_scenes": len(scenes),
        "num_points": total_points,
        "num_classes": num_classes,
        "num_fg_classes": len(fg_idx),
        "ignore_label": ignore_label,
        "miou_fg_mosaic": miou_fg_mosaic,
        "miou_all_lz": miou_all_lz,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Score fixed scene predictions with two metric protocols.")
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--split_file", type=Path, required=True)
    parser.add_argument("--scannet_root", type=Path, required=True)
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()

    result = score_fixed_preds(
        pred_dir=args.pred_dir,
        split_file=args.split_file,
        scannet_root=args.scannet_root,
        output_json=args.output_json,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
