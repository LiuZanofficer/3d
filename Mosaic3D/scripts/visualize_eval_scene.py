import argparse
import time
from pathlib import Path

import numpy as np


GT_COLOR = np.array([0, 180, 0], dtype=np.uint8)
PRED_COLOR = np.array([220, 0, 0], dtype=np.uint8)
OVERLAP_COLOR = np.array([255, 220, 0], dtype=np.uint8)
FN_COLOR = np.array([0, 120, 255], dtype=np.uint8)
FP_COLOR = np.array([180, 0, 220], dtype=np.uint8)


def _decode_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def load_dump(pred_file: Path):
    data = np.load(pred_file, allow_pickle=True)
    return {key: _decode_scalar(data[key]) for key in data.files}


def normalize_colors(colors: np.ndarray) -> np.ndarray:
    colors = colors.astype(np.float32)
    if colors.max() > 1.0:
        colors = colors / 255.0
    return np.clip(colors, 0.0, 1.0)


def class_index(class_names: np.ndarray, class_name: str) -> int:
    matches = np.where(class_names == class_name)[0]
    if len(matches) == 0:
        raise ValueError(f"Unknown class '{class_name}'.")
    return int(matches[0])


def summarize_classes(dump):
    class_names = dump["class_names"]
    gt_segment = dump["gt_segment"]
    pred_point_classes = dump["pred_point_classes"]
    pred_mask_classes = dump["pred_mask_classes"]
    pred_mask_scores = dump["pred_mask_scores"]
    pred_masks = dump["pred_masks"]

    rows = []
    for idx, name in enumerate(class_names):
        gt_points = int(np.count_nonzero(gt_segment == idx))
        pred_points = int(np.count_nonzero(pred_point_classes == idx))
        mask_ids = np.where(pred_mask_classes == idx)[0]
        if len(mask_ids):
            pred_mask_points = int(np.count_nonzero(np.any(pred_masks[mask_ids], axis=0)))
            best_score = float(np.max(pred_mask_scores[mask_ids]))
        else:
            pred_mask_points = 0
            best_score = 0.0
        if gt_points or pred_points or pred_mask_points:
            rows.append((name, gt_points, pred_points, len(mask_ids), pred_mask_points, best_score))
    return rows


def print_class_summary(dump):
    scene_name = dump["scene_name"]
    print(f"Scene: {scene_name}")
    print("class\tgt_points\tpred_point_points\tpred_masks\tpred_mask_points\tbest_mask_score")
    for row in summarize_classes(dump):
        print("{}\t{}\t{}\t{}\t{}\t{:.4f}".format(*row))


def build_class_overlay(
    scene_dir: Path,
    dump,
    class_name: str,
    mask_topk: int,
):
    coords = np.load(scene_dir / "coord.npy")
    colors = np.load(scene_dir / "color.npy").astype(np.uint8)
    gt_segment = dump["gt_segment"]
    pred_point_classes = dump["pred_point_classes"]
    pred_mask_classes = dump["pred_mask_classes"]
    pred_mask_scores = dump["pred_mask_scores"]
    pred_masks = dump["pred_masks"]
    class_names = dump["class_names"]

    idx = class_index(class_names, class_name)
    gt_mask = gt_segment == idx
    pred_point_mask = pred_point_classes == idx

    mask_ids = np.where(pred_mask_classes == idx)[0]
    if len(mask_ids):
        order = np.argsort(pred_mask_scores[mask_ids])[::-1]
        mask_ids = mask_ids[order[:mask_topk]]
        pred_instance_mask = np.any(pred_masks[mask_ids], axis=0)
    else:
        pred_instance_mask = np.zeros_like(gt_mask, dtype=bool)

    pred_mask = np.logical_or(pred_point_mask, pred_instance_mask)
    overlap = np.logical_and(gt_mask, pred_mask)
    false_negative = np.logical_and(gt_mask, ~pred_mask)
    false_positive = np.logical_and(~gt_mask, pred_mask)

    overlay = (colors.astype(np.float32) * 0.28).astype(np.uint8)
    overlay[false_negative] = FN_COLOR
    overlay[false_positive] = FP_COLOR
    overlay[gt_mask] = GT_COLOR
    overlay[pred_mask] = PRED_COLOR
    overlay[overlap] = OVERLAP_COLOR

    union = np.count_nonzero(np.logical_or(gt_mask, pred_mask))
    iou = np.count_nonzero(overlap) / union if union else 0.0
    stats = {
        "gt_points": int(np.count_nonzero(gt_mask)),
        "pred_point_points": int(np.count_nonzero(pred_point_mask)),
        "pred_mask_points": int(np.count_nonzero(pred_instance_mask)),
        "overlap_points": int(np.count_nonzero(overlap)),
        "visual_iou": float(iou),
        "num_pred_masks": int(len(mask_ids)),
        "mask_ids": mask_ids.tolist(),
    }
    return coords, normalize_colors(overlay), stats


def visualize(
    scene_dir: Path,
    pred_file: Path,
    class_name: str,
    point_size: float,
    mask_topk: int,
):
    import viser

    dump = load_dump(pred_file)
    coords, colors, stats = build_class_overlay(
        scene_dir,
        dump,
        class_name,
        mask_topk,
    )
    print(f"Scene: {dump['scene_name']}")
    print(f"Class: {class_name}")
    for key, value in stats.items():
        print(f"{key}: {value}")
    print("Colors: green=GT, red=prediction, yellow=overlap, blue=missed GT, purple=false positive")

    server_cls = getattr(viser, "Server", None) or getattr(viser, "ViserServer", None)
    if server_cls is None:
        raise AttributeError("Installed viser has neither Server nor ViserServer.")
    server = server_cls()
    server.scene.add_point_cloud(
        name=f"{dump['scene_name']}::{class_name}",
        points=coords,
        colors=colors,
        point_size=point_size,
    )
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="Inspect Mosaic3D eval predictions for a single validation scene."
    )
    parser.add_argument("--scene-dir", type=Path, required=True)
    parser.add_argument("--pred-file", type=Path, required=True)
    parser.add_argument("--class-name", type=str)
    parser.add_argument("--list-classes", action="store_true")
    parser.add_argument("--point-size", type=float, default=0.02)
    parser.add_argument("--mask-topk", type=int, default=20)
    args = parser.parse_args()

    dump = load_dump(args.pred_file)
    if args.list_classes:
        print_class_summary(dump)
        return
    if not args.class_name:
        raise ValueError("Provide --class-name or use --list-classes.")
    visualize(
        args.scene_dir,
        args.pred_file,
        args.class_name,
        args.point_size,
        args.mask_topk,
    )


if __name__ == "__main__":
    main()
