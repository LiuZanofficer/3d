"""Analyze where a class's points go wrong in a Mosaic3D eval dump.

For a chosen class C in a single eval-scene dump (.npz), this reports:
  - overview: GT / TP / FN / FP point counts and point-level IoU
  - FN breakdown: the missed GT-C points were predicted as which classes
    (answers "missed cabinet -> predicted as what?")
  - FP breakdown: the points wrongly predicted as C were actually which classes

Dump files are produced by language_module._dump_eval_scene_prediction when
the env var MOSAIC3D_DUMP_EVAL=1 is set during eval. They live under
<output_dir>/eval_scene_dumps/<postfix>/<scene>.npz and contain:
  gt_segment, pred_point_classes, pred_mask_classes, pred_mask_scores,
  pred_masks, class_names, scene_name, gt_instance.

This script is read-only and only needs numpy.
"""

import argparse
from pathlib import Path

import numpy as np


def _decode_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def load_dump(pred_file: Path):
    data = np.load(pred_file, allow_pickle=True)
    return {key: _decode_scalar(data[key]) for key in data.files}


def class_index(class_names: np.ndarray, class_name: str) -> int:
    matches = np.where(class_names == class_name)[0]
    if len(matches) == 0:
        raise ValueError(
            f"Unknown class '{class_name}'. Use --list-classes to see available names."
        )
    return int(matches[0])


def name_of(class_names: np.ndarray, idx: int) -> str:
    """Safely map a class index to a name (handles ignore/out-of-range labels)."""
    if 0 <= idx < len(class_names):
        return str(class_names[idx])
    return f"<ignore/invalid:{idx}>"


def resolve_pred_points(dump, pred_mode: str):
    """Return (pred_point_classes, mode_used).

    semantic: use `pred_semantic` (argmax over foreground classes only) so the
              TP/FN/FP match the f-mIoU scoring path. Falls back to argmax with a
              warning if the dump predates the extended format.
    argmax:   use `pred_point_classes` (raw full-vocabulary argmax).
    """
    if pred_mode == "semantic":
        if "pred_semantic" in dump:
            return dump["pred_semantic"], "semantic"
        print("[warn] dump has no 'pred_semantic' (old dump); falling back to --pred argmax.")
        return dump["pred_point_classes"], "argmax(fallback)"
    return dump["pred_point_classes"], "argmax"


def valid_point_mask(dump, pred_mode: str) -> np.ndarray:
    """Points that count toward semantic scoring: drop ignore_label and (semantic
    mode) background-class GT, matching the foreground f-mIoU convention.
    argmax mode keeps the legacy behavior (all points valid)."""
    gt_segment = dump["gt_segment"]
    valid = np.ones(gt_segment.shape, dtype=bool)
    if pred_mode != "semantic":
        return valid
    ignore_label = int(dump["ignore_label"]) if "ignore_label" in dump else -100
    valid &= gt_segment != ignore_label
    if "bg_class_idx" in dump:
        bg = np.asarray(dump["bg_class_idx"]).reshape(-1)
        if bg.size:
            valid &= ~np.isin(gt_segment, bg)
    return valid


def predicted_class_mask(dump, idx: int, use_mask: bool, pred_points: np.ndarray) -> np.ndarray:
    """Boolean per-point mask of points predicted as class `idx`."""
    pred_point_mask = pred_points == idx

    if not use_mask:
        return pred_point_mask

    pred_mask_classes = dump["pred_mask_classes"]
    pred_masks = dump["pred_masks"]
    mask_ids = np.where(pred_mask_classes == idx)[0]
    if len(mask_ids):
        pred_instance_mask = np.any(pred_masks[mask_ids], axis=0)
    else:
        pred_instance_mask = np.zeros_like(pred_point_mask, dtype=bool)
    return np.logical_or(pred_point_mask, pred_instance_mask)


def distribution(labels: np.ndarray, class_names: np.ndarray, topk: int):
    """Return [(name, count, pct), ...] sorted by count desc."""
    total = int(labels.size)
    if total == 0:
        return [], 0
    values, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    rows = []
    for j in order[:topk] if topk else order:
        v = int(values[j])
        c = int(counts[j])
        rows.append((name_of(class_names, v), c, 100.0 * c / total))
    return rows, total


def list_classes(dump):
    class_names = dump["class_names"]
    gt_segment = dump["gt_segment"]
    pred_point_classes = dump["pred_point_classes"]
    print(f"Scene: {dump['scene_name']}")
    print("idx\tclass\tgt_points\tpred_point_points")
    for idx, name in enumerate(class_names):
        gt_points = int(np.count_nonzero(gt_segment == idx))
        pred_points = int(np.count_nonzero(pred_point_classes == idx))
        if gt_points or pred_points:
            print(f"{idx}\t{name}\t{gt_points}\t{pred_points}")


def analyze(dump, class_name: str, use_mask: bool, topk: int, pred_mode: str):
    class_names = dump["class_names"]
    gt_segment = dump["gt_segment"]
    pred_points, mode_used = resolve_pred_points(dump, pred_mode)
    valid = valid_point_mask(dump, pred_mode)

    idx = class_index(class_names, class_name)
    gt_mask = (gt_segment == idx) & valid
    pred_mask = predicted_class_mask(dump, idx, use_mask, pred_points) & valid

    tp = np.logical_and(gt_mask, pred_mask)
    fn = np.logical_and(gt_mask, ~pred_mask)
    fp = np.logical_and(~gt_mask, pred_mask)

    n_gt = int(np.count_nonzero(gt_mask))
    n_pred = int(np.count_nonzero(pred_mask))
    n_tp = int(np.count_nonzero(tp))
    n_fn = int(np.count_nonzero(fn))
    n_fp = int(np.count_nonzero(fp))
    union = n_tp + n_fn + n_fp
    iou = n_tp / union if union else 0.0

    base = "per-point argmax + instance masks" if use_mask else "per-point"
    src = f"{base} [{mode_used}]"
    print("=" * 64)
    print(f"Scene : {dump['scene_name']}")
    print(f"Class : {class_name}  (idx={idx})")
    print(f"Pred source: {src}")
    print("-" * 64)
    print(f"GT points        : {n_gt}")
    print(f"Predicted points : {n_pred}")
    print(f"TP (correct)     : {n_tp}")
    print(f"FN (missed)      : {n_fn}")
    print(f"FP (wrong)       : {n_fp}")
    print(f"point-IoU        : {iou:.4f}")
    print("-" * 64)

    print(f"[FN] missed '{class_name}' points were predicted as:")
    rows, total = distribution(pred_points[fn], class_names, topk)
    if total == 0:
        print("   (none)")
    else:
        for name, c, pct in rows:
            print(f"   {name:<24} {c:>8}  ({pct:5.1f}%)")
    print("-" * 64)

    print(f"[FP] points wrongly predicted as '{class_name}' are actually:")
    rows, total = distribution(gt_segment[fp], class_names, topk)
    if total == 0:
        print("   (none)")
    else:
        for name, c, pct in rows:
            print(f"   {name:<24} {c:>8}  ({pct:5.1f}%)")
    print("=" * 64)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze FN/FP predicted-label distribution for one class in a Mosaic3D eval dump."
    )
    parser.add_argument("--pred-file", type=Path, required=True, help="Path to <scene>.npz dump.")
    parser.add_argument("--class-name", type=str, help="Class to analyze, e.g. cabinet.")
    parser.add_argument("--list-classes", action="store_true", help="List classes present in the scene.")
    parser.add_argument("--topk", type=int, default=15, help="Show top-K classes in each breakdown (0=all).")
    parser.add_argument(
        "--use-mask",
        action="store_true",
        help="Combine per-point argmax with predicted instance masks (default: per-point only).",
    )
    parser.add_argument(
        "--pred",
        choices=["semantic", "argmax"],
        default="semantic",
        help="Prediction source. semantic=foreground-only argmax aligned with f-mIoU "
        "(default); argmax=raw full-vocabulary argmax (legacy, counts ignore/bg).",
    )
    args = parser.parse_args()

    dump = load_dump(args.pred_file)
    if args.list_classes:
        list_classes(dump)
        return
    if not args.class_name:
        raise ValueError("Provide --class-name, or use --list-classes.")
    analyze(dump, args.class_name, args.use_mask, args.topk, args.pred)


if __name__ == "__main__":
    main()
