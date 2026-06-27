"""Aggregate error analysis over the whole validation set of a Mosaic3D eval run.

This reads every <scene>.npz under an eval dump dir, rebuilds the foreground
confusion matrix exactly the way validation_epoch_end does, and produces:

  - a fg-mIoU self-check (must match the eval-reported value, e.g. ~15.x), so we
    know the analysis is on the same scoring footing as the metric;
  - per_class_report.csv: gt/pred counts, IoU/recall/precision and a "cost"
    (lost GT points = gt_count*(1-recall)) for ranking the most worth-looking
    classes;
  - fn_destinations.md: for the top target classes, where their GT points were
    predicted instead (concentration -> likely synonym confusion; diffuse ->
    feature problem);
  - targets.md: ranked targets, dead classes (never predicted), and the worst
    scenes per target with ready-to-run visualization commands.

Scoring convention (mirrors language_module.validation_step / _epoch_end):
  pred = pred_semantic (argmax over foreground classes only)
  gt   = gt_segment with bg_class_idx mapped to ignore_label, ignore dropped
  IoU_c = tp / (tp + fp + fn)  (0 when union == 0, matching the codebase)
  fg-mIoU = mean(IoU_c for c in fg_class_idx)

Requires the *extended* dump (pred_semantic + fg/bg idx + ignore_label), i.e.
produced after the dump change. Read-only; needs only numpy.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


def _decode_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def load_dump(pred_file: Path):
    data = np.load(pred_file, allow_pickle=True)
    return {key: _decode_scalar(data[key]) for key in data.files}


def scene_confmat(gt_fg: np.ndarray, pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Confusion matrix [N,N] (rows=gt, cols=pred) over valid points only."""
    flat = gt_fg.astype(np.int64) * num_classes + pred.astype(np.int64)
    return np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def main():
    parser = argparse.ArgumentParser(description="Whole-val aggregate error analysis for a Mosaic3D eval dump.")
    parser.add_argument("--dump-dir", type=Path, required=True,
                        help="eval_scene_dumps/<postfix> dir containing <scene>.npz files.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where to write reports.")
    parser.add_argument("--topn", type=int, default=20, help="How many target classes to detail.")
    parser.add_argument("--fn-topk", type=int, default=8, help="Top-K FN destinations per class.")
    parser.add_argument("--worst-scenes", type=int, default=5, help="Worst scenes listed per target class.")
    parser.add_argument("--min-scene-gt", type=int, default=50,
                        help="Ignore scenes with fewer than this many GT points of the class when ranking worst scenes.")
    parser.add_argument("--scene-root", type=str, default="/datasets/mosaic3d/data/scannet",
                        help="Scene data root, used to compose suggested visualization commands.")
    args = parser.parse_args()

    files = sorted(args.dump_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz dumps under {args.dump_dir}")

    # peek first dump for class metadata
    first = load_dump(files[0])
    if "pred_semantic" not in first:
        raise SystemExit(
            "Dump has no 'pred_semantic'. These are old dumps; re-run eval with the "
            "extended dump (MOSAIC3D_DUMP_EVAL=1 after the language_module change)."
        )
    class_names = np.asarray(first["class_names"])
    num_classes = len(class_names)
    fg_idx = np.asarray(first["fg_class_idx"]).reshape(-1).astype(int)
    bg_idx = np.asarray(first.get("bg_class_idx", np.array([]))).reshape(-1).astype(int)
    ignore_label = int(first.get("ignore_label", -100))
    is_fg = np.zeros(num_classes, dtype=bool)
    is_fg[fg_idx] = True

    global_cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    # per-scene per-class stats for worst-scene ranking
    scene_names = []
    per_scene_tp = []   # [S, N]
    per_scene_row = []  # gt counts [S, N]
    per_scene_col = []  # pred counts [S, N]

    bg_set = set(int(b) for b in bg_idx)
    for f in files:
        d = load_dump(f)
        gt = np.asarray(d["gt_segment"]).astype(np.int64)
        pred = np.asarray(d["pred_semantic"]).astype(np.int64)
        # map background gt -> ignore (matches segment_fg construction)
        gt_fg = gt.copy()
        if bg_set:
            gt_fg[np.isin(gt_fg, bg_idx)] = ignore_label
        valid = (gt_fg != ignore_label) & (gt_fg >= 0) & (gt_fg < num_classes) \
            & (pred >= 0) & (pred < num_classes)
        cm = scene_confmat(gt_fg[valid], pred[valid], num_classes)
        global_cm += cm
        scene_names.append(str(d["scene_name"]))
        diag = np.diag(cm)
        per_scene_tp.append(diag)
        per_scene_row.append(cm.sum(axis=1))
        per_scene_col.append(cm.sum(axis=0))

    per_scene_tp = np.stack(per_scene_tp)    # [S, N]
    per_scene_row = np.stack(per_scene_row)
    per_scene_col = np.stack(per_scene_col)

    tp = np.diag(global_cm).astype(np.float64)
    gt_count = global_cm.sum(axis=1).astype(np.float64)   # row
    pred_count = global_cm.sum(axis=0).astype(np.float64)  # col
    fp = pred_count - tp
    fn = gt_count - tp
    union = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, 0.0)
        recall = np.where(gt_count > 0, tp / gt_count, 0.0)
        precision = np.where(pred_count > 0, tp / pred_count, 0.0)
    cost = gt_count * (1.0 - recall)  # lost GT points == fn

    # fg-mIoU self-check: IoU=0 when union==0 (matches codebase, NOT nan-skip)
    fg_miou = float(np.mean([iou[i] for i in fg_idx]))

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- per_class_report.csv ----
    csv_path = args.out_dir / "per_class_report.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "class", "is_fg", "gt_count", "pred_count",
                    "tp", "fp", "fn", "iou", "recall", "precision", "cost"])
        for i in range(num_classes):
            w.writerow([i, class_names[i], int(is_fg[i]), int(gt_count[i]), int(pred_count[i]),
                        int(tp[i]), int(fp[i]), int(fn[i]),
                        f"{iou[i]:.4f}", f"{recall[i]:.4f}", f"{precision[i]:.4f}", int(cost[i])])

    # ranking among fg classes by cost (lost points) desc
    fg_order = [i for i in fg_idx]
    fg_order.sort(key=lambda i: cost[i], reverse=True)
    targets = fg_order[: args.topn]

    # dead classes: fg, has GT, but essentially never predicted
    dead = [i for i in fg_idx if gt_count[i] > 0 and pred_count[i] <= max(1, 0.001 * gt_count[i])]
    dead.sort(key=lambda i: gt_count[i], reverse=True)

    # ---- fn_destinations.md ----
    fn_path = args.out_dir / "fn_destinations.md"
    with open(fn_path, "w") as fh:
        fh.write(f"# FN destinations (top {args.topn} target classes by cost)\n\n")
        fh.write("For each class C, where did its GT points get predicted (excluding the correct "
                 "TP)? High top-1 share => synonym/confusion; diffuse => feature/recall problem.\n\n")
        for i in targets:
            row = global_cm[i].astype(np.float64).copy()
            tp_i = row[i]
            row[i] = 0  # drop TP -> keep only the misses
            total_fn = row.sum()
            fh.write(f"## {class_names[i]} (idx={i})\n")
            fh.write(f"- gt={int(gt_count[i])} pred={int(pred_count[i])} IoU={iou[i]:.4f} "
                     f"recall={recall[i]:.4f} precision={precision[i]:.4f} cost={int(cost[i])}\n")
            if total_fn <= 0:
                fh.write("- (no FN)\n\n")
                continue
            order = np.argsort(row)[::-1]
            top1 = row[order[0]] / total_fn if total_fn else 0.0
            fh.write(f"- FN total={int(total_fn)}, top-1 destination share={top1:.1%}\n")
            for j in order[: args.fn_topk]:
                c = row[j]
                if c <= 0:
                    break
                fh.write(f"    - {class_names[j]}: {int(c)} ({100.0 * c / total_fn:.1f}%)\n")
            fh.write("\n")

    # ---- targets.md ----
    tg_path = args.out_dir / "targets.md"
    with open(tg_path, "w") as fh:
        fh.write("# Error-analysis targets\n\n")
        fh.write(f"- scenes analyzed: **{len(files)}**\n")
        fh.write(f"- foreground classes: **{len(fg_idx)}**\n")
        fh.write(f"- **fg-mIoU self-check: {fg_miou:.4f}** (should match the eval-reported "
                 "val miou; if not, the scoring convention is off)\n\n")

        fh.write("## Most worth looking at (fg classes by lost GT points = cost)\n\n")
        fh.write("| rank | class | gt | pred | IoU | recall | prec | cost |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")
        for r, i in enumerate(targets, 1):
            fh.write(f"| {r} | {class_names[i]} | {int(gt_count[i])} | {int(pred_count[i])} | "
                     f"{iou[i]:.4f} | {recall[i]:.4f} | {precision[i]:.4f} | {int(cost[i])} |\n")
        fh.write("\n")

        fh.write("## Dead classes (have GT but ~never predicted)\n\n")
        if not dead:
            fh.write("- none\n\n")
        else:
            fh.write("| class | gt | pred | IoU |\n|---|---|---|---|\n")
            for i in dead:
                fh.write(f"| {class_names[i]} | {int(gt_count[i])} | {int(pred_count[i])} | {iou[i]:.4f} |\n")
            fh.write("\n")

        fh.write("## Worst scenes per target (for visualization)\n\n")
        scene_arr = np.asarray(scene_names)
        for i in targets:
            gt_s = per_scene_row[:, i]
            tp_s = per_scene_tp[:, i]
            col_s = per_scene_col[:, i]
            union_s = tp_s + (col_s - tp_s) + (gt_s - tp_s)
            with np.errstate(divide="ignore", invalid="ignore"):
                iou_s = np.where(union_s > 0, tp_s / union_s, np.nan)
            cand = np.where(gt_s >= args.min_scene_gt)[0]
            if cand.size == 0:
                cand = np.where(gt_s > 0)[0]
            if cand.size == 0:
                continue
            cand = cand[np.argsort(iou_s[cand])]  # ascending IoU = worst first
            worst = cand[: args.worst_scenes]
            fh.write(f"### {class_names[i]} (idx={i})\n")
            for s in worst:
                sn = scene_arr[s]
                fh.write(f"- `{sn}` IoU={iou_s[s]:.3f} gt={int(gt_s[s])} pred={int(col_s[s])}\n")
                fh.write(f"    - `python scripts/visualize_eval_query.py "
                         f"--scene-dir {args.scene_root}/{sn} "
                         f"--pred-file {args.dump_dir}/{sn}.npz --default-class \"{class_names[i]}\"`\n")
            fh.write("\n")

    print("=" * 64)
    print(f"Scenes analyzed : {len(files)}")
    print(f"Foreground cls  : {len(fg_idx)}")
    print(f"fg-mIoU (self-check): {fg_miou:.4f}")
    print("-" * 64)
    print(f"per-class report : {csv_path}")
    print(f"FN destinations  : {fn_path}")
    print(f"targets          : {tg_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()
