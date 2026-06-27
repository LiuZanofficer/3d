"""Task 0 - real 200-class label-space oracle for S3 flipped instances.

This is a ceiling test, not a method. It keeps the original 200 ScanNet200
classes and only oracle-corrects GT instances that S3 would call "wholesale
flipped": for a victim class C with top-1 absorber J, if an instance of C has
frac_J >= flip_thr, all points of that GT instance are changed back to C.

PRE-REGISTERED CRITERIA (locked before running):
  - oracle_fgmiou <= 0.18: this line has a low method ceiling; warn.
  - oracle_fgmiou >= 0.21: there is physical room to target 20+ fg-mIoU.
  - otherwise: continue, but treat expected gain as modest.

Self-check:
  The script must reproduce baseline fg-mIoU ~= 0.1548 using pred_semantic,
  background-as-ignore, and fg_class_idx. If it does not, fix scoring first.
"""

import argparse
from pathlib import Path

import numpy as np


def _decode_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def load_dump(path: Path):
    data = np.load(path, allow_pickle=True)
    return {key: _decode_scalar(data[key]) for key in data.files}


def scene_confmat(gt, pred, num_classes):
    flat = gt.astype(np.int64) * num_classes + pred.astype(np.int64)
    return np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def valid_scored_gt_pred(dump, num_classes, bg_idx, ignore_label, pred_override=None):
    gt = np.asarray(dump["gt_segment"]).astype(np.int64)
    pred = np.asarray(dump["pred_semantic"] if pred_override is None else pred_override).astype(np.int64)
    gt_fg = gt.copy()
    if bg_idx.size:
        gt_fg[np.isin(gt_fg, bg_idx)] = ignore_label
    valid = (
        (gt_fg != ignore_label)
        & (gt_fg >= 0)
        & (gt_fg < num_classes)
        & (pred >= 0)
        & (pred < num_classes)
    )
    return gt_fg[valid], pred[valid], valid


def build_confmat(files, num_classes, bg_idx, ignore_label):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for path in files:
        dump = load_dump(path)
        gt, pred, _ = valid_scored_gt_pred(dump, num_classes, bg_idx, ignore_label)
        cm += scene_confmat(gt, pred, num_classes)
    return cm


def fg_miou_from_cm(cm, fg_idx):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    union = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, 0.0)
    return float(np.mean(iou[fg_idx])), iou


def top_absorbers(cm, fg_idx):
    gt_count = cm.sum(axis=1).astype(np.float64)
    tp = np.diag(cm).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(gt_count > 0, tp / gt_count, 0.0)
    cost = gt_count * (1.0 - recall)
    absorber = {}
    for c in fg_idx:
        row = cm[c].astype(np.float64).copy()
        row[c] = 0
        if row.sum() > 0:
            absorber[int(c)] = int(np.argmax(row))
    ranked = sorted(absorber, key=lambda c: cost[c], reverse=True)
    return absorber, ranked, cost


def oracle_confmat(files, num_classes, bg_idx, ignore_label, pairs, flip_thr):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    corrected_points = {c: 0 for c in pairs}
    corrected_instances = {c: 0 for c in pairs}
    for path in files:
        dump = load_dump(path)
        gt_raw = np.asarray(dump["gt_segment"]).astype(np.int64)
        inst = np.asarray(dump["gt_instance"]).astype(np.int64)
        pred = np.asarray(dump["pred_semantic"]).astype(np.int64).copy()

        for c, j in pairs.items():
            mask_c = gt_raw == c
            if not mask_c.any():
                continue
            inst_c = inst[mask_c]
            pred_c = pred[mask_c]
            raw_idx_c = np.flatnonzero(mask_c)
            for iid in np.unique(inst_c):
                if iid < 0:
                    continue
                local = inst_c == iid
                n = int(local.sum())
                if n == 0:
                    continue
                frac_j = float(np.mean(pred_c[local] == j))
                if frac_j >= flip_thr:
                    raw_idx = raw_idx_c[local]
                    pred[raw_idx] = c
                    corrected_points[c] += n
                    corrected_instances[c] += 1

        gt, pred_valid, _ = valid_scored_gt_pred(dump, num_classes, bg_idx, ignore_label, pred)
        cm += scene_confmat(gt, pred_valid, num_classes)
    return cm, corrected_points, corrected_instances


def verdict(oracle_miou):
    if oracle_miou <= 0.18:
        return "LOW_CEILING (<=0.18): warn, this line alone is unlikely to be enough"
    if oracle_miou >= 0.21:
        return "ROOM_FOR_20_PLUS (>=0.21): physical room exists"
    return "MODEST_ROOM (0.18,0.21): continue, but lower expectations"


def main():
    ap = argparse.ArgumentParser(description="Task 0: real-labelspace oracle ceiling.")
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--topk", type=str, default="25,50,all")
    ap.add_argument("--flip-thr", type=float, default=0.5)
    ap.add_argument("--baseline-ref", type=float, default=0.1548)
    args = ap.parse_args()

    files = sorted(args.dump_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz under {args.dump_dir}")
    first = load_dump(files[0])
    if "pred_semantic" not in first or "gt_instance" not in first:
        raise SystemExit("Dump missing pred_semantic / gt_instance.")

    class_names = [str(c) for c in np.asarray(first["class_names"])]
    num_classes = len(class_names)
    fg_idx = np.asarray(first["fg_class_idx"]).reshape(-1).astype(int)
    bg_idx = np.asarray(first.get("bg_class_idx", np.array([]))).reshape(-1).astype(int)
    ignore_label = int(first.get("ignore_label", -100))

    base_cm = build_confmat(files, num_classes, bg_idx, ignore_label)
    base_miou, base_iou = fg_miou_from_cm(base_cm, fg_idx)
    absorber, ranked, cost = top_absorbers(base_cm, fg_idx)
    self_check = "OK" if abs(base_miou - args.baseline_ref) < 0.005 else "MISMATCH"

    topk_specs = []
    for token in args.topk.split(","):
        token = token.strip().lower()
        if token == "all":
            topk_specs.append(("all", len(ranked)))
        else:
            topk_specs.append((token, int(token)))

    results = []
    per_class_rows = {}
    for label, k in topk_specs:
        selected = ranked[:k]
        pairs = {c: absorber[c] for c in selected}
        cm_oracle, corr_pts, corr_inst = oracle_confmat(
            files, num_classes, bg_idx, ignore_label, pairs, args.flip_thr
        )
        miou, iou = fg_miou_from_cm(cm_oracle, fg_idx)
        results.append((label, len(selected), miou, miou - base_miou, verdict(miou)))
        rows = []
        for c in selected:
            rows.append(
                (
                    class_names[c],
                    class_names[absorber[c]],
                    int(cost[c]),
                    int(corr_inst[c]),
                    int(corr_pts[c]),
                    float(base_iou[c]),
                    float(iou[c]),
                    float(iou[c] - base_iou[c]),
                )
            )
        per_class_rows[label] = rows

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "oracle_real_labelspace.md"
    with open(out, "w") as fh:
        fh.write("# Task 0 real 200-class oracle ceiling\n\n")
        fh.write("## Self-check\n")
        fh.write(f"- baseline fg-mIoU = **{base_miou:.4f}** (ref {args.baseline_ref}; {self_check})\n")
        fh.write(f"- flip threshold = {args.flip_thr}\n")
        fh.write("- scoring: pred_semantic, background->ignore, invalid labels dropped, fg_class_idx mean\n\n")
        fh.write("## PRE-REGISTERED criteria\n")
        fh.write("- <=0.18: low ceiling / warn\n")
        fh.write("- >=0.21: physical room for 20+ fg-mIoU\n")
        fh.write("- otherwise: modest room\n\n")
        fh.write("## Oracle results\n\n")
        fh.write("| top-k victim classes | n_classes | oracle fg-mIoU | gain over baseline | verdict |\n")
        fh.write("|---|---:|---:|---:|---|\n")
        for label, n, miou, gain, v in results:
            fh.write(f"| {label} | {n} | {miou:.4f} | {gain:+.4f} | {v} |\n")
        for label, rows in per_class_rows.items():
            fh.write(f"\n## Per-class contribution ({label})\n\n")
            fh.write("| C | absorber J | cost | corrected_inst | corrected_points | base_iou | oracle_iou | delta_iou |\n")
            fh.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
            for c, j, co, ni, np_, bi, oi, di in rows:
                fh.write(f"| {c} | {j} | {co} | {ni} | {np_} | {bi:.4f} | {oi:.4f} | {di:+.4f} |\n")

    print("=" * 64)
    print(f"baseline fg-mIoU={base_miou:.4f} self-check={self_check}")
    for label, n, miou, gain, v in results:
        print(f"[{label}] n={n} oracle={miou:.4f} gain={gain:+.4f} -> {v}")
    print(f"outputs -> {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
