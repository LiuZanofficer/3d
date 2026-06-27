"""Task 1B - instance visual feature separability probe.

Uses Task 1A per-instance pooled features to test whether wholesale-flipped
instances are separable from correct instances in the visual feature space.

PRE-REGISTERED CRITERIA:
  - AUC >= 0.80: separable; decision/readout/aggregation/calibration issue.
  - AUC <= 0.60: not separable; representation collapse / feature insufficiency.
  - 0.60 < AUC < 0.80: inconclusive.

Controls:
  - random-label AUC should be near 0.5.
  - baseline fg-mIoU from the matching scene dump must reproduce ~=0.1548.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _decode_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    return {key: _decode_scalar(data[key]) for key in data.files}


def scene_confmat(gt, pred, num_classes):
    flat = gt.astype(np.int64) * num_classes + pred.astype(np.int64)
    return np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def build_confmat(files, num_classes, bg_idx, ignore_label):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for path in files:
        d = load_npz(path)
        gt = np.asarray(d["gt_segment"]).astype(np.int64)
        pred = np.asarray(d["pred_semantic"]).astype(np.int64)
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
        cm += scene_confmat(gt_fg[valid], pred[valid], num_classes)
    return cm


def fg_miou(cm, fg_idx):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    union = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, 0.0)
    return float(np.mean(iou[fg_idx]))


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


def collect_instances(feature_dir: Path):
    xs = []
    cls = []
    npts = []
    pred_hist = []
    scene = []
    inst_id = []
    class_names = None
    fg_idx = None
    bg_idx = None
    for path in sorted(feature_dir.glob("*.npz")):
        d = load_npz(path)
        if class_names is None:
            class_names = [str(c) for c in np.asarray(d["class_names"])]
            fg_idx = np.asarray(d["fg_class_idx"]).reshape(-1).astype(int)
            bg_idx = np.asarray(d.get("bg_class_idx", np.array([]))).reshape(-1).astype(int)
        xs.append(np.asarray(d["pooled_feat"]).astype(np.float32))
        cls.append(np.asarray(d["true_class"]).astype(np.int64))
        npts.append(np.asarray(d["n_points"]).astype(np.int64))
        pred_hist.append(np.asarray(d["pred_hist"]).astype(np.int64))
        scene.extend([str(d["scene_name"])] * len(d["true_class"]))
        inst_id.append(np.asarray(d["gt_instance_id"]).astype(np.int64))
    return dict(
        x=np.concatenate(xs, axis=0),
        true_class=np.concatenate(cls, axis=0),
        n_points=np.concatenate(npts, axis=0),
        pred_hist=np.concatenate(pred_hist, axis=0),
        scene=np.asarray(scene),
        gt_instance_id=np.concatenate(inst_id, axis=0),
        class_names=class_names,
        fg_idx=fg_idx,
        bg_idx=bg_idx,
    )


def safe_auc_probe(x, y, seed, max_folds=5):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) != 2:
        return np.nan, np.nan, 0
    counts = np.bincount(y)
    min_count = int(counts.min())
    if min_count < 3:
        return np.nan, np.nan, min_count
    n_splits = min(max_folds, min_count)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_score = np.zeros(len(y), dtype=np.float64)
    for train_idx, test_idx in cv.split(x, y):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        )
        clf.fit(x[train_idx], y[train_idx])
        y_score[test_idx] = clf.predict_proba(x[test_idx])[:, 1]
    auc = float(roc_auc_score(y, y_score))

    rng = np.random.default_rng(seed)
    y_rand = rng.permutation(y)
    y_rand_score = np.zeros(len(y_rand), dtype=np.float64)
    for train_idx, test_idx in cv.split(x, y_rand):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear"),
        )
        clf.fit(x[train_idx], y_rand[train_idx])
        y_rand_score[test_idx] = clf.predict_proba(x[test_idx])[:, 1]
    auc_rand = float(roc_auc_score(y_rand, y_rand_score))
    return auc, auc_rand, min_count


def verdict(auc):
    if np.isnan(auc):
        return "LOW_SAMPLE"
    if auc >= 0.80:
        return "SEPARABLE (decision/readout/aggregation/calibration issue)"
    if auc <= 0.60:
        return "NOT_SEPARABLE (representation/feature issue)"
    return "INCONCLUSIVE"


def main():
    ap = argparse.ArgumentParser(description="Task 1B: instance feature separability probe.")
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--topk-cost", type=int, default=25)
    ap.add_argument("--flip-thr", type=float, default=0.5)
    ap.add_argument("--correct-thr", type=float, default=0.5)
    ap.add_argument("--baseline-ref", type=float, default=0.1548)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scene_files = sorted(args.dump_dir.glob("*.npz"))
    if not scene_files:
        raise FileNotFoundError(f"No scene dumps under {args.dump_dir}")
    first = load_npz(scene_files[0])
    num_classes = len(np.asarray(first["class_names"]))
    fg_idx = np.asarray(first["fg_class_idx"]).reshape(-1).astype(int)
    bg_idx = np.asarray(first.get("bg_class_idx", np.array([]))).reshape(-1).astype(int)
    ignore_label = int(first.get("ignore_label", -100))
    cm = build_confmat(scene_files, num_classes, bg_idx, ignore_label)
    baseline = fg_miou(cm, fg_idx)
    self_check = "OK" if abs(baseline - args.baseline_ref) < 0.005 else "MISMATCH"
    absorber, ranked, cost = top_absorbers(cm, fg_idx)
    topk = ranked[: args.topk_cost]

    data = collect_instances(args.feature_dir)
    x = data["x"]
    true_class = data["true_class"]
    n_points = data["n_points"]
    pred_hist = data["pred_hist"]
    class_names = data["class_names"]

    rows = []
    aucs_flip = []
    aucs_sibling = []
    aucs_rand = []
    for c in topk:
        j = absorber[c]
        idx_c = np.flatnonzero(true_class == c)
        if idx_c.size == 0:
            continue
        frac_j = pred_hist[idx_c, j] / np.maximum(n_points[idx_c], 1)
        frac_c = pred_hist[idx_c, c] / np.maximum(n_points[idx_c], 1)
        flip_idx = idx_c[frac_j >= args.flip_thr]
        correct_idx = idx_c[frac_c >= args.correct_thr]

        idx_probe = np.concatenate([correct_idx, flip_idx])
        y_probe = np.concatenate([np.zeros(len(correct_idx)), np.ones(len(flip_idx))])
        auc_flip, auc_rand, min_count = safe_auc_probe(x[idx_probe], y_probe, args.seed)

        idx_j = np.flatnonzero(true_class == j)
        idx_sib = np.concatenate([idx_c, idx_j])
        y_sib = np.concatenate([np.zeros(len(idx_c)), np.ones(len(idx_j))])
        auc_sib, auc_sib_rand, min_count_sib = safe_auc_probe(x[idx_sib], y_sib, args.seed + 17)

        if not np.isnan(auc_flip):
            aucs_flip.append(auc_flip)
            aucs_rand.append(auc_rand)
        if not np.isnan(auc_sib):
            aucs_sibling.append(auc_sib)

        rows.append(
            dict(
                c=class_names[c],
                j=class_names[j],
                cost=int(cost[c]),
                n_correct=len(correct_idx),
                n_flipped=len(flip_idx),
                n_true_c=len(idx_c),
                n_true_j=len(idx_j),
                auc_flip=auc_flip,
                auc_rand=auc_rand,
                min_count=min_count,
                auc_sibling=auc_sib,
                auc_sibling_rand=auc_sib_rand,
                min_count_sibling=min_count_sib,
                verdict=verdict(auc_flip),
            )
        )

    mean_auc = float(np.nanmean(aucs_flip)) if aucs_flip else np.nan
    median_auc = float(np.nanmedian(aucs_flip)) if aucs_flip else np.nan
    mean_rand = float(np.nanmean(aucs_rand)) if aucs_rand else np.nan
    mean_sib = float(np.nanmean(aucs_sibling)) if aucs_sibling else np.nan
    headline_verdict = verdict(mean_auc)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "instance_separability.md"
    with open(out, "w") as fh:
        fh.write("# Task 1B instance visual feature separability\n\n")
        fh.write("## Self-check\n")
        fh.write(f"- baseline fg-mIoU = **{baseline:.4f}** (ref {args.baseline_ref}; {self_check})\n")
        fh.write(f"- top-k cost pairs = {args.topk_cost}; flip_thr={args.flip_thr}; correct_thr={args.correct_thr}\n")
        fh.write(f"- instances loaded = {len(x)} from `{args.feature_dir}`\n\n")
        fh.write("## PRE-REGISTERED criteria\n")
        fh.write("- AUC>=0.80: separable (decision/readout/aggregation/calibration)\n")
        fh.write("- AUC<=0.60: not separable (representation/feature issue)\n")
        fh.write("- otherwise: inconclusive\n\n")
        fh.write("## Summary\n")
        fh.write(f"- mean AUC flipped-vs-correct = **{mean_auc:.3f}**; median={median_auc:.3f}\n")
        fh.write(f"- random-label control mean AUC = **{mean_rand:.3f}**\n")
        fh.write(f"- mean true-C-vs-true-J sibling AUC = **{mean_sib:.3f}**\n")
        fh.write(f"- **Verdict: {headline_verdict}**\n\n")
        fh.write("## Per-pair probes\n\n")
        fh.write(
            "| C | absorber J | cost | n_correct | n_flipped | AUC flip-vs-correct | "
            "random AUC | AUC true C-vs-J | verdict |\n"
        )
        fh.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            fh.write(
                f"| {r['c']} | {r['j']} | {r['cost']} | {r['n_correct']} | {r['n_flipped']} | "
                f"{r['auc_flip']:.3f} | {r['auc_rand']:.3f} | {r['auc_sibling']:.3f} | {r['verdict']} |\n"
            )

    print("=" * 64)
    print(f"baseline fg-mIoU={baseline:.4f} self-check={self_check}")
    print(f"mean flip-vs-correct AUC={mean_auc:.3f} random={mean_rand:.3f} sibling={mean_sib:.3f}")
    print(f"PRE-REGISTERED VERDICT: {headline_verdict}")
    print(f"outputs -> {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
