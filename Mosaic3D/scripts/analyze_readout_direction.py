"""Step 0 - compare visual discriminant direction with text readout direction.

For each high-cost sibling pair (C -> J):
  d_vis = LDA direction separating true-C instances and true-J instances.
  d_txt = normalize(text_emb[C] - text_emb[J]).
Report |cos(d_vis, d_txt)| and a random-direction control.

PRE-REGISTERED:
  If a majority of valid pairs have |cos| < 0.5 and the mean is not above the
  random control by >=0.05, confirm "text readout direction != visual decision
  direction"; otherwise inconclusive/refuted.

GT instance/class labels are used only for analysis, not for any method.
"""

import argparse
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


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


def fg_miou(cm, fg_idx):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    union = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, 0.0)
    return float(np.mean(iou[fg_idx]))


def collect_features(feature_dir: Path):
    xs, ys = [], []
    class_names = None
    for path in sorted(feature_dir.glob("*.npz")):
        d = load_npz(path)
        if class_names is None:
            class_names = [str(c) for c in np.asarray(d["class_names"])]
        xs.append(np.asarray(d["pooled_feat"], dtype=np.float64))
        ys.append(np.asarray(d["true_class"], dtype=np.int64))
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0), class_names


def unit(x):
    n = np.linalg.norm(x)
    if n <= 1e-12:
        return x
    return x / n


def main():
    ap = argparse.ArgumentParser(description="Step 0: visual LDA direction vs text readout direction.")
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--text-embeddings", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--topk-cost", type=int, default=25)
    ap.add_argument("--min-instances", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline-ref", type=float, default=0.1548)
    args = ap.parse_args()

    files = sorted(args.dump_dir.glob("*.npz"))
    first = load_npz(files[0])
    num_classes = len(np.asarray(first["class_names"]))
    fg_idx = np.asarray(first["fg_class_idx"]).reshape(-1).astype(int)
    bg_idx = np.asarray(first.get("bg_class_idx", np.array([]))).reshape(-1).astype(int)
    ignore_label = int(first.get("ignore_label", -100))
    cm = build_confmat(files, num_classes, bg_idx, ignore_label)
    baseline = fg_miou(cm, fg_idx)
    self_check = "OK" if abs(baseline - args.baseline_ref) < 0.005 else "MISMATCH"
    absorber, ranked, cost = top_absorbers(cm, fg_idx)

    x, y, class_names = collect_features(args.feature_dir)
    text = np.load(args.text_embeddings, allow_pickle=True)["emb"].astype(np.float64)
    text = text / np.maximum(np.linalg.norm(text, axis=1, keepdims=True), 1e-12)

    rng = np.random.default_rng(args.seed)
    rows = []
    cos_vals = []
    rand_vals = []
    for c in ranked[: args.topk_cost]:
        j = absorber[c]
        idx_c = np.flatnonzero(y == c)
        idx_j = np.flatnonzero(y == j)
        if len(idx_c) < args.min_instances or len(idx_j) < args.min_instances:
            rows.append((class_names[c], class_names[j], int(cost[c]), len(idx_c), len(idx_j), np.nan, np.nan, "LOW_SAMPLE"))
            continue
        xx = np.concatenate([x[idx_c], x[idx_j]], axis=0)
        yy = np.concatenate([np.zeros(len(idx_c), dtype=int), np.ones(len(idx_j), dtype=int)])
        lda = LinearDiscriminantAnalysis(solver="svd")
        lda.fit(xx, yy)
        d_vis = unit(lda.coef_[0].astype(np.float64))
        d_txt = unit(text[c] - text[j])
        cos = float(abs(np.dot(d_vis, d_txt)))
        rand_dir = unit(rng.normal(size=d_txt.shape))
        rand_cos = float(abs(np.dot(rand_dir, d_txt)))
        cos_vals.append(cos)
        rand_vals.append(rand_cos)
        rows.append((class_names[c], class_names[j], int(cost[c]), len(idx_c), len(idx_j), cos, rand_cos, "OK"))

    mean_cos = float(np.nanmean(cos_vals)) if cos_vals else np.nan
    median_cos = float(np.nanmedian(cos_vals)) if cos_vals else np.nan
    frac_low = float(np.mean(np.asarray(cos_vals) < 0.5)) if cos_vals else np.nan
    mean_rand = float(np.nanmean(rand_vals)) if rand_vals else np.nan

    if frac_low > 0.5 and (mean_cos - mean_rand) < 0.05:
        verdict = "CONFIRM (text readout direction differs from visual discriminant)"
    elif mean_cos >= 0.5:
        verdict = "REFUTE (text direction broadly aligns with visual discriminant)"
    else:
        verdict = "INCONCLUSIVE"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "readout_direction.md"
    with open(out, "w") as fh:
        fh.write("# Step 0 readout direction: visual LDA vs text anchor\n\n")
        fh.write("## Self-check\n")
        fh.write(f"- baseline fg-mIoU = **{baseline:.4f}** (ref {args.baseline_ref}; {self_check})\n")
        fh.write(f"- valid pairs = {len(cos_vals)} / top-{args.topk_cost}\n")
        fh.write("- GT labels are used for this analysis only, not for any method.\n\n")
        fh.write("## PRE-REGISTERED criteria\n")
        fh.write("- CONFIRM if majority pairs have |cos|<0.5 and mean is not > random by 0.05\n")
        fh.write("- REFUTE if mean |cos|>=0.5\n")
        fh.write("- otherwise INCONCLUSIVE\n\n")
        fh.write("## Summary\n")
        fh.write(f"- mean |cos(d_vis,d_txt)| = **{mean_cos:.3f}**; median={median_cos:.3f}\n")
        fh.write(f"- fraction <0.5 = **{frac_low:.3f}**\n")
        fh.write(f"- random direction mean |cos| = **{mean_rand:.3f}**\n")
        fh.write(f"- **Verdict: {verdict}**\n\n")
        fh.write("## Per-pair\n\n")
        fh.write("| C | J | cost | n_C | n_J | abs_cos | random_abs_cos | status |\n")
        fh.write("|---|---|---:|---:|---:|---:|---:|---|\n")
        for c_name, j_name, co, nc, nj, cos, rand_cos, status in rows:
            cos_s = f"{cos:.3f}" if not np.isnan(cos) else "nan"
            rand_s = f"{rand_cos:.3f}" if not np.isnan(rand_cos) else "nan"
            fh.write(f"| {c_name} | {j_name} | {co} | {nc} | {nj} | {cos_s} | {rand_s} | {status} |\n")

    print("=" * 64)
    print(f"baseline={baseline:.4f} self-check={self_check}")
    print(f"mean_cos={mean_cos:.3f} median={median_cos:.3f} frac_low={frac_low:.3f} random={mean_rand:.3f}")
    print(f"PRE-REGISTERED VERDICT: {verdict}")
    print(f"outputs -> {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
