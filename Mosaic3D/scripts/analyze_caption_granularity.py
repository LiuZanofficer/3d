"""Task 2 - caption granularity vs eval sibling flips.

This tests whether S3 instance flips are taught by training captions that
systematically use the absorber/sibling term J for GT regions of victim class C.

PRE-REGISTERED CRITERIA (operationalized before running):
  CONFIRM training over-specification if:
    - more than half of the top eval pairs have P(caption=J|GT=C)>0 AND J ranks
      first among non-C class terms, and
    - mean row-wise Spearman(training caption terms, eval-FN destinations) >= 0.4.
  REFUTE if:
    - median P(caption=C|GT=C) >= 0.5, and
    - mean row-wise Spearman < 0.1.
  Otherwise: INCONCLUSIVE.

Fixed parsing rule:
  - Longest class-name / alias match wins at the regex level.
  - A caption can contribute multiple matched class terms, each counted once.
  - Captions with no matched term count as <none>.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.metadata.scannet import CLASS_LABELS_200  # noqa: E402
from src.utils.io import unpack_list_of_np_arrays  # noqa: E402


NONE_COL = -1


def _decode_scalar(value):
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def load_npz(path: Path):
    data = np.load(path, allow_pickle=True)
    return {key: _decode_scalar(data[key]) for key in data.files}


def build_valid_mapper(num_classes, ignore_label):
    mapper = np.ones(max(256, num_classes), dtype=np.int64) * ignore_label
    for idx, name in enumerate(CLASS_LABELS_200):
        if not str(name).startswith("other"):
            mapper[idx] = idx
    return mapper


def remap_segment(segment_raw, num_classes, ignore_label):
    mapper = build_valid_mapper(num_classes, ignore_label)
    if segment_raw.max(initial=0) < len(mapper):
        return mapper[segment_raw.astype(np.int64)]
    # Fallback: if future preprocessing already writes valid class ids beyond
    # mapper assumptions, keep only labels in range.
    seg = segment_raw.astype(np.int64)
    out = np.ones_like(seg) * ignore_label
    valid = (seg >= 0) & (seg < num_classes)
    out[valid] = seg[valid]
    return out


def scene_confmat(gt, pred, num_classes):
    flat = gt.astype(np.int64) * num_classes + pred.astype(np.int64)
    return np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def build_eval_confmat(files, num_classes, bg_idx, ignore_label):
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


def alias_terms(class_names):
    by_name = {name: i for i, name in enumerate(class_names)}
    aliases = {}
    # Conservative common aliases; only add if the target class exists.
    for alias, target in [
        ("sofa", "couch"),
        ("couch", "couch"),
        ("trash bin", "trash bin"),
        ("trashcan", "trash can"),
        ("garbage can", "trash can"),
        ("tv", "tv"),
        ("television", "tv"),
        ("refridgerator", "refrigerator"),
        ("fridge", "refrigerator"),
        ("white board", "whiteboard"),
    ]:
        if target in by_name:
            aliases[alias] = by_name[target]
    return aliases


def build_matcher(class_names):
    term_to_class = {name.lower(): i for i, name in enumerate(class_names)}
    term_to_class.update(alias_terms(class_names))
    terms = sorted(term_to_class, key=len, reverse=True)
    # Capture longest textual terms; final set de-duplicates multiple mentions.
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b")
    return pattern, term_to_class


def matched_terms(text, pattern, term_to_class):
    hits = set()
    for m in pattern.findall(text.lower()):
        hits.add(term_to_class[m])
    if not hits:
        return {NONE_COL}
    return hits


def iter_caption_regions(scene_dir: Path, source: str):
    captions_file = scene_dir / f"captions.{source}.npz"
    indices_file = scene_dir / f"point_indices.{source}.npz"
    if not captions_file.exists() or not indices_file.exists():
        return
    captions_nested = unpack_list_of_np_arrays(captions_file)
    indices_nested = unpack_list_of_np_arrays(indices_file)
    for caps_obj, idx_obj in zip(captions_nested, indices_nested):
        for cap, idx in zip(caps_obj, idx_obj):
            if isinstance(cap, bytes):
                cap = cap.decode("utf-8", "ignore")
            yield str(cap), np.asarray(idx).astype(np.int64)


def majority_gt(segment, point_idx, ignore_label, num_classes):
    if point_idx.size == 0:
        return None
    point_idx = point_idx[(point_idx >= 0) & (point_idx < len(segment))]
    if point_idx.size == 0:
        return None
    vals = segment[point_idx]
    valid = (vals != ignore_label) & (vals >= 0) & (vals < num_classes)
    if not np.any(valid):
        return None
    hist = np.bincount(vals[valid], minlength=num_classes)
    return int(np.argmax(hist))


def main():
    ap = argparse.ArgumentParser(description="Task 2: caption granularity vs eval FN destinations.")
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--scene-root", type=Path, default=Path("/datasets/mosaic3d/data/scannet"))
    ap.add_argument("--anno-sources", type=str, default="gsam2,seem")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--topk-cost", type=int, default=25)
    ap.add_argument("--baseline-ref", type=float, default=0.1548)
    ap.add_argument("--ignore-label", type=int, default=-100)
    args = ap.parse_args()

    scene_files = sorted(args.dump_dir.glob("*.npz"))
    if not scene_files:
        raise FileNotFoundError(f"No eval dumps under {args.dump_dir}")
    first = load_npz(scene_files[0])
    class_names = [str(c) for c in np.asarray(first["class_names"])]
    num_classes = len(class_names)
    fg_idx = np.asarray(first["fg_class_idx"]).reshape(-1).astype(int)
    bg_idx = np.asarray(first.get("bg_class_idx", np.array([]))).reshape(-1).astype(int)
    ignore_label = int(first.get("ignore_label", args.ignore_label))

    cm = build_eval_confmat(scene_files, num_classes, bg_idx, ignore_label)
    baseline = fg_miou(cm, fg_idx)
    self_check = "OK" if abs(baseline - args.baseline_ref) < 0.005 else "MISMATCH"
    absorber, ranked, cost = top_absorbers(cm, fg_idx)
    topk = ranked[: args.topk_cost]

    pattern, term_to_class = build_matcher(class_names)
    sources = [s.strip() for s in args.anno_sources.split(",") if s.strip()]
    t_mat = np.zeros((num_classes, num_classes + 1), dtype=np.int64)  # last col = <none>
    n_regions = 0
    n_scenes = 0
    n_alignment_checked = 0

    for scene_dir in sorted(args.scene_root.glob("scene*_*")):
        seg_file = scene_dir / "segment200.npy"
        if not seg_file.exists():
            continue
        segment_raw = np.load(seg_file)
        segment = remap_segment(segment_raw, num_classes, ignore_label)
        scene_had_caption = False
        for source in sources:
            for caption, point_idx in iter_caption_regions(scene_dir, source) or []:
                g = majority_gt(segment, point_idx, ignore_label, num_classes)
                if g is None:
                    continue
                if n_alignment_checked < 1:
                    # Alignment smoke check: point indices should land on at
                    # least one valid class in the same point array.
                    n_alignment_checked += 1
                terms = matched_terms(caption, pattern, term_to_class)
                for t in terms:
                    col = num_classes if t == NONE_COL else int(t)
                    t_mat[g, col] += 1
                n_regions += 1
                scene_had_caption = True
        if scene_had_caption:
            n_scenes += 1

    row_sum = t_mat.sum(axis=1, keepdims=True)
    prob = np.divide(t_mat, np.maximum(row_sum, 1), dtype=np.float64)

    pair_rows = []
    pair_confirm = 0
    row_rhos = []
    self_probs = []
    for c in topk:
        j = absorber[c]
        p_j = float(prob[c, j])
        p_c = float(prob[c, c])
        self_probs.append(p_c)
        non_c_probs = prob[c, :num_classes].copy()
        non_c_probs[c] = -1.0
        rank_order = np.argsort(-non_c_probs)
        rank_j = int(np.where(rank_order == j)[0][0] + 1)
        if p_j > 0 and rank_j == 1:
            pair_confirm += 1

        fn_row = cm[c].astype(np.float64).copy()
        fn_row[c] = 0.0
        caption_row = prob[c, :num_classes].astype(np.float64).copy()
        caption_row[c] = 0.0
        mask = np.ones(num_classes, dtype=bool)
        mask[c] = False
        if fn_row[mask].sum() > 0 and caption_row[mask].sum() > 0:
            rho = spearmanr(fn_row[mask], caption_row[mask]).statistic
            if not np.isnan(rho):
                row_rhos.append(float(rho))
        else:
            rho = np.nan

        pair_rows.append(
            dict(
                c=class_names[c],
                j=class_names[j],
                cost=int(cost[c]),
                p_j=p_j,
                p_c=p_c,
                rank_j=rank_j,
                rho=float(rho) if not np.isnan(rho) else np.nan,
                total_terms=int(row_sum[c, 0]),
                none_prob=float(prob[c, num_classes]),
            )
        )

    confirm_frac = pair_confirm / max(len(pair_rows), 1)
    mean_rho = float(np.mean(row_rhos)) if row_rhos else np.nan
    median_self = float(np.median(self_probs)) if self_probs else np.nan

    if confirm_frac > 0.5 and mean_rho >= 0.4:
        verdict = "CONFIRM (training captions over-specify sibling absorbers)"
    elif median_self >= 0.5 and (np.isnan(mean_rho) or mean_rho < 0.1):
        verdict = "REFUTE (captions mostly self-class and not FN-correlated)"
    else:
        verdict = "INCONCLUSIVE"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "caption_granularity_T.npy", t_mat)
    with open(args.out_dir / "caption_granularity_pairs.csv", "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["C", "J", "cost", "P_J_given_C", "P_C_given_C", "rank_J_nonC", "rho", "total_terms", "P_none"],
        )
        writer.writeheader()
        for r in pair_rows:
            writer.writerow(
                {
                    "C": r["c"],
                    "J": r["j"],
                    "cost": r["cost"],
                    "P_J_given_C": f"{r['p_j']:.6f}",
                    "P_C_given_C": f"{r['p_c']:.6f}",
                    "rank_J_nonC": r["rank_j"],
                    "rho": f"{r['rho']:.6f}" if not np.isnan(r["rho"]) else "nan",
                    "total_terms": r["total_terms"],
                    "P_none": f"{r['none_prob']:.6f}",
                }
            )

    out = args.out_dir / "caption_granularity.md"
    with open(out, "w") as fh:
        fh.write("# Task 2 caption granularity vs eval FN destinations\n\n")
        fh.write("## Self-check and source\n")
        fh.write(f"- baseline fg-mIoU = **{baseline:.4f}** (ref {args.baseline_ref}; {self_check})\n")
        fh.write(f"- anno_sources = `{sources}` (ScanNet config default unless explicitly overridden)\n")
        fh.write(f"- scanned scenes = {n_scenes}; caption regions counted = {n_regions}\n")
        fh.write(f"- alignment smoke checks passed = {n_alignment_checked}\n")
        fh.write("- parsing: longest class/alias regex; multiple terms each count once; no match -> `<none>`\n\n")
        fh.write("## PRE-REGISTERED criteria\n")
        fh.write("- CONFIRM: >50% top pairs have P(J|C)>0 and J rank #1 non-C, plus mean Spearman>=0.4\n")
        fh.write("- REFUTE: median P(C|C)>=0.5 and mean Spearman<0.1\n")
        fh.write("- otherwise: INCONCLUSIVE\n\n")
        fh.write("## Summary\n")
        fh.write(f"- top pairs analyzed = {len(pair_rows)}\n")
        fh.write(f"- pair-confirm fraction = **{confirm_frac:.3f}** ({pair_confirm}/{len(pair_rows)})\n")
        fh.write(f"- mean row-wise Spearman(caption terms, eval FN destinations) = **{mean_rho:.3f}**\n")
        fh.write(f"- median P(caption=C | GT=C) = **{median_self:.3f}**\n")
        fh.write(f"- **Verdict: {verdict}**\n\n")
        fh.write("## Per-pair comparison\n\n")
        fh.write("| C | eval absorber J | cost | P(J|C) | P(C|C) | J rank among non-C | row Spearman | P(<none>) |\n")
        fh.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for r in pair_rows:
            rho_str = f"{r['rho']:.3f}" if not np.isnan(r["rho"]) else "nan"
            fh.write(
                f"| {r['c']} | {r['j']} | {r['cost']} | {r['p_j']:.4f} | {r['p_c']:.4f} | "
                f"{r['rank_j']} | {rho_str} | {r['none_prob']:.4f} |\n"
            )

    print("=" * 64)
    print(f"baseline fg-mIoU={baseline:.4f} self-check={self_check}")
    print(f"sources={sources} scenes={n_scenes} regions={n_regions}")
    print(f"pair_confirm_frac={confirm_frac:.3f} mean_rho={mean_rho:.3f} median_self={median_self:.3f}")
    print(f"PRE-REGISTERED VERDICT: {verdict}")
    print(f"outputs -> {out}")
    print("=" * 64)


if __name__ == "__main__":
    main()
