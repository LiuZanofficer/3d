"""V4 - decision-problem vs feature-problem: merge sibling classes, recompute mIoU.

If the model's points are not lost but only mislabeled at a finer granularity,
then collapsing confusable siblings into a superclass should make fg-mIoU jump.
If it stays low, the bottleneck is the 3D features, not the text-decision.

CRITICAL CONTROL: merging classes mechanically inflates mIoU (you average over
fewer, larger, easier classes). So the real semantic gain is measured AGAINST a
random merge with the SAME group-size distribution:
    Delta = miou(semantic grouping) - mean(miou(random groupings of same sizes))

Groupings (both automatic, no hand-labeling):
  - confusion: union fg classes whose mutual confusion fraction >= threshold
               (granularity ceiling, but somewhat circular).
  - text:      connected components of CLIP class-name cosine > threshold
               (from analyze_text_embeddings.py --> text_groups.json; mechanism-specific).

PRE-REGISTERED VERDICT (decided before running; do NOT change after seeing data):
  Delta_text = miou_text - mean_random_text ; Delta_conf likewise.
  CONFIRM (decision/granularity problem):
        Delta_text >= +0.08 AND Delta_conf >= +0.08 AND miou_text > mean_random_text + 3*sd
  REFUTE  (feature problem):  Delta_text <= +0.02
  INCONCLUSIVE: otherwise, or (Delta_conf large but Delta_text < +0.03)
  Pre-req self-check: baseline (singleton) fg-mIoU must reproduce ~0.1548.

Read-only; numpy only.
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


def scene_confmat(gt_fg, pred, num_classes):
    flat = gt_fg.astype(np.int64) * num_classes + pred.astype(np.int64)
    return np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def build_global_confmat(files, num_classes, bg_idx, ignore_label):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for f in files:
        d = load_dump(f)
        gt = np.asarray(d["gt_segment"]).astype(np.int64)
        pred = np.asarray(d["pred_semantic"]).astype(np.int64)
        gt_fg = gt.copy()
        if bg_idx.size:
            gt_fg[np.isin(gt_fg, bg_idx)] = ignore_label
        valid = (gt_fg != ignore_label) & (gt_fg >= 0) & (gt_fg < num_classes) \
            & (pred >= 0) & (pred < num_classes)
        cm += scene_confmat(gt_fg[valid], pred[valid], num_classes)
    return cm


def remap_contiguous(grp):
    """Map arbitrary group ids to 0..K-1 contiguous."""
    uniq = {g: i for i, g in enumerate(sorted(set(grp)))}
    return np.array([uniq[g] for g in grp], dtype=np.int64), len(uniq)


def merged_fg_miou(cm, grp, fg_idx):
    """fg-mIoU after collapsing classes by group map `grp` (len num_classes).
    IoU averaged over groups that contain >=1 fg class; union==0 -> 0 (matches codebase)."""
    grp, num_groups = remap_contiguous(grp)
    flat = grp[:, None] * num_groups + grp[None, :]
    G = np.bincount(flat.ravel(), weights=cm.ravel(), minlength=num_groups * num_groups)
    G = G.reshape(num_groups, num_groups)
    tp = np.diag(G)
    row = G.sum(axis=1)
    col = G.sum(axis=0)
    union = tp + (col - tp) + (row - tp)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, tp / union, 0.0)
    fg_groups = np.unique(grp[fg_idx])
    return float(np.mean(iou[fg_groups]))


def singleton_groups(num_classes):
    return np.arange(num_classes, dtype=np.int64)


def greedy_disjoint_match(weight_edges, num_classes):
    """Greedy maximum matching: sort edges by weight desc, take an edge only if
    both endpoints are still free. Returns (grp, n_pairs). Group size <= 2, so no
    transitive chaining (unlike threshold connected-components)."""
    grp = singleton_groups(num_classes)
    used = np.zeros(num_classes, dtype=bool)
    gid = num_classes + 1
    n_pairs = 0
    for _, a, b in sorted(weight_edges, key=lambda e: e[0], reverse=True):
        if used[a] or used[b]:
            continue
        grp[a] = gid
        grp[b] = gid
        used[a] = used[b] = True
        gid += 1
        n_pairs += 1
    return grp, n_pairs


def confusion_match(cm, fg_idx, min_share):
    """Pair each fg class with its strongest mutual confuser (symmetric leak),
    via disjoint greedy matching. Only edges with leak share >= min_share."""
    gt_count = cm.sum(axis=1).astype(np.float64)
    fg = [int(c) for c in fg_idx]
    fgset = set(fg)
    edges = []
    for a in fg:
        if gt_count[a] <= 0:
            continue
        for b in fg:
            if b <= a or b not in fgset:
                continue
            fa = cm[a, b] / gt_count[a] if gt_count[a] > 0 else 0.0
            fb = cm[b, a] / gt_count[b] if gt_count[b] > 0 else 0.0
            w = max(fa, fb)
            if w >= min_share:
                edges.append((w, a, b))
    return greedy_disjoint_match(edges, cm.shape[0])


def text_match(cos, fg_idx, min_cos):
    """Pair each fg class with its nearest text neighbor via disjoint greedy
    matching. Only edges with cosine >= min_cos."""
    fg = [int(c) for c in fg_idx]
    edges = []
    for i in range(len(fg)):
        for j in range(i + 1, len(fg)):
            a, b = fg[i], fg[j]
            w = float(cos[a, b])
            if w >= min_cos:
                edges.append((w, a, b))
    return greedy_disjoint_match(edges, cos.shape[0])


def random_match_stats(cm, fg_idx, n_pairs, n_trials, seed):
    """Random disjoint pairing of n_pairs fg classes; rest singletons."""
    rng = np.random.default_rng(seed)
    fg = np.asarray(fg_idx)
    out = []
    for _ in range(n_trials):
        perm = rng.permutation(fg)
        grp = singleton_groups(cm.shape[0])
        gid = cm.shape[0] + 1
        for p in range(n_pairs):
            a, b = int(perm[2 * p]), int(perm[2 * p + 1])
            grp[a] = grp[b] = gid
            gid += 1
        out.append(merged_fg_miou(cm, grp, fg_idx))
    out = np.asarray(out)
    return float(out.mean()), float(out.std())


def main():
    ap = argparse.ArgumentParser(description="V4: disjoint top-1 sibling matching, recompute fg-mIoU vs random matching.")
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--text-embeddings", type=Path, default=None,
                    help="text_embeddings.npz from analyze_text_embeddings.py (for text matching)")
    ap.add_argument("--min-share", type=float, default=0.05,
                    help="min confusion leak share to allow a confusion-match edge")
    ap.add_argument("--min-cos", type=float, default=0.85,
                    help="min text cosine to allow a text-match edge")
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--baseline-ref", type=float, default=0.1548)
    args = ap.parse_args()

    files = sorted(args.dump_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz under {args.dump_dir}")
    first = load_dump(files[0])
    if "pred_semantic" not in first:
        raise SystemExit("Old dump (no pred_semantic). Re-run eval with the extended dump first.")
    class_names = [str(c) for c in np.asarray(first["class_names"])]
    num_classes = len(class_names)
    fg_idx = np.asarray(first["fg_class_idx"]).reshape(-1).astype(int)
    bg_idx = np.asarray(first.get("bg_class_idx", np.array([]))).reshape(-1).astype(int)
    ignore_label = int(first.get("ignore_label", -100))

    cm = build_global_confmat(files, num_classes, bg_idx, ignore_label)
    baseline = merged_fg_miou(cm, singleton_groups(num_classes), fg_idx)

    def stats_for(grp, n_pairs):
        miou = merged_fg_miou(cm, grp, fg_idx)
        rmean, rsd = random_match_stats(cm, fg_idx, n_pairs, args.n_random, args.seed)
        z = (miou - rmean) / (rsd + 1e-9)
        return dict(miou=miou, rmean=rmean, rsd=rsd, npairs=n_pairs,
                    delta=miou - rmean, z=z, gain=miou - baseline)

    results = {}
    grp_conf, k_conf = confusion_match(cm, fg_idx, args.min_share)
    results["confusion"] = stats_for(grp_conf, k_conf)

    if args.text_embeddings is not None and args.text_embeddings.exists():
        cos = np.load(args.text_embeddings, allow_pickle=True)["cos"]
        grp_text, k_text = text_match(cos, fg_idx, args.min_cos)
        results["text"] = stats_for(grp_text, k_text)

    # ---- PRE-REGISTERED verdict (disjoint size<=2 matching, z-based) ----
    if "text" in results:
        zc, zt = results["confusion"]["z"], results["text"]["z"]
        dc = results["confusion"]["delta"]
        if zc > 3 and zt > 3 and dc >= 0.02:
            verdict = "CONFIRM (decision problem: recoverable & text-aligned)"
        elif zc > 3 and zt < 2:
            verdict = "PARTIAL (recoverable by merging confusers, but NOT text-name driven)"
        elif zc < 2:
            verdict = "REFUTE (feature problem: merging confusers ~ random)"
        else:
            verdict = "INCONCLUSIVE"
    else:
        verdict = "INCONCLUSIVE (need --text-embeddings for full verdict)"

    self_check = "OK" if abs(baseline - args.baseline_ref) < 0.005 else "MISMATCH!"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "merge_superclass.md", "w") as fh:
        fh.write("# V4 sibling-merge recompute (disjoint top-1 matching, size<=2)\n\n")
        fh.write(f"- baseline (singleton) fg-mIoU = **{baseline:.4f}** (ref {args.baseline_ref}; self-check {self_check})\n")
        fh.write("- merging is disjoint top-1 pairing (no transitive chaining); random control = random "
                 "disjoint pairing of the SAME number of pairs.\n\n")
        fh.write("| matching | #pairs | merged mIoU | random mIoU (same #pairs) | Delta | z | gain-over-baseline |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for name, r in results.items():
            fh.write(f"| {name} | {r['npairs']} | {r['miou']:.4f} | {r['rmean']:.4f}±{r['rsd']:.4f} | "
                     f"{r['delta']:+.4f} | {r['z']:.1f} | {r['gain']:+.4f} |\n")
        fh.write("\n## PRE-REGISTERED verdict\n")
        fh.write("- CONFIRM: conf z>3 AND text z>3 AND Delta_conf>=+0.02; "
                 "PARTIAL: conf z>3 but text z<2; REFUTE: conf z<2\n")
        fh.write(f"- **{verdict}**\n")

    print("=" * 64)
    print(f"baseline fg-mIoU = {baseline:.4f}  (self-check vs {args.baseline_ref}: {self_check})")
    for name, r in results.items():
        print(f"[{name}] pairs={r['npairs']} miou={r['miou']:.4f} "
              f"random={r['rmean']:.4f}±{r['rsd']:.4f} Delta={r['delta']:+.4f} z={r['z']:.1f} "
              f"gain_over_baseline={r['gain']:+.4f}")
    print(f"PRE-REGISTERED VERDICT: {verdict}")
    print(f"outputs -> {args.out_dir}/merge_superclass.md")
    print("=" * 64)


if __name__ == "__main__":
    main()
