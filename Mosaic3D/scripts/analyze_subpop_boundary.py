"""S1 (new direction) - are the misrouted points of a victim class boundary points?

Evidence so far: sibling confusion is real & concentrated, but it's NOT a class-mean
text-decision effect (V1b refuted). So the misrouting happens for a SUBPOPULATION of
each victim class's points. S1 tests the simplest per-point handle: are the points that
leak to the top-1 absorber located near class boundaries (vs interior)?

For each victim->absorber pair (C -> J), per scene with gt==C:
  boundary_score(point) = fraction of its k nearest 3D neighbors whose GT label != C.
  Compare boundary_score of  FN->J points (gt=C & pred=J)  vs  TP points (gt=C & pred=C).

PRE-REGISTERED VERDICT (decided before running; do NOT change after seeing data):
  diff = mean_over_pairs( mean_boundary(FN->J) - mean_boundary(TP) )
  frac_pos = fraction of pairs with that per-pair difference >= 0
  CONFIRM (boundary-driven): diff >= +0.15 AND frac_pos >= 0.70
  REFUTE (not boundary):     diff <= +0.03
  INCONCLUSIVE: otherwise (escalate to S2 instance-size / per-point features)

Read-only; needs numpy + scipy (cKDTree). Run inside the docker container.
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
        flat = gt_fg[valid] * num_classes + pred[valid]
        cm += np.bincount(flat, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return cm


def main():
    ap = argparse.ArgumentParser(description="S1: are misrouted victim points boundary points?")
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--scene-root", type=str, default="/datasets/mosaic3d/data/scannet")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--topk-cost", type=int, default=25)
    ap.add_argument("--k", type=int, default=16, help="num 3D neighbors for boundary score")
    args = ap.parse_args()

    from scipy.spatial import cKDTree

    files = sorted(args.dump_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz under {args.dump_dir}")
    first = load_dump(files[0])
    if "pred_semantic" not in first:
        raise SystemExit("Old dump (no pred_semantic). Re-run eval with extended dump first.")
    class_names = [str(c) for c in np.asarray(first["class_names"])]
    num_classes = len(class_names)
    fg_idx = np.asarray(first["fg_class_idx"]).reshape(-1).astype(int)
    bg_idx = np.asarray(first.get("bg_class_idx", np.array([]))).reshape(-1).astype(int)
    ignore_label = int(first.get("ignore_label", -100))

    cm = build_global_confmat(files, num_classes, bg_idx, ignore_label)
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
    topk = sorted(absorber, key=lambda c: cost[c], reverse=True)[: args.topk_cost]
    J_of = {c: absorber[c] for c in topk}

    # accumulators per victim class
    sum_tp = {c: 0.0 for c in topk}
    n_tp = {c: 0 for c in topk}
    sum_fnj = {c: 0.0 for c in topk}
    n_fnj = {c: 0 for c in topk}

    topk_set = set(topk)
    for f in files:
        d = load_dump(f)
        gt = np.asarray(d["gt_segment"]).astype(np.int64)
        pred = np.asarray(d["pred_semantic"]).astype(np.int64)
        present = [c for c in topk_set if np.any(gt == c)]
        if not present:
            continue
        p = Path(args.scene_root) / str(d["scene_name"]) / "coord.npy"
        if not p.exists():
            continue
        coord = np.load(p)
        if coord.shape[0] != gt.shape[0]:
            continue
        tree = cKDTree(coord)
        for c in present:
            mask_c = gt == c
            pts = coord[mask_c]
            if pts.shape[0] < 2:
                continue
            kk = min(args.k + 1, coord.shape[0])
            _, nn = tree.query(pts, k=kk)
            nn = nn[:, 1:]  # drop self
            neigh_gt = gt[nn]  # [Npts, k]
            bscore = np.mean(neigh_gt != c, axis=1)  # boundary score per victim point
            pred_c = pred[mask_c]
            tp_m = pred_c == c
            fnj_m = pred_c == J_of[c]
            sum_tp[c] += float(bscore[tp_m].sum())
            n_tp[c] += int(tp_m.sum())
            sum_fnj[c] += float(bscore[fnj_m].sum())
            n_fnj[c] += int(fnj_m.sum())

    rows = []
    diffs = []
    for c in topk:
        if n_tp[c] == 0 or n_fnj[c] == 0:
            continue
        m_tp = sum_tp[c] / n_tp[c]
        m_fnj = sum_fnj[c] / n_fnj[c]
        diff = m_fnj - m_tp
        diffs.append(diff)
        rows.append((class_names[c], class_names[J_of[c]], m_tp, m_fnj, diff,
                     n_tp[c], n_fnj[c], int(cost[c])))

    diffs = np.asarray(diffs)
    headline = float(diffs.mean()) if diffs.size else float("nan")
    frac_pos = float(np.mean(diffs >= 0)) if diffs.size else float("nan")

    if headline >= 0.15 and frac_pos >= 0.70:
        verdict = "CONFIRM (boundary-driven misrouting)"
    elif headline <= 0.03:
        verdict = "REFUTE (not a boundary effect)"
    else:
        verdict = "INCONCLUSIVE"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "subpop_boundary.md", "w") as fh:
        fh.write("# S1 boundary subpopulation test\n\n")
        fh.write(f"- k={args.k} neighbors; boundary_score = frac of 3D neighbors with GT != victim class\n")
        fh.write(f"- pairs analyzed: {len(rows)} (top-{args.topk_cost} cost)\n")
        fh.write(f"- **mean per-pair diff (FN->J minus TP) = {headline:+.4f}**, frac pairs positive = {frac_pos:.2f}\n\n")
        fh.write("## PRE-REGISTERED verdict\n")
        fh.write("- CONFIRM: diff>=+0.15 AND frac_pos>=0.70; REFUTE: diff<=+0.03\n")
        fh.write(f"- **{verdict}**\n\n")
        fh.write("## Per-pair (boundary score)\n\n")
        fh.write("| C | absorber J | TP bnd | FN->J bnd | diff | n_TP | n_FN->J | cost |\n")
        fh.write("|---|---|---|---|---|---|---|---|\n")
        for nc, nj, mt, mf, df, nt, nf, co in rows:
            fh.write(f"| {nc} | {nj} | {mt:.3f} | {mf:.3f} | {df:+.3f} | {nt} | {nf} | {co} |\n")

    print("=" * 64)
    print(f"pairs={len(rows)}  mean per-pair diff(FN->J - TP)={headline:+.4f}  frac_pos={frac_pos:.2f}")
    print(f"PRE-REGISTERED VERDICT: {verdict}")
    print(f"outputs -> {args.out_dir}/subpop_boundary.md")
    print("=" * 64)


if __name__ == "__main__":
    main()
