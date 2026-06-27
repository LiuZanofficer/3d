"""S3 (new direction) - is the leak to the absorber decided per-INSTANCE (wholesale
flip) or scattered per-point?

S1 refuted the boundary hypothesis and showed the misrouted points are interior.
If whole GT instances of the victim class C are labeled as absorber J (not scattered
points), the right handle is instance/mask-level appearance, not point noise.

For each victim->absorber pair (C -> J), per GT instance of class C:
  frac_J = fraction of that instance's points predicted as J.
A "flipped" instance has frac_J >= 0.5.
  leak_in_flipped = (#FN->J points living in flipped instances) / (#FN->J points)
i.e. how concentrated the leak is in wholesale-flipped instances vs scattered across
otherwise-correct instances.

PRE-REGISTERED VERDICT (decided before running; do NOT change after seeing data):
  m = mean over pairs of leak_in_flipped.
  CONFIRM (instance-level wholesale flip): m >= 0.60
  REFUTE (scattered point-level leak):     m <= 0.30
  INCONCLUSIVE: otherwise
Also report the bimodality of per-instance frac_J (mass near 0/1 vs middle).

Read-only; numpy only. Run inside the docker container.
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
    ap = argparse.ArgumentParser(description="S3: is the absorber leak a per-instance wholesale flip?")
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--topk-cost", type=int, default=25)
    ap.add_argument("--flip-thr", type=float, default=0.5)
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
    topk_set = set(topk)

    # per pair accumulators
    fnj_total = {c: 0 for c in topk}
    fnj_in_flipped = {c: 0 for c in topk}
    frac_j_hist = {c: [] for c in topk}      # per-instance frac_J (weighted view via list of (frac,size))

    for f in files:
        d = load_dump(f)
        gt = np.asarray(d["gt_segment"]).astype(np.int64)
        pred = np.asarray(d["pred_semantic"]).astype(np.int64)
        inst = np.asarray(d["gt_instance"]).astype(np.int64)
        present = [c for c in topk_set if np.any(gt == c)]
        for c in present:
            J = J_of[c]
            mask_c = gt == c
            inst_c = inst[mask_c]
            pred_c = pred[mask_c]
            for iid in np.unique(inst_c):
                if iid < 0:
                    continue
                sel = inst_c == iid
                n = int(sel.sum())
                if n == 0:
                    continue
                pj = pred_c[sel]
                frac_J = float(np.mean(pj == J))
                n_fnj = int(np.sum(pj == J))
                fnj_total[c] += n_fnj
                if frac_J >= args.flip_thr:
                    fnj_in_flipped[c] += n_fnj
                frac_j_hist[c].append((frac_J, n))

    rows = []
    vals = []
    all_fracs = []
    for c in topk:
        if fnj_total[c] == 0:
            continue
        lif = fnj_in_flipped[c] / fnj_total[c]
        vals.append(lif)
        # instance-count level fraction of flipped instances
        fr = np.array([f for f, _ in frac_j_hist[c]])
        all_fracs.extend(fr.tolist())
        flipped_inst = float(np.mean(fr >= args.flip_thr)) if fr.size else float("nan")
        rows.append((class_names[c], class_names[J_of[c]], lif, flipped_inst, fnj_total[c],
                     len(frac_j_hist[c]), int(cost[c])))

    m = float(np.mean(vals)) if vals else float("nan")
    af = np.asarray(all_fracs)
    # bimodality: mass near 0/1 vs middle
    near01 = float(np.mean((af <= 0.1) | (af >= 0.9))) if af.size else float("nan")
    middle = float(np.mean((af > 0.3) & (af < 0.7))) if af.size else float("nan")

    if m >= 0.60:
        verdict = "CONFIRM (instance-level wholesale flip)"
    elif m <= 0.30:
        verdict = "REFUTE (scattered point-level leak)"
    else:
        verdict = "INCONCLUSIVE"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "subpop_instance.md", "w") as fh:
        fh.write("# S3 instance-level wholesale-flip test\n\n")
        fh.write(f"- flip threshold = {args.flip_thr}; pairs analyzed = {len(rows)}\n")
        fh.write(f"- **mean leak_in_flipped = {m:.3f}** (share of FN->J points living in flipped instances)\n")
        fh.write(f"- per-instance frac_J bimodality: near 0/1 = {near01:.2f}, middle(0.3-0.7) = {middle:.2f}\n\n")
        fh.write("## PRE-REGISTERED verdict\n")
        fh.write("- CONFIRM: mean>=0.60; REFUTE: mean<=0.30\n")
        fh.write(f"- **{verdict}**\n\n")
        fh.write("## Per-pair\n\n| C | absorber J | leak_in_flipped | flipped_inst_frac | n_FN->J | n_inst | cost |\n")
        fh.write("|---|---|---|---|---|---|---|\n")
        for nc, nj, lif, fi, nf, ni, co in rows:
            fh.write(f"| {nc} | {nj} | {lif:.3f} | {fi:.3f} | {nf} | {ni} | {co} |\n")

    print("=" * 64)
    print(f"pairs={len(rows)}  mean leak_in_flipped={m:.3f}  near0/1={near01:.2f} middle={middle:.2f}")
    print(f"PRE-REGISTERED VERDICT: {verdict}")
    print(f"outputs -> {args.out_dir}/subpop_instance.md")
    print("=" * 64)


if __name__ == "__main__":
    main()
