"""V2 - part-whole test: are small-object points physically on/in their container?

Mechanism 2 hypothesis: small objects (book, pillow, doorframe) get their
container's label (bookshelf, bed, door) because they physically sit on/in it,
not because of text confusion. If so, the small-class points that were predicted
as the container should lie spatially inside / adjacent to the container surface.

For each pair (small, container) and each scene that has both GT classes:
  target = coords[gt==small & pred_semantic==container]   (the part-whole misses)
  frac_in = fraction of `target` within `--dist` of any GT-container point.
CONTROL: same target points vs a random UNRELATED big class present in the scene
  (frac_rand), to rule out "any large furniture is nearby".

PRE-REGISTERED VERDICT (decided before running; do NOT change after seeing data):
  frac_in = point-weighted fraction over all scenes; dist = 5cm (KNN to surface).
  CONFIRM (physical containment, needs geometric fix):
        frac_in >= 0.70 AND (frac_in - frac_rand) >= 0.30
  REFUTE (not containment): frac_in <= 0.30 OR (frac_in - frac_rand) <= 0.10
  INCONCLUSIVE: otherwise (mixed mechanism; treat as limitation, not main line)

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


def class_index(class_names, name):
    for i, c in enumerate(class_names):
        if str(c) == name:
            return i
    return -1


def frac_within(target_pts, ref_pts, dist):
    """Fraction of target points within `dist` of any ref point."""
    if target_pts.shape[0] == 0 or ref_pts.shape[0] == 0:
        return None, 0
    from scipy.spatial import cKDTree
    tree = cKDTree(ref_pts)
    d, _ = tree.query(target_pts, k=1)
    return float(np.mean(d <= dist)), int(target_pts.shape[0])


def main():
    ap = argparse.ArgumentParser(description="V2: part-whole geometric containment test.")
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--scene-root", type=str, default="/datasets/mosaic3d/data/scannet")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--pairs", type=str, default="book:bookshelf,pillow:bed,doorframe:door")
    ap.add_argument("--control-classes", type=str,
                    default="refrigerator,couch,door,window,toilet,sink,bathtub")
    ap.add_argument("--dist", type=float, default=0.05, help="surface KNN distance in meters")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(args.dump_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No .npz under {args.dump_dir}")
    first = load_dump(files[0])
    if "pred_semantic" not in first:
        raise SystemExit("Old dump (no pred_semantic). Re-run eval with the extended dump first.")
    class_names = [str(c) for c in np.asarray(first["class_names"])]

    pairs = []
    for tok in args.pairs.split(","):
        s, c = tok.split(":")
        pairs.append((s.strip(), c.strip()))
    control_names = [x.strip() for x in args.control_classes.split(",")]
    rng = np.random.default_rng(args.seed)

    # cache coords per scene lazily
    def scene_coords(scene_name, n_expected):
        p = Path(args.scene_root) / scene_name / "coord.npy"
        if not p.exists():
            return None
        coord = np.load(p)
        if coord.shape[0] != n_expected:
            return None
        return coord

    report_lines = ["# V2 part-whole geometric containment\n",
                    f"- surface KNN distance = {args.dist*100:.0f} cm; frac is point-weighted over scenes\n"]
    print("=" * 64)
    overall = {}
    for small, container in pairs:
        si = class_index(class_names, small)
        ci = class_index(class_names, container)
        if si < 0 or ci < 0:
            report_lines.append(f"\n## {small} -> {container}: class not found, skipped\n")
            continue
        ctrl_idx = [class_index(class_names, n) for n in control_names]
        ctrl_idx = [x for x in ctrl_idx if x >= 0 and x != ci]

        tgt_in = 0  # weighted numerator
        tgt_tot = 0
        ctrl_in = 0
        ctrl_tot = 0
        n_scenes = 0
        for f in files:
            d = load_dump(f)
            gt = np.asarray(d["gt_segment"]).astype(np.int64)
            pred = np.asarray(d["pred_semantic"]).astype(np.int64)
            target_mask = (gt == si) & (pred == ci)
            if not target_mask.any():
                continue
            coord = scene_coords(str(d["scene_name"]), gt.shape[0])
            if coord is None:
                continue
            cont_pts = coord[gt == ci]
            if cont_pts.shape[0] == 0:
                continue
            target_pts = coord[target_mask]
            f_in, n = frac_within(target_pts, cont_pts, args.dist)
            if f_in is None:
                continue
            tgt_in += f_in * n
            tgt_tot += n
            n_scenes += 1
            # control: nearest among available control-class GT in this scene
            present_ctrl = [k for k in ctrl_idx if (gt == k).sum() > 0]
            if present_ctrl:
                k = int(rng.choice(present_ctrl))
                f_c, _ = frac_within(target_pts, coord[gt == k], args.dist)
                if f_c is not None:
                    ctrl_in += f_c * n
                    ctrl_tot += n

        frac_in = tgt_in / tgt_tot if tgt_tot else float("nan")
        frac_rand = ctrl_in / ctrl_tot if ctrl_tot else float("nan")
        diff = frac_in - frac_rand if ctrl_tot else float("nan")

        if not np.isnan(frac_in) and frac_in >= 0.70 and (ctrl_tot and diff >= 0.30):
            verdict = "CONFIRM (physical containment)"
        elif not np.isnan(frac_in) and (frac_in <= 0.30 or (ctrl_tot and diff <= 0.10)):
            verdict = "REFUTE (not containment)"
        else:
            verdict = "INCONCLUSIVE"

        overall[(small, container)] = (frac_in, frac_rand, diff, tgt_tot, n_scenes, verdict)
        report_lines.append(
            f"\n## {small} -> {container}\n"
            f"- scenes={n_scenes}, target points (gt={small} & pred={container})={tgt_tot}\n"
            f"- frac within {args.dist*100:.0f}cm of true {container}: **{frac_in:.3f}**\n"
            f"- random-container control: {frac_rand:.3f}  (diff={diff:+.3f})\n"
            f"- **{verdict}**\n")
        print(f"[{small}->{container}] frac_in={frac_in:.3f} frac_rand={frac_rand:.3f} "
              f"diff={diff:+.3f} pts={tgt_tot} scenes={n_scenes} -> {verdict}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "part_whole.md", "w") as fh:
        fh.write("".join(report_lines))
        fh.write("\n## PRE-REGISTERED verdict rule\n")
        fh.write("- CONFIRM: frac_in>=0.70 AND (frac_in-frac_rand)>=0.30; "
                 "REFUTE: frac_in<=0.30 OR diff<=0.10\n")
    print(f"outputs -> {args.out_dir}/part_whole.md")
    print("=" * 64)


if __name__ == "__main__":
    main()
