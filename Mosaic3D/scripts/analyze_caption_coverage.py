"""V3 (Phase 3) - is a class "dead" because its name is missing from training captions?

Counts, over the scannet training captions (anno_sources gsam2+seem, the in-domain
source for scannet200 eval), how many captions contain each fg class name as a whole
word, then compares that coverage `freq` against the class's IoU.

PRE-REGISTERED VERDICT (decided before running; do NOT change after seeing data):
  dead = fg classes with IoU < 0.01 ; rho = Spearman(log(freq+1), IoU) over fg classes.
  CONFIRM (coverage gap explains dead):
        rho >= +0.40 AND median(freq[dead]) <= 0.25 * median(freq[live])
  REFUTE (coverage not the main cause):
        rho <= +0.10 OR median(freq[dead]) >= median(freq[live])
  INCONCLUSIVE: otherwise
'object' is flagged as a pathological catch-all class and excluded from the rho fit.

Scope limitation: only scannet-domain captions are counted (arkit / scannet++ not
included). Read-only; needs numpy + scipy + the project's io util (run in docker).
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.io import unpack_list_of_np_arrays  # noqa: E402


def read_report(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append((r["class"], int(r["is_fg"]), float(r["iou"])))
    return rows


def iter_captions(caption_file):
    """Yield every caption string in a captions.<src>.npz file."""
    objects = unpack_list_of_np_arrays(caption_file)  # list (per object) of arrays of strings
    for arr in objects:
        for s in np.asarray(arr).reshape(-1):
            if isinstance(s, bytes):
                s = s.decode("utf-8", "ignore")
            yield str(s)


def main():
    ap = argparse.ArgumentParser(description="V3: dead-class caption coverage vs IoU.")
    ap.add_argument("--report", type=Path, required=True, help="per_class_report.csv")
    ap.add_argument("--scene-root", type=str, default="/datasets/mosaic3d/data/scannet")
    ap.add_argument("--anno-sources", type=str, default="gsam2,seem")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--dead-iou", type=float, default=0.01)
    args = ap.parse_args()

    rows = read_report(args.report)
    fg_classes = [c for c, fg, _ in rows if fg == 1]
    iou_of = {c: i for c, fg, i in rows if fg == 1}

    # whole-word regex over all fg names (longest first to prefer specific names)
    names_sorted = sorted(fg_classes, key=len, reverse=True)
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in names_sorted) + r")\b")

    freq = {c: 0 for c in fg_classes}
    n_captions = 0
    sources = [s.strip() for s in args.anno_sources.split(",")]
    root = Path(args.scene_root)
    cap_files = []
    for src in sources:
        cap_files.extend(sorted(root.glob(f"*/captions.{src}.npz")))
    print(f"scanning {len(cap_files)} caption files ({sources}) ...")
    for k, cf in enumerate(cap_files):
        try:
            for s in iter_captions(cf):
                n_captions += 1
                hits = set(pattern.findall(s.lower()))
                for h in hits:
                    freq[h] += 1
        except Exception as e:  # noqa: BLE001
            print(f"  skip {cf}: {e}")
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{len(cap_files)} files, {n_captions} captions so far")

    # ---- stats ----
    from scipy.stats import spearmanr
    fit_classes = [c for c in fg_classes if c != "object"]
    f = np.array([freq[c] for c in fit_classes], dtype=np.float64)
    iou = np.array([iou_of[c] for c in fit_classes], dtype=np.float64)
    rho, pval = spearmanr(np.log(f + 1.0), iou)

    dead = [c for c in fit_classes if iou_of[c] < args.dead_iou]
    live = [c for c in fit_classes if iou_of[c] >= args.dead_iou]
    med_dead = float(np.median([freq[c] for c in dead])) if dead else float("nan")
    med_live = float(np.median([freq[c] for c in live])) if live else float("nan")

    if rho >= 0.40 and med_dead <= 0.25 * med_live:
        verdict = "CONFIRM (coverage gap explains dead classes)"
    elif rho <= 0.10 or (not np.isnan(med_live) and med_dead >= med_live):
        verdict = "REFUTE (coverage not the main cause)"
    else:
        verdict = "INCONCLUSIVE"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "caption_coverage.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "is_fg", "iou", "caption_freq"])
        for c in fg_classes:
            w.writerow([c, 1, f"{iou_of[c]:.4f}", freq[c]])

    zero_cov_dead = [c for c in dead if freq[c] == 0]
    with open(args.out_dir / "caption_coverage.md", "w") as fh:
        fh.write("# V3 dead-class caption coverage vs IoU\n\n")
        fh.write(f"- sources={sources}, caption files={len(cap_files)}, total captions={n_captions}\n")
        fh.write(f"- Spearman(log(freq+1), IoU) over {len(fit_classes)} fg classes (excl. 'object'): "
                 f"**rho={rho:.3f}** (p={pval:.2e})\n")
        fh.write(f"- median freq: dead(IoU<{args.dead_iou}, n={len(dead)})={med_dead:.0f}  "
                 f"live(n={len(live)})={med_live:.0f}\n")
        fh.write(f"- dead classes with ZERO caption coverage: {len(zero_cov_dead)}/{len(dead)} "
                 f"-> {zero_cov_dead}\n\n")
        fh.write("## PRE-REGISTERED verdict\n")
        fh.write("- CONFIRM: rho>=+0.40 AND median(freq[dead])<=0.25*median(freq[live]); "
                 "REFUTE: rho<=+0.10 OR median(freq[dead])>=median(freq[live])\n")
        fh.write(f"- **{verdict}**\n\n")
        fh.write("## Lowest-coverage fg classes\n\n| class | iou | caption_freq |\n|---|---|---|\n")
        for c in sorted(fg_classes, key=lambda x: freq[x])[:30]:
            fh.write(f"| {c} | {iou_of[c]:.3f} | {freq[c]} |\n")

    print("=" * 64)
    print(f"total captions={n_captions}  rho(log freq, IoU)={rho:.3f} (p={pval:.2e})")
    print(f"median freq dead={med_dead:.0f} live={med_live:.0f}  "
          f"(dead with zero coverage: {len(zero_cov_dead)}/{len(dead)})")
    print(f"PRE-REGISTERED VERDICT: {verdict}")
    print(f"outputs -> {args.out_dir}/caption_coverage.md, caption_coverage.csv")
    print("=" * 64)


if __name__ == "__main__":
    main()
