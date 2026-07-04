"""Per-class fg-mIoU + sibling-cluster decomposition over scene dumps.
fg-mIoU 口径: pred=pred_semantic; ignore=(gt<0)|(gt>=C)|(gt in bg); IoU over fg classes.
Compares dump dirs and decomposes gain into tight-sibling-cluster classes vs the rest.
"""
import argparse, glob, json, os
import numpy as np

C = 200

def score_dir(d, fg, bg):
    TP = np.zeros(C); FP = np.zeros(C); FN = np.zeros(C)
    bgset = set(bg.tolist())
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        gt = np.asarray(z["gt_segment"]).reshape(-1).astype(np.int64)
        pr = np.asarray(z["pred_semantic"]).reshape(-1).astype(np.int64)
        valid = (gt >= 0) & (gt < C)
        valid &= ~np.isin(gt, list(bgset))
        gt = gt[valid]; pr = pr[valid]
        for c in fg:
            pc = pr == c; gc = gt == c
            TP[c] += np.sum(pc & gc); FP[c] += np.sum(pc & ~gc); FN[c] += np.sum(~pc & gc)
    iou = {}
    for c in fg:
        den = TP[c] + FP[c] + FN[c]
        iou[c] = float(TP[c] / den) if den > 0 else float("nan")
    return iou, TP, FP, FN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default="/workspace/Mosaic3D/logs/eval/runs/2026-06-27_06-58-17/eval_scene_dumps/scannet200")
    ap.add_argument("--name-mtv", default="/root/runs/attr_dump/name_mtv/eval_scene_dumps/scannet200")
    ap.add_argument("--fuse-mtv", default="/root/runs/attr_dump/fuse07_mtv/eval_scene_dumps/scannet200")
    ap.add_argument("--clusters", default="/root/Mosaic3D_work/error_analysis/reports/task2/clusters_thr0.90.npz")
    ap.add_argument("--out", default="/root/Mosaic3D_work/error_analysis/reports/attr_online/per_class_breakdown.json")
    args = ap.parse_args()

    z0 = np.load(sorted(glob.glob(args.baseline + "/*.npz"))[0], allow_pickle=True)
    class_names = [str(x) for x in np.asarray(z0["class_names"])]
    fg = np.asarray(z0["fg_class_idx"]).reshape(-1).astype(int)
    bg = np.asarray(z0["bg_class_idx"]).reshape(-1).astype(int)

    cl = np.load(args.clusters, allow_pickle=True)
    cid = np.asarray(cl["cluster_id"]).astype(int)
    from collections import defaultdict
    mem = defaultdict(list); fgset = set(fg.tolist())
    for c, g in enumerate(cid):
        if c in fgset: mem[g].append(c)
    tight_members = sorted({c for g, ms in mem.items() if 2 <= len(ms) <= 8 for c in ms})
    tight_set = set(tight_members)

    runs = {}
    def miou(iou): 
        vals=[v for v in iou.values() if not np.isnan(v)]
        return float(np.mean(vals))
    iou_b, *_ = score_dir(args.baseline, fg, bg)
    iou_n, *_ = score_dir(args.name_mtv, fg, bg)
    iou_f, *_ = score_dir(args.fuse_mtv, fg, bg)
    print("fg-mIoU  baseline(name+base)=%.5f  name+MTV=%.5f  fuse0.7+MTV=%.5f" % (miou(iou_b), miou(iou_n), miou(iou_f)))

    # decomposition: mean IoU over sibling vs non-sibling; and sum of per-class deltas
    def submean(iou, idxs):
        vals=[iou[c] for c in idxs if not np.isnan(iou[c])]
        return float(np.mean(vals)) if vals else float("nan")
    nonsib = [c for c in fg if c not in tight_set]
    for label, ref in [("fuse0.7+MTV vs name+MTV", iou_n), ("fuse0.7+MTV vs baseline", iou_b)]:
        d_sib = submean(iou_f, tight_members) - submean(ref, tight_members)
        d_non = submean(iou_f, nonsib) - submean(ref, nonsib)
        print("[decomp] %-28s  sibling dMeanIoU=%+.4f  non-sibling dMeanIoU=%+.4f" % (label, d_sib, d_non))

    # top per-class movers (fuse vs name+MTV)
    deltas = sorted([(class_names[c], round(iou_f[c]-iou_n[c],4), round(iou_n[c],4), round(iou_f[c],4), c in tight_set)
                     for c in fg if not (np.isnan(iou_f[c]) or np.isnan(iou_n[c]))], key=lambda x: x[1])
    print("\nTop 12 losers (fuse0.7+MTV - name+MTV):")
    for nm,dl,a,b,sib in deltas[:12]: print("  %-22s %+.4f  (%.3f->%.3f) sib=%s"%(nm,dl,a,b,sib))
    print("Top 12 gainers:")
    for nm,dl,a,b,sib in deltas[-12:][::-1]: print("  %-22s %+.4f  (%.3f->%.3f) sib=%s"%(nm,dl,a,b,sib))

    out = dict(
        miou=dict(baseline=round(miou(iou_b),5), name_mtv=round(miou(iou_n),5), fuse07_mtv=round(miou(iou_f),5)),
        sibling_members=[class_names[c] for c in tight_members],
        decomp_vs_name_mtv=dict(sibling=round(submean(iou_f,tight_members)-submean(iou_n,tight_members),4),
                                 nonsibling=round(submean(iou_f,nonsib)-submean(iou_n,nonsib),4)),
        decomp_vs_baseline=dict(sibling=round(submean(iou_f,tight_members)-submean(iou_b,tight_members),4),
                                 nonsibling=round(submean(iou_f,nonsib)-submean(iou_b,nonsib),4)),
        per_class=[{"cls":class_names[c],"idx":int(c),"sib":c in tight_set,
                    "iou_base":round(iou_b[c],4) if not np.isnan(iou_b[c]) else None,
                    "iou_name_mtv":round(iou_n[c],4) if not np.isnan(iou_n[c]) else None,
                    "iou_fuse_mtv":round(iou_f[c],4) if not np.isnan(iou_f[c]) else None} for c in fg],
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out,"w"), indent=1)
    print("\nsaved", args.out)

if __name__ == "__main__":
    main()
