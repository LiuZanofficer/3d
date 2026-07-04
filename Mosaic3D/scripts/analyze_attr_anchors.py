"""A2/A3/A4-offline: evaluate attribute anchors on GT-instance pooled features.
Offline is an oracle upper-bound diagnostic (GT-instance pooling allowed for scoring/oracle
only). Deployable numbers come from the online eval with masks_binary.
Metrics: sibling cos (name vs attr), cos(d_vis,d_txt) for sibling pairs, sibling naming acc
(restricted-to-cluster), full-200 instance acc, instance-renaming fg-mIoU upper bound.
"""
import argparse, glob, json, os
import numpy as np

def load_feats(feat_dir):
    feats, tcls, npts = [], [], []
    cn = None; fg = None
    for f in sorted(glob.glob(os.path.join(feat_dir, "*.npz"))):
        d = np.load(f, allow_pickle=True)
        feats.append(np.asarray(d["pooled_feat"], dtype=np.float64))
        tcls.append(np.asarray(d["true_class"]).reshape(-1).astype(int))
        npts.append(np.asarray(d["n_points"]).reshape(-1).astype(np.int64))
        if cn is None:
            cn = [str(x) for x in np.asarray(d["class_names"])]
            fg = np.asarray(d["fg_class_idx"]).reshape(-1).astype(int)
    F = np.concatenate(feats, 0)
    F = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-12)
    return F, np.concatenate(tcls, 0), np.concatenate(npts, 0), cn, fg

def norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)

def fg_miou_from_conf(true_c, pred_c, w, fg):
    C = 200
    TP = np.zeros(C); FP = np.zeros(C); FN = np.zeros(C)
    for t, p, n in zip(true_c, pred_c, w):
        if t == p: TP[t] += n
        else: FP[p] += n; FN[t] += n
    ious = []
    for c in fg:
        denom = TP[c] + FP[c] + FN[c]
        if denom > 0: ious.append(TP[c] / denom)
    return float(np.mean(ious)) if ious else 0.0

def build_scores(F, anchors_variant, fg):
    # anchors_variant: dict describing how to score -> returns [N,200]
    kind = anchors_variant["kind"]
    if kind == "single_vec":
        A = anchors_variant["A"]  # [200,D] normalized
        S = F @ A.T
    elif kind == "maxsim":
        emb_all = anchors_variant["emb_all"]  # [M,D]
        class_of = anchors_variant["class_of"]  # [M]
        allS = F @ emb_all.T  # [N,M]
        S = np.full((F.shape[0], 200), -1e9)
        for c in range(200):
            cols = np.where(class_of == c)[0]
            if len(cols): S[:, c] = allS[:, cols].max(1)
    mask = np.full(200, -1e9); mask[fg] = 0.0
    return S + mask[None, :]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default="/workspace/Mosaic3D/logs/eval/runs/2026-06-27_06-58-17/eval_instance_features/scannet200")
    ap.add_argument("--anchors", default="/root/Mosaic3D_work/error_analysis/reports/attr_anchors.npz")
    ap.add_argument("--clusters", default="/root/Mosaic3D_work/error_analysis/reports/task2/clusters_thr0.90.npz")
    ap.add_argument("--max-cluster-size", type=int, default=8)
    ap.add_argument("--out", default="/root/Mosaic3D_work/error_analysis/reports/task2/attr_offline.json")
    args = ap.parse_args()

    F, tcls, npts, cn, fg = load_feats(args.feat_dir)
    print("instances=%d dim=%d" % F.shape)
    an = np.load(args.anchors, allow_pickle=True)
    name_emb = norm(np.asarray(an["name_emb"], dtype=np.float64))
    mean_emb = norm(np.asarray(an["mean_emb"], dtype=np.float64))
    emb_all = norm(np.asarray(an["emb_all"], dtype=np.float64))
    class_of = np.asarray(an["class_of"]).astype(int)

    cl = np.load(args.clusters, allow_pickle=True)
    cluster_id = np.asarray(cl["cluster_id"]).astype(int)
    fgset = set(fg.tolist())
    # tight multi-member clusters (size 2..cap), fg-only members
    from collections import defaultdict
    members = defaultdict(list)
    for c, cid in enumerate(cluster_id):
        if c in fgset: members[cid].append(c)
    tight = {cid: ms for cid, ms in members.items() if 2 <= len(ms) <= args.max_cluster_size}
    tight_members = sorted({c for ms in tight.values() for c in ms})
    print("tight clusters=%d covering %d fg classes" % (len(tight), len(tight_members)))

    # class->cluster members map for restricted naming
    c2cluster = {}
    for cid, ms in tight.items():
        for c in ms: c2cluster[c] = ms

    # instances whose true class is in a tight cluster
    in_tight = np.array([t in c2cluster for t in tcls])
    print("instances in tight clusters=%d" % int(in_tight.sum()))

    # anchor variants (A2)
    variants = {}
    variants["name(baseline)"] = {"kind": "single_vec", "A": name_emb}
    variants["attr_mean"] = {"kind": "single_vec", "A": mean_emb}
    variants["attr_single"] = {"kind": "single_vec", "A": norm(emb_all[np.array([np.where(class_of==c)[0][0] for c in range(200)])])}
    variants["attr_maxsim"] = {"kind": "maxsim", "emb_all": emb_all, "class_of": class_of}
    for w in (0.3, 0.5, 0.7):
        fused = norm(w * name_emb + (1 - w) * mean_emb)
        variants["fuse%.1f(name*w+attr)" % w] = {"kind": "single_vec", "A": fused}

    results = {}
    for vname, vdef in variants.items():
        S = build_scores(F, vdef, fg)
        pred_full = S.argmax(1)
        # full-200 instance acc (unweighted / point-weighted)
        acc_inst = float((pred_full == tcls).mean())
        acc_pts = float((npts * (pred_full == tcls)).sum() / npts.sum())
        # sibling naming acc: restrict to cluster members
        correct = 0; tot = 0; cpts = 0; tpts = 0
        pred_sib = pred_full.copy()
        for i in np.where(in_tight)[0]:
            ms = c2cluster[tcls[i]]
            sub = S[i, ms]
            p = ms[int(sub.argmax())]
            pred_sib[i] = p
            tot += 1; tpts += npts[i]
            if p == tcls[i]: correct += 1; cpts += npts[i]
        sib_acc = correct / max(tot, 1)
        sib_acc_pts = cpts / max(tpts, 1)
        # instance-renaming fg-mIoU upper bound (full-200 argmax)
        miou_full = fg_miou_from_conf(tcls, pred_full, npts, fg)
        # instance-renaming fg-mIoU with in-cluster restricted naming (others keep full argmax)
        miou_sib = fg_miou_from_conf(tcls, pred_sib, npts, fg)
        results[vname] = dict(acc_inst=round(acc_inst,4), acc_pts=round(acc_pts,4),
                              sib_acc=round(sib_acc,4), sib_acc_pts=round(sib_acc_pts,4),
                              miou_rename_full=round(miou_full,4), miou_rename_incluster=round(miou_sib,4))
        print("%-26s full_acc=%.4f pts=%.4f | sibNaming acc=%.4f pts=%.4f | miou_full=%.4f miou_incluster=%.4f"
              % (vname, acc_inst, acc_pts, sib_acc, sib_acc_pts, miou_full, miou_sib))

    # A3 mechanism: sibling cos (name vs attr) per tight cluster + overall
    def cluster_cos(A):
        vals = []
        per = {}
        for cid, ms in tight.items():
            if len(ms) < 2: continue
            sub = A[ms]
            cc = sub @ sub.T
            iu = cc[np.triu_indices(len(ms), 1)]
            per[cid] = float(np.mean(np.abs(iu)))
            vals.extend(np.abs(iu).tolist())
        return float(np.mean(vals)), per
    name_cos, name_per = cluster_cos(name_emb)
    attr_cos, attr_per = cluster_cos(mean_emb)
    print("\n[A3] sibling anchor |cos|  name=%.4f  attr_mean=%.4f  (want attr << name)" % (name_cos, attr_cos))

    # A3 cos(d_vis, d_txt) for sibling pairs (d_vis from class-mean features, GT for analysis only)
    clsmean = {}
    for c in tight_members:
        idx = np.where(tcls == c)[0]
        if len(idx) >= 2: clsmean[c] = norm(F[idx].mean(0)[None, :])[0]
    def dvis_dtxt(A):
        vals = []
        for cid, ms in tight.items():
            ms2 = [c for c in ms if c in clsmean]
            for a in range(len(ms2)):
                for b in range(a+1, len(ms2)):
                    i, j = ms2[a], ms2[b]
                    dv = clsmean[i] - clsmean[j]; dt = A[i] - A[j]
                    dv = dv/max(np.linalg.norm(dv),1e-12); dt = dt/max(np.linalg.norm(dt),1e-12)
                    vals.append(abs(float(dv @ dt)))
        return float(np.mean(vals)) if vals else 0.0
    name_align = dvis_dtxt(name_emb); attr_align = dvis_dtxt(mean_emb)
    print("[A3] cos(d_vis,d_txt) sibling pairs  name=%.4f  attr_mean=%.4f  (baseline ~0.03; want up)" % (name_align, attr_align))

    out = dict(n_instances=int(F.shape[0]), tight_clusters=len(tight), tight_members=len(tight_members),
               n_in_tight=int(in_tight.sum()),
               variants=results,
               sibling_cos=dict(name=round(name_cos,4), attr_mean=round(attr_cos,4),
                                per_cluster_name={str(k):round(v,4) for k,v in name_per.items()},
                                per_cluster_attr={str(k):round(v,4) for k,v in attr_per.items()}),
               dvis_dtxt=dict(name=round(name_align,4), attr_mean=round(attr_align,4)))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=1)
    print("\nsaved", args.out)

if __name__ == "__main__":
    main()
