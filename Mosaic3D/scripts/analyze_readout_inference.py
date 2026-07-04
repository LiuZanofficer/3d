"""TASK2 offline readout-inference diagnosis (annotation-free methods).

Operates on existing eval dumps (no GPU, no retraining):
  - eval_instance_features/*.npz : per-GT-instance pooled_feat (768), true_class.
  - eval_scene_dumps/*.npz       : per-point gt_segment/gt_instance/pred_semantic.

Decision inputs for methods are features + text anchors only (annotation-free).
GT is used ONLY for scoring and for explicitly-labelled oracle ceilings.

fg-mIoU here is an *instance-renaming ceiling*: it repaints whole GT instances
(GT grouping) with the method's class, so it upper-bounds a deployable readout
that would instead group by Segment3D masks. It is a diagnostic, not a method.
Baseline point-level fg-mIoU (0.1548) is reproduced as a self-check.

Within-cluster methods only act on clusters with 2 <= size <= max_cluster_size,
because text-cosine connected components produce one huge blob of near-collinear
anchors on which within-cluster decorrelation/kmeans degenerate.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.readout_decorrelate import whiten_pick  # noqa: E402


def _decode(v):
    return v.item() if isinstance(v, np.ndarray) and v.shape == () else v


def load_npz(p):
    d = np.load(p, allow_pickle=True)
    return {k: _decode(d[k]) for k in d.files}


def scene_confmat(gt, pred, C):
    flat = gt.astype(np.int64) * C + pred.astype(np.int64)
    return np.bincount(flat, minlength=C * C).reshape(C, C)


def valid_gt_pred(gt, pred, C, bg_idx, ignore_label):
    gt_fg = gt.copy()
    if bg_idx.size:
        gt_fg[np.isin(gt_fg, bg_idx)] = ignore_label
    valid = (gt_fg != ignore_label) & (gt_fg >= 0) & (gt_fg < C) & (pred >= 0) & (pred < C)
    return gt_fg[valid], pred[valid]


def fg_miou(cm, fg_idx):
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(0).astype(np.float64) - tp
    fn = cm.sum(1).astype(np.float64) - tp
    u = tp + fp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(u > 0, tp / u, 0.0)
    return float(np.mean(iou[fg_idx])), iou


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--scene-dump-dir", type=Path, required=True)
    ap.add_argument("--text-embeddings", type=Path, required=True)
    ap.add_argument("--clusters", type=Path, required=True, help="clusters_thrX.npz")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--proto-path", type=Path, default=None, help="clean caption prototypes npz")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps-ratio", type=float, default=1e-2)
    ap.add_argument("--max-cluster-size", type=int, default=8)
    args = ap.parse_args()

    te = load_npz(args.text_embeddings)
    emb = np.asarray(te["emb"], dtype=np.float64)
    T = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    class_names = [str(c) for c in np.asarray(te["class_names"])]
    C = len(class_names)
    fg_idx = np.asarray(te["fg_class_idx"]).reshape(-1).astype(int)
    fg_mask = np.zeros(C, dtype=bool)
    fg_mask[fg_idx] = True

    cl = load_npz(args.clusters)
    cluster_id = np.asarray(cl["cluster_id"]).astype(int)
    thr = float(cl["threshold"])
    raw_clusters = {}
    for c in range(C):
        g = cluster_id[c]
        if g >= 0:
            raw_clusters.setdefault(g, []).append(c)
    cap = args.max_cluster_size
    # effective clusters: only tight sibling groups act; huge blob -> singletons
    clusters = {g: m for g, m in raw_clusters.items() if 2 <= len(m) <= cap}
    c2members = {}
    for m in clusters.values():
        for c in m:
            c2members[c] = m
    multi_class = np.array([c in c2members for c in range(C)])

    proto = None
    if args.proto_path is not None:
        pd = np.load(args.proto_path, allow_pickle=True)
        proto = pd["proto"].astype(np.float64)
        proto = proto / np.maximum(np.linalg.norm(proto, axis=1, keepdims=True), 1e-12)
        proto_count = pd["count"].astype(np.int64)

    feats, ytrue, scenes, iids = [], [], [], []
    for p in sorted(args.feature_dir.glob("*.npz")):
        d = load_npz(p)
        Xi = np.asarray(d["pooled_feat"], dtype=np.float64)
        feats.append(Xi)
        ytrue.append(np.asarray(d["true_class"], dtype=np.int64))
        scenes.append(np.array([str(d["scene_name"])] * len(Xi)))
        iids.append(np.asarray(d["gt_instance_id"], dtype=np.int64))
    X = np.concatenate(feats, 0)
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    ytrue = np.concatenate(ytrue, 0)
    scenes = np.concatenate(scenes, 0)
    iids = np.concatenate(iids, 0)
    N = len(X)

    sim_fg = X @ T.T
    sim_fg[:, ~fg_mask] = -np.inf
    base_top = np.argmax(sim_fg, axis=1)

    preds = {"baseline_inst": base_top.copy()}

    dec = base_top.copy()
    for i in range(N):
        members = c2members.get(int(base_top[i]))
        if members is None:
            continue
        j = whiten_pick(X[i], T[members], eps_ratio=args.eps_ratio)
        dec[i] = members[j]
    preds["anchor_decorrelate"] = dec

    if proto is not None:
        cp = base_top.copy()
        for i in range(N):
            members = c2members.get(int(base_top[i]))
            if members is None:
                continue
            valid = [c for c in members if proto_count[c] > 0]
            if len(valid) <= 1:
                continue
            cp[i] = valid[int(np.argmax(X[i] @ proto[valid].T))]
        preds["caption_proto"] = cp

    unsup_text = base_top.copy()
    unsup_oracle = base_top.copy()
    for g, members in clusters.items():
        sel = np.flatnonzero(np.isin(base_top, members))
        if len(sel) < len(members):
            continue
        k = len(members)
        km = KMeans(n_clusters=k, n_init=10, random_state=args.seed).fit(X[sel])
        lab = km.labels_
        m = np.array(members)
        for gi in range(k):
            gsel = sel[lab == gi]
            if len(gsel) == 0:
                continue
            centroid = X[gsel].mean(0)
            centroid /= max(np.linalg.norm(centroid), 1e-12)
            unsup_text[gsel] = m[int(np.argmax(centroid @ T[m].T))]
            vals, cnts = np.unique(ytrue[gsel], return_counts=True)
            unsup_oracle[gsel] = vals[int(np.argmax(cnts))]
    preds["unsup_kmeans_textname"] = unsup_text
    preds["unsup_kmeans_oracle"] = unsup_oracle

    orc = base_top.copy()
    for i in range(N):
        members = c2members.get(int(base_top[i]))
        if members is not None and ytrue[i] in members:
            orc[i] = ytrue[i]
    preds["oracle_incluster"] = orc

    sib = multi_class[ytrue]
    acc = {}
    for name, pr in preds.items():
        acc[name] = {
            "acc_all": float(np.mean(pr == ytrue)),
            "acc_sibling": float(np.mean(pr[sib] == ytrue[sib])) if sib.any() else float("nan"),
            "n_changed_vs_base": int(np.sum(pr != base_top)),
        }

    inst_pred_by_scene = {}
    for name, pr in preds.items():
        mm = {}
        for i in range(N):
            mm.setdefault(scenes[i], {})[int(iids[i])] = int(pr[i])
        inst_pred_by_scene[name] = mm

    bg_idx = None
    ignore_label = -100
    base_cm = None
    method_cm = {name: None for name in preds}
    for p in sorted(args.scene_dump_dir.glob("*.npz")):
        d = load_npz(p)
        sname = str(d["scene_name"])
        gt = np.asarray(d["gt_segment"]).astype(np.int64)
        base_pred = np.asarray(d["pred_semantic"]).astype(np.int64)
        ginst = np.asarray(d["gt_instance"]).astype(np.int64)
        if bg_idx is None:
            bg_idx = np.asarray(d.get("bg_class_idx", np.array([]))).reshape(-1).astype(int)
            ignore_label = int(d.get("ignore_label", -100))
        g, pcut = valid_gt_pred(gt, base_pred, C, bg_idx, ignore_label)
        cm = scene_confmat(g, pcut, C)
        base_cm = cm if base_cm is None else base_cm + cm
        for name in preds:
            pr = base_pred.copy()
            for iid, cls in inst_pred_by_scene[name].get(sname, {}).items():
                pr[ginst == iid] = cls
            g2, p2 = valid_gt_pred(gt, pr, C, bg_idx, ignore_label)
            cm2 = scene_confmat(g2, p2, C)
            method_cm[name] = cm2 if method_cm[name] is None else method_cm[name] + cm2

    base_point_miou, _ = fg_miou(base_cm, fg_idx)
    miou = {name: fg_miou(method_cm[name], fg_idx)[0] for name in preds}

    result = {
        "threshold": thr,
        "max_cluster_size": cap,
        "n_effective_multi_clusters": len(clusters),
        "n_instances": int(N),
        "n_sibling_instances": int(sib.sum()),
        "baseline_point_fgmiou": base_point_miou,
        "baseline_point_fgmiou_selfcheck": "OK" if abs(base_point_miou - 0.1548) < 0.005 else "MISMATCH",
        "instance_accuracy": acc,
        "instance_renaming_fgmiou": miou,
        "proto_used": args.proto_path is not None,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
