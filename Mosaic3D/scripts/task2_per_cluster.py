"""Per-cluster instance-level breakdown for TASK2 (fast, no fg-mIoU repaint)."""
import argparse, json
from pathlib import Path
import numpy as np
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.readout_decorrelate import whiten_pick


def _dec(v):
    return v.item() if isinstance(v, np.ndarray) and v.shape == () else v


def load(p):
    d = np.load(p, allow_pickle=True)
    return {k: _dec(d[k]) for k in d.files}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature-dir", type=Path, required=True)
    ap.add_argument("--text-embeddings", type=Path, required=True)
    ap.add_argument("--clusters", type=Path, required=True)
    ap.add_argument("--out-md", type=Path, required=True)
    ap.add_argument("--max-cluster-size", type=int, default=8)
    args = ap.parse_args()

    te = load(args.text_embeddings)
    emb = np.asarray(te["emb"], dtype=np.float64)
    T = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    names = [str(c) for c in np.asarray(te["class_names"])]
    C = len(names)
    fg_idx = np.asarray(te["fg_class_idx"]).reshape(-1).astype(int)
    fg_mask = np.zeros(C, bool); fg_mask[fg_idx] = True

    cl = load(args.clusters)
    cid = np.asarray(cl["cluster_id"]).astype(int)
    thr = float(cl["threshold"])
    raw = {}
    for c in range(C):
        if cid[c] >= 0:
            raw.setdefault(cid[c], []).append(c)
    clusters = {g: m for g, m in raw.items() if 2 <= len(m) <= args.max_cluster_size}

    feats, ytrue = [], []
    for p in sorted(args.feature_dir.glob("*.npz")):
        d = load(p)
        feats.append(np.asarray(d["pooled_feat"], dtype=np.float64))
        ytrue.append(np.asarray(d["true_class"], dtype=np.int64))
    X = np.concatenate(feats, 0); X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    ytrue = np.concatenate(ytrue, 0)
    sim = X @ T.T; sim[:, ~fg_mask] = -np.inf
    base_top = np.argmax(sim, 1)

    rows = []
    for g, m in clusters.items():
        sel = np.flatnonzero(np.isin(base_top, m) | np.isin(ytrue, m))
        if len(sel) == 0:
            continue
        yt = ytrue[sel]
        in_cluster = np.isin(yt, m)
        base_acc = float(np.mean(base_top[sel] == yt))
        # decorrelate
        dec = base_top[sel].copy()
        for k, i in enumerate(sel):
            if int(base_top[i]) in m:
                j = whiten_pick(X[i], T[m]); dec[k] = m[j]
        dec_acc = float(np.mean(dec == yt))
        oracle = base_top[sel].copy()
        for k, i in enumerate(sel):
            if yt[k] in m:
                oracle[k] = yt[k]
        orc_acc = float(np.mean(oracle == yt))
        rows.append((", ".join(names[c] for c in m), len(sel), int(in_cluster.sum()),
                     base_acc, dec_acc, orc_acc))
    rows.sort(key=lambda r: r[1], reverse=True)

    lines = [f"### Per-cluster instance accuracy (thr={thr:.2f}, cap={args.max_cluster_size})", "",
             "| cluster members | n_inst | n_true_in | base_acc | decorr_acc | oracle_acc |",
             "|---|---:|---:|---:|---:|---:|"]
    for nm, n, nin, ba, da, oa in rows:
        lines.append(f"| {nm} | {n} | {nin} | {ba:.3f} | {da:.3f} | {oa:.3f} |")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
