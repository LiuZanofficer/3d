"""TASK2 - build annotation-free sibling clusters from class-name text embeddings.

Uses only text embeddings of the 200 ScanNet200 class names (no GT labels).
Connected components over text cosine >= threshold, foreground classes only.

Outputs one npz per threshold with the cluster assignment plus a markdown
summary of multi-member clusters. Consumed by anchor_decorrelate readout and
the oracle-vs-unsupervised ablation.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.class_term_utils import build_text_clusters  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text-embeddings", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--thresholds", type=str, default="0.85,0.88,0.90")
    args = ap.parse_args()

    te = np.load(args.text_embeddings, allow_pickle=True)
    emb = np.asarray(te["emb"], dtype=np.float64)
    class_names = [str(c) for c in np.asarray(te["class_names"])]
    fg_idx = np.asarray(te["fg_class_idx"]).reshape(-1).astype(int)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = ["# TASK2 sibling clusters (annotation-free, text cosine CC)\n"]
    lines.append(f"- source: {args.text_embeddings}")
    lines.append(f"- foreground classes: {len(fg_idx)}\n")

    summary = {}
    for thr in [float(t) for t in args.thresholds.split(",")]:
        clusters = build_text_clusters(emb, fg_idx, thr)
        multi = [c for c in clusters if len(c) >= 2]
        n_multi_classes = sum(len(c) for c in multi)
        summary[f"{thr:.2f}"] = {
            "n_clusters": len(clusters),
            "n_multi_clusters": len(multi),
            "n_classes_in_multi": n_multi_classes,
        }
        cid = np.full(len(class_names), -1, dtype=np.int64)
        for gi, cl in enumerate(clusters):
            for c in cl:
                cid[c] = gi
        np.savez(
            args.out_dir / f"clusters_thr{thr:.2f}.npz",
            cluster_id=cid,
            fg_class_idx=fg_idx,
            class_names=np.asarray(class_names, dtype=object),
            threshold=np.float64(thr),
        )

        lines.append(f"## threshold = {thr:.2f}")
        lines.append(
            f"- clusters={len(clusters)}, multi-member clusters={len(multi)}, "
            f"classes covered by multi-member clusters={n_multi_classes}"
        )
        lines.append("")
        lines.append("| cluster | size | members |")
        lines.append("|---|---:|---|")
        for cl in sorted(multi, key=len, reverse=True):
            names = ", ".join(class_names[c] for c in cl)
            lines.append(f"| {cl[0]} | {len(cl)} | {names} |")
        lines.append("")

    (args.out_dir / "sibling_clusters.md").write_text("\n".join(lines))
    (args.out_dir / "sibling_clusters_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print("outputs ->", args.out_dir / "sibling_clusters.md")


if __name__ == "__main__":
    main()
