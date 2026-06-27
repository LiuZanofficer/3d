"""V1a - test whether sibling/absorber class text embeddings are near-collinear.

Reproduces the EXACT classifier the model uses at eval time:
  text_emb = normalize(CLIP.encode_text("a {c} in a scene"))   (use_prompt=True)
(see language_module.on_validation_epoch_start lines ~269-277 and
 caption_utils.forward_text_encoder).

It then:
  - derives "absorber pairs" data-drivenly: for each foreground class C, its
    top-1 FN destination J (where most of C's missed points were predicted),
    computed from the same confusion matrix used for f-mIoU;
  - compares cosine(text[C], text[J]) for absorber pairs vs a random
    non-sibling baseline distribution;
  - exports connected components at cosine>threshold as a class->group map for
    V4 (analyze_merge_superclass.py --grouping text).

PRE-REGISTERED VERDICT (decided before running; do NOT change after seeing data):
  cos_abs = mean cosine over the top-(topk-cost) absorber pairs.
  CONFIRM (text near-collinear): cos_abs >= p95_rand AND cos_abs >= mu_rand + 0.10
  REFUTE  (text not the mechanism): cos_abs <= mu_rand + 0.02
  INCONCLUSIVE: otherwise (escalate to V1b visual-direction test)

Needs numpy + torch + the project's CLIP builder (run inside the docker container).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# allow importing sibling script + project src when run as `python scripts/...`
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


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


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main():
    ap = argparse.ArgumentParser(description="V1a: sibling/absorber text-embedding collinearity test.")
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--model-id", type=str, default="hf-hub:UCSC-VLAA/ViT-L-16-HTxt-Recap-CLIP")
    ap.add_argument("--use-prompt", action="store_true", default=True)
    ap.add_argument("--no-prompt", dest="use_prompt", action="store_false")
    ap.add_argument("--topk-cost", type=int, default=25, help="absorber pairs from the top-K cost fg classes")
    ap.add_argument("--cos-threshold", type=float, default=0.90, help="edge threshold for text grouping export")
    ap.add_argument("--rand-pairs", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default=None)
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
    is_fg = np.zeros(num_classes, dtype=bool)
    is_fg[fg_idx] = True

    cm = build_global_confmat(files, num_classes, bg_idx, ignore_label)
    gt_count = cm.sum(axis=1).astype(np.float64)
    tp = np.diag(cm).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(gt_count > 0, tp / gt_count, 0.0)
    cost = gt_count * (1.0 - recall)

    # absorber pair = (C, top-1 off-diagonal FN destination of C)
    absorber_pairs = {}  # c -> (j, share)
    for c in fg_idx:
        row = cm[c].astype(np.float64).copy()
        tp_c = row[c]
        row[c] = 0
        total_fn = row.sum()
        if total_fn <= 0:
            continue
        j = int(np.argmax(row))
        absorber_pairs[int(c)] = (j, float(row[j] / total_fn))

    fg_by_cost = [int(c) for c in fg_idx]
    fg_by_cost.sort(key=lambda c: cost[c], reverse=True)
    topk_classes = [c for c in fg_by_cost if c in absorber_pairs][: args.topk_cost]

    # ---- build CLIP and encode class names with the model's template ----
    import torch
    from src.models.utils.clip_models import build_clip_model
    from src.utils import caption_utils

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = {"model_id": args.model_id}
    clip_encoder = build_clip_model(model_cfg, device=device).to(device).eval()

    if args.use_prompt:
        prompts = [f"a {c} in a scene" if "other" not in c else "other" for c in class_names]
    else:
        prompts = class_names
    with torch.no_grad():
        emb = caption_utils.forward_text_encoder(prompts, clip_encoder, normalize=True, device=device)
    emb = emb.float().cpu().numpy()  # [N, D], unit-norm
    cos = emb @ emb.T  # cosine since normalized

    # absorber-pair cosines
    rows = []
    for c in topk_classes:
        j, share = absorber_pairs[c]
        rows.append((class_names[c], class_names[j], float(cos[c, j]), share, int(cost[c])))
    cos_abs_topk = float(np.mean([r[2] for r in rows])) if rows else float("nan")
    # all-fg absorber pairs (broader)
    all_abs_cos = [float(cos[c, j]) for c, (j, _) in absorber_pairs.items()]
    cos_abs_all = float(np.mean(all_abs_cos)) if all_abs_cos else float("nan")

    # ---- random non-sibling baseline ----
    absorber_set = set()
    for c, (j, _) in absorber_pairs.items():
        absorber_set.add((c, j))
        absorber_set.add((j, c))
    rng = np.random.default_rng(args.seed)
    fg_arr = np.asarray(fg_idx)
    rand_cos = []
    tries = 0
    while len(rand_cos) < args.rand_pairs and tries < args.rand_pairs * 20:
        a, b = rng.choice(fg_arr, size=2, replace=False)
        tries += 1
        if (int(a), int(b)) in absorber_set:
            continue
        rand_cos.append(float(cos[a, b]))
    rand_cos = np.asarray(rand_cos)
    mu_rand = float(rand_cos.mean())
    sd_rand = float(rand_cos.std())
    p95_rand = float(np.percentile(rand_cos, 95))

    # ---- PRE-REGISTERED verdict ----
    if cos_abs_topk >= p95_rand and cos_abs_topk >= mu_rand + 0.10:
        verdict = "CONFIRM (text near-collinear)"
    elif cos_abs_topk <= mu_rand + 0.02:
        verdict = "REFUTE (text not the mechanism)"
    else:
        verdict = "INCONCLUSIVE (escalate to V1b)"

    # ---- text grouping export (connected components at cosine>threshold, fg only) ----
    uf = UnionFind(num_classes)
    for ii in range(len(fg_idx)):
        for jj in range(ii + 1, len(fg_idx)):
            a, b = int(fg_idx[ii]), int(fg_idx[jj])
            if cos[a, b] > args.cos_threshold:
                uf.union(a, b)
    group_of = {}
    for c in fg_idx:
        group_of[int(c)] = uf.find(int(c))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "text_embeddings.npz",
        class_names=np.array(class_names, dtype=object),
        emb=emb, cos=cos.astype(np.float32),
        fg_class_idx=fg_idx,
    )
    with open(args.out_dir / "text_groups.json", "w") as fh:
        json.dump({str(k): int(v) for k, v in group_of.items()},
                  fh, ensure_ascii=False, indent=2)

    with open(args.out_dir / "text_collinearity.md", "w") as fh:
        fh.write("# V1a text-embedding collinearity\n\n")
        fh.write(f"- model: `{args.model_id}`  use_prompt={args.use_prompt}\n")
        fh.write(f"- absorber pairs (top-{args.topk_cost} cost): mean cosine **{cos_abs_topk:.4f}**\n")
        fh.write(f"- absorber pairs (all fg, n={len(all_abs_cos)}): mean cosine **{cos_abs_all:.4f}**\n")
        fh.write(f"- random non-sibling baseline: mu={mu_rand:.4f} sd={sd_rand:.4f} p95={p95_rand:.4f}\n\n")
        fh.write("## PRE-REGISTERED verdict\n")
        fh.write(f"- rule: CONFIRM if cos_abs>=p95_rand({p95_rand:.4f}) AND >=mu+0.10({mu_rand + 0.10:.4f}); "
                 f"REFUTE if cos_abs<=mu+0.02({mu_rand + 0.02:.4f})\n")
        fh.write(f"- **cos_abs(top{args.topk_cost})={cos_abs_topk:.4f} -> {verdict}**\n\n")
        fh.write("## Absorber pairs (C -> top-1 FN destination J)\n\n")
        fh.write("| C | J | cosine | FN-share | cost |\n|---|---|---|---|---|\n")
        for nc, nj, cv, sh, co in rows:
            fh.write(f"| {nc} | {nj} | {cv:.4f} | {sh:.1%} | {co} |\n")

    print("=" * 64)
    print(f"absorber cos (top{args.topk_cost})={cos_abs_topk:.4f}  all-fg={cos_abs_all:.4f}")
    print(f"random baseline: mu={mu_rand:.4f} p95={p95_rand:.4f} (sd={sd_rand:.4f})")
    print(f"PRE-REGISTERED VERDICT: {verdict}")
    print(f"text groups (cos>{args.cos_threshold}): "
          f"{len(set(group_of.values()))} groups for {len(fg_idx)} fg classes")
    print(f"outputs -> {args.out_dir}/ (text_collinearity.md, text_groups.json, text_embeddings.npz)")
    print("=" * 64)


if __name__ == "__main__":
    main()
