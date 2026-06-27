"""V1b - winner-take-all direction test (Phase 2).

The decision is argmax_j  sim(visual_feat, text[j]).  Mechanism question:
for a victim class C whose points mostly leak to absorber J, is the absorber's
text embedding ACTUALLY closer to C's average visual feature than C's own text?
If yes systematically, the encoder's features for C are decoded into J's text
region -> a text-decision / winner-take-all failure (not "points lost").

Uses per-class mean of the L2-normalized visual feature (the exact vector used
for the argmax, dumped by language_module._dump_visual_means) and the model's
own text classifier (emb_target).

For each absorber pair (C -> top-1 FN destination J), computed from this run's
confusion matrix:
    sim_self = <vis_mean[C], text[C]>
    sim_abs  = <vis_mean[C], text[J]>
    winner   = sim_abs > sim_self
  r_winner = fraction of absorber pairs (top-K cost classes) with winner True.

PRE-REGISTERED VERDICT (decided before running; do NOT change after seeing data):
  CONFIRM (winner-take-all direction): r_winner >= 0.70
  REFUTE: r_winner <= 0.50 (no better than a coin flip)
  INCONCLUSIVE: 0.50 < r_winner < 0.70
Random control (context, not part of the rule): r_winner_rand over random J'.

Read-only; numpy only.
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
    ap = argparse.ArgumentParser(description="V1b: winner-take-all direction test.")
    ap.add_argument("--visual-means", type=Path, required=True, help="eval_visual_means/<postfix>.npz")
    ap.add_argument("--dump-dir", type=Path, required=True, help="matching eval_scene_dumps/<postfix>")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--topk-cost", type=int, default=25)
    ap.add_argument("--rand-trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    vm = np.load(args.visual_means, allow_pickle=True)
    class_names = [str(c) for c in np.asarray(vm["class_names"])]
    num_classes = len(class_names)
    vis_sum = np.asarray(vm["vis_sum"], dtype=np.float64)
    vis_cnt = np.asarray(vm["vis_cnt"], dtype=np.float64)
    emb = np.asarray(vm["emb_target"], dtype=np.float64)
    if emb.shape[0] != num_classes:
        raise SystemExit("emb_target missing/mismatched in visual-means npz.")
    fg_idx = np.asarray(vm["fg_class_idx"]).reshape(-1).astype(int)
    bg_idx = np.asarray(vm.get("bg_class_idx", np.array([]))).reshape(-1).astype(int)

    # mean visual direction (renormalized); emb_target already unit-norm
    with np.errstate(divide="ignore", invalid="ignore"):
        vis_mean = vis_sum / np.maximum(vis_cnt[:, None], 1.0)
    norms = np.linalg.norm(vis_mean, axis=1, keepdims=True)
    vis_mean = np.where(norms > 0, vis_mean / np.maximum(norms, 1e-9), 0.0)
    emb = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9)

    # absorber pairs from this run's confusion
    cm = build_global_confmat(sorted(args.dump_dir.glob("*.npz")), num_classes,
                              bg_idx, int(vm.get("ignore_label", -100)) if "ignore_label" in vm else -100)
    gt_count = cm.sum(axis=1).astype(np.float64)
    tp = np.diag(cm).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(gt_count > 0, tp / gt_count, 0.0)
    cost = gt_count * (1.0 - recall)

    absorber = {}
    for c in fg_idx:
        row = cm[c].astype(np.float64).copy()
        row[c] = 0
        if row.sum() <= 0 or vis_cnt[c] <= 0:
            continue
        absorber[int(c)] = int(np.argmax(row))

    fg_by_cost = sorted([c for c in absorber], key=lambda c: cost[c], reverse=True)
    topk = fg_by_cost[: args.topk_cost]

    rows = []
    wins = 0
    margins = []
    for c in topk:
        j = absorber[c]
        s_self = float(vis_mean[c] @ emb[c])
        s_abs = float(vis_mean[c] @ emb[j])
        win = s_abs > s_self
        wins += int(win)
        margins.append(s_abs - s_self)
        rows.append((class_names[c], class_names[j], s_self, s_abs, win, int(cost[c])))
    r_winner = wins / len(topk) if topk else float("nan")
    mean_margin = float(np.mean(margins)) if margins else float("nan")

    # random control: for each victim C in topk, sample random fg J' (!= C), how
    # often does a random class beat self?
    rng = np.random.default_rng(args.seed)
    fg_arr = np.asarray(fg_idx)
    rwins = 0
    rtot = 0
    for c in topk:
        s_self = float(vis_mean[c] @ emb[c])
        for _ in range(max(1, args.rand_trials // max(1, len(topk)))):
            jp = int(rng.choice(fg_arr))
            if jp == c:
                continue
            rtot += 1
            if float(vis_mean[c] @ emb[jp]) > s_self:
                rwins += 1
    r_winner_rand = rwins / rtot if rtot else float("nan")

    if r_winner >= 0.70:
        verdict = "CONFIRM (winner-take-all direction)"
    elif r_winner <= 0.50:
        verdict = "REFUTE (no better than coin flip)"
    else:
        verdict = "INCONCLUSIVE"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_dir / "winner_take_all.md", "w") as fh:
        fh.write("# V1b winner-take-all direction\n\n")
        fh.write(f"- absorber pairs from top-{args.topk_cost} cost fg classes (n={len(topk)})\n")
        fh.write(f"- **r_winner = {r_winner:.3f}** (absorber text closer to victim visual mean than victim's own text)\n")
        fh.write(f"- mean margin (sim_abs - sim_self) = {mean_margin:+.4f}\n")
        fh.write(f"- random-class control r_winner_rand = {r_winner_rand:.3f}\n\n")
        fh.write("## PRE-REGISTERED verdict\n")
        fh.write("- CONFIRM if r_winner>=0.70; REFUTE if r_winner<=0.50; else INCONCLUSIVE\n")
        fh.write(f"- **{verdict}**\n\n")
        fh.write("## Per-pair (C -> absorber J)\n\n")
        fh.write("| C | J | sim_self | sim_abs | absorber wins | cost |\n|---|---|---|---|---|---|\n")
        for nc, nj, ss, sa, w, co in rows:
            fh.write(f"| {nc} | {nj} | {ss:.4f} | {sa:.4f} | {'YES' if w else 'no'} | {co} |\n")

    print("=" * 64)
    print(f"r_winner (top{args.topk_cost}) = {r_winner:.3f}  mean_margin={mean_margin:+.4f}")
    print(f"random-class control r_winner_rand = {r_winner_rand:.3f}")
    print(f"PRE-REGISTERED VERDICT: {verdict}")
    print(f"outputs -> {args.out_dir}/winner_take_all.md")
    print("=" * 64)


if __name__ == "__main__":
    main()
