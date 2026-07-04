# Mosaic3D readout-failure: Method B training + annotation-free readout diagnosis

Artifacts for the ScanNet200 open-vocab 3D segmentation "readout failure" study.
All numbers are annotation-free (no ScanNet200 GT labels used for training).

## Headline result (b): trained Method B (caption_siglip_hardneg) does NOT improve
Same checkpoint (epoch 7, plateaued), fixed-koujing fg-mIoU (== native val) + mAP:

| model / readout | fg-mIoU | d vs baseline 0.15475 | d vs base+MTV 0.17573 | mAP |
|---|---|---|---|---|
| baseline (reproduced) | 0.15475 | - | - | 0.11486 |
| baseline + mask_text_vote (Method A) | 0.17573 | +0.02098 | - | 0.11236 |
| B, baseline readout | 0.1380 | -0.0168 | -0.0377 | 0.1111 |
| B + mask_text_vote | 0.1598 | +0.0050 | -0.0159 | 0.1084 |
| B + anchor_decorrelate | 0.1549 | +0.0001 | -0.0208 | 0.1061 |

- Pre-registered (1) B or B+MTV > 0.17573: NOT met (max 0.1598).
- Pre-registered (2) cos(d_vis,d_txt) rises from ~0.03: REFUTED (0.031 vs 0.034).
- CLIP text encoder is frozen (B text anchors == baseline, diff 3e-7), so a frozen-text
  hard-negative loss structurally cannot move the readout direction.
- Init verified healthy: `missing=1/unexpected=588` is the frozen CLIP by design;
  the 3D backbone loaded from qz correctly.

The only validated annotation-free gain remains **Method A `mask_text_vote` on the
reproduced baseline: fg-mIoU 0.15475 -> 0.17573 (+0.02098)**, readout-only, no training.

## Layout
- `error_analysis/reports/` — all reports. Start with `DELIVERY.md`, then `method_b.md`,
  `init_load_diagnosis.md`, `readout_inference_diag.md`, `axes_summary.md`.
- `error_analysis/reports/method_b_autoeval/` — the three-tier eval logs + oracle self-checks + cos mechanism.
- `scripts/` — new annotation-free analysis/diagnosis scripts.
- `src/models/lightning_modules/language_module.py` — patched (adds `anchor_decorrelate` readout mode).
- `src/eval.py` — patched (`torch.load(weights_only=False)` for torch>=2.6 checkpoints).
- `src/utils/readout_decorrelate.py` — `whiten_pick` within-cluster anchor decorrelation.
- `container_commits.patch` — full `git format-patch` of the three working commits.

## Next-direction note
Any future Method-B attempt must also unfreeze/adapt the text side (or use a different
objective); prioritize annotation-free instance-level readout/calibration/reranking over
frozen-text hard-negative training.
