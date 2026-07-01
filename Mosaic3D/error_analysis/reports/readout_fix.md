# Readout-failure fix report (partial, Method A done; Method B queued)

## Pre-registered criteria
- Step 0 confirms readout-direction mismatch if most high-cost pairs have `|cos(d_vis,d_txt)| < 0.5` and the mean is not above random by `>=0.05`.
- Method A is useful only if it improves ScanNet200 fg-mIoU without materially damaging instance mAP.
- Method B must be judged after training by reporting `B` and `B+A` against the same baseline, with all metric deltas.

## Step 0: readout direction
- Baseline self-check: ScanNet200 fg-mIoU `0.1548`.
- Mean `|cos(d_vis,d_txt)| = 0.034`, median `0.029`, fraction `<0.5 = 1.000`.
- Random direction mean `|cos| = 0.028`.
- Verdict: CONFIRM. The text readout direction is effectively orthogonal to the visual discriminant direction.

Report: `error_analysis/reports/readout_direction.md`.

## Method A: readout-only controls

All values are from `src/eval.py` on ScanNet200 with `qz/sc+ar+sc++.ckpt`.

| setting | fg-mIoU | delta | mIoU head | delta | mIoU common | delta | mIoU tail | delta | mAcc | delta | mAP | delta | mAP25 | delta | mAP50 | delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.15475 | - | 0.33277 | - | 0.10035 | - | 0.02575 | - | 0.28086 | - | 0.11486 | - | 0.17793 | - | 0.15497 | - |
| mask_text_vote | 0.17573 | +0.02098 | 0.36793 | +0.03516 | 0.11980 | +0.01945 | 0.03317 | +0.00742 | 0.29655 | +0.01569 | 0.11236 | -0.00250 | 0.17382 | -0.00411 | 0.15133 | -0.00365 |
| caption_proto_300b | 0.14868 | -0.00607 | 0.32413 | -0.00864 | 0.09598 | -0.00438 | 0.02079 | -0.00496 | 0.28214 | +0.00128 | 0.09886 | -0.01599 | 0.15115 | -0.02677 | 0.13375 | -0.02123 |

Interpretation:
- `mask_text_vote` is a real readout-only semantic gain: `+2.10` fg-mIoU points, with the largest gain on head/common and a smaller but positive tail gain.
- `mask_text_vote` slightly hurts instance mAP: `-0.25` mAP points. This is acceptable as a diagnostic but not yet a clean final method.
- `caption_proto_300b` is not useful in its current form: semantic and instance metrics both drop. The 300-batch prototype source is bounded and annotation-free, but the prototype readout is too noisy to keep.

## Method B status
- Implemented `HardNegativeCaptionLoss`, driven only by caption text, class names, aliases, and text-embedding clusters. It does not use ScanNet200 GT labels.
- Added config `configs/model/loss/caption_siglip_hardneg.yaml`.
- Added `init_ckpt_path` to `src/train.py` so the reproduced raw state_dict checkpoint can initialize training without Lightning-resume metadata.
- 1-GPU 2-step smoke passed with `hard_negative_caption_loss` logged (`~0.003` in the smoke run).
- 8-GPU smoke is queued until all 8 GPUs have `>20GB` free. Current blocker: shared GPUs are occupied.

## Current decision
- Method A confirms the core readout/aggregation opportunity, but the best simple variant is `mask_text_vote`, not caption prototypes.
- Do not present `caption_proto_300b` as an improvement.
- The next decisive number is trained `B` and `B+A`; until that runs, the best verified improvement is:
  - fg-mIoU: `0.15475 -> 0.17573`, delta `+0.02098`.
  - mAP: `0.11486 -> 0.11236`, delta `-0.00250`.
