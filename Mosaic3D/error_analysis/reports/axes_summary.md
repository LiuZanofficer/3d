# TASK3 - Axes accounting (ScanNet200, annotation-free)

## Mask provenance (Segment3D, not GT)
`batch["masks_binary"]` / dumped `pred_masks` are **Segment3D** class-agnostic proposals
(`/.../scannet_masks/segment3d/point_indices.npz`, fields `packed`/`lengths`/`scores`), NOT GT instances:
- mask count != GT-instance count in **302/312** val scenes (13165 proposal masks vs 10044 GT instances);
- each mask carries a class-agnostic `pred_mask_scores` (float32, e.g. [0.008, 0.66]), no GT labels.
So mask-based instance consistency is annotation-free.

## What each axis is
- **baseline**: per-point foreground argmax of `feat·text` → point fg-mIoU **0.15475** (self-checked口径).
- **+mask_text_vote (Method A)**: pool per-point prob within each Segment3D mask, argmax, paint mask-consistently.
  This is an **instance-consistency** axis (denoising scattered point noise), NOT a direction fix.
- **+direction-fix (inference)**: annotation-free re-orientation of the text readout at eval time
  (anchor_decorrelate / caption_proto / unsup text-naming). TASK2 result: **no gain** (all <= baseline).
- **+direction-fix (training = Method B)**: hard-negative caption loss retrains the visual<->text alignment. Pending Phase-4 eval.

## Combination-axis table (fg-mIoU; mAP)
| axis | fg-mIoU | Δ vs baseline | mAP | source |
|---|---:|---:|---:|---|
| baseline | 0.15475 | — | 0.11486 | live eval + offline self-check (OK) |
| + mask_text_vote (consistency) | 0.17573 | +0.02098 | 0.11236 (-0.00250) | prior live eval (established) |
| + direction-fix (inference) | ~0.15475 | +0.000 | — | TASK2 refuted (best annotation-free = baseline) |
| + mask_text_vote + direction-fix (inference) | ~0.17573 | +0.02098 | — | inference fix adds nothing on top of consistency |
| + direction-fix (training, Method B) | TBD | TBD | ~0.115 target | Phase-4 eval (B ckpt) |
| + mask_text_vote + Method B | TBD | TBD | TBD | Phase-4 eval |

## Gain decomposition
- The only realized gain so far is **instance-consistency** (+0.02098 from mask_text_vote), which is orthogonal
  to naming: it removes scattered per-point noise but keeps each mask's (possibly wrong) sibling name.
- **Inference-time direction-fix contributes 0** (TASK2): the text anchors are collinear and the separable
  structure cannot be named without labels.
- Remaining head-room is bounded by naming ceilings, all annotation-free-unreachable:
  Task0 real-labelspace oracle top25=0.1906 / top50=0.2118 / all=0.3257; within-tight-sibling naming ceiling
  (oracle_incluster) instance-renaming fg-mIoU=0.2311. Method B must convert part of this naming head-room.

## Reading
mask_text_vote is a baseline/consistency axis and must not be double-counted as a "fix". The scientific
question for Method B is whether *trained* direction correction adds fg-mIoU **on top of** the 0.17573
consistency axis, and whether the gain concentrates on the sibling pairs identified in TASK2.
