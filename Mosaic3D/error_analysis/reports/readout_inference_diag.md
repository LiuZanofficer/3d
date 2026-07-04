# TASK2 - Readout-inference diagnosis (annotation-free, ScanNet200)

**One-line verdict:** the sibling-confusion errors are a *naming* dead-knot, not a
*separation* problem. Instance features are separable, but **no** annotation-free
inference-time readout (anchor decorrelation, caption prototypes, unsupervised
2-means + text naming) beats the baseline instance readout; only oracle (GT)
naming recovers the room. This motivates training (Method B), not an inference patch.

## Setup and iron-rule compliance
- Data: existing baseline eval dump `logs/eval/runs/2026-06-27_06-58-17`
  (`eval_scene_dumps` per-point, `eval_instance_features` per-GT-instance 768-d pooled feats).
- Caption source confirmed = **gsam2/seem** (`src/data/dataset_base.py` default `anno_sources=["gsam2","seem"]`; training config did not override).
- Scoring口径 fixed and self-checked: argmax=`pred_semantic`, bg->ignore, `valid=(gt>=0)&(gt<C)&(pred valid)`,
  fg mean over `fg_class_idx`. **baseline point fg-mIoU = 0.15475 (self-check OK)**.
- Decision inputs for every *method* are visual features + class-name text anchors only. GT is used
  ONLY for scoring and for explicitly-labelled oracle ceilings.
- **Important scope note (per-point features are not dumped):** a deployable point-level fg-mIoU for a
  new text-anchor readout requires the GPU eval (Phase 4). Offline we quantify the *naming ceiling* via
  an instance-renaming fg-mIoU that repaints whole GT instances (GT grouping) — this **upper-bounds** a
  real Segment3D-mask readout and is a diagnostic, not a method. Instance accuracy is the primary metric.

## Sibling clusters (annotation-free, text-cosine connected components)
Class-name text embeddings are highly collinear, so a global threshold collapses into one blob:

| threshold | #clusters | #multi-member | classes in multi | note |
|---|---:|---:|---:|---|
| 0.85 | 2 | 1 | 196 | one 196-blob (unusable) |
| 0.88 | 29 | 5 | 173 | one 163-blob + 4 tight groups |
| 0.90 | 70 | 18 | 145 | one 101-blob + 17 tight groups |

Within-cluster methods therefore only act on **tight** clusters (`2<=size<=8`); the blob is left at baseline.
Primary analysis uses threshold 0.90, cap 8 (17 tight sibling clusters, 1099 sibling instances).

## Mode comparison (threshold 0.90, cap 8)
Instance-level readout over 7769 fg instances; renaming fg-mIoU = instance-renaming ceiling.

| mode | acc_all | acc_sibling | inst-renaming fg-mIoU | verdict |
|---|---:|---:|---:|---|
| baseline_inst (argmax feat·text) | 0.452 | 0.379 | 0.1999 | reference |
| anchor_decorrelate | 0.446 | 0.336 | 0.1889 | **hurts** |
| caption_proto (300b protos) | 0.451 | 0.375 | 0.1944 | no help (≈ prior -0.006) |
| unsup_kmeans_textname | 0.452 | 0.377 | 0.1886 | no help |
| unsup_kmeans_oracle (diag) | 0.512 | 0.335 | 0.1950 | separation works, naming leaks |
| oracle_incluster (ceiling) | 0.465 | 0.470 | **0.2311** | naming ceiling |

Threshold 0.88, cap 8 (only 4 tight clusters, 124 sibling instances) shows the same ordering with tiny magnitude
(baseline renaming 0.1999, oracle ceiling 0.2084, decorrelate 0.1967, textname 0.2004).

### Separate vs name (uncapped, over the full blob)
When within-cluster 2-means is allowed to run over the whole collinear blob:
**unsupervised 2-means + oracle naming reaches instance acc 0.620 and oracle_incluster 0.732** (vs baseline 0.452),
while **text-based naming of the same groups stays <= baseline (0.428)**. Separation is easy; naming is the wall.

## Per-cluster breakdown (thr 0.90, cap 8) — the naming gap
| cluster | n_inst | base_acc | decorr_acc | oracle_acc |
|---|---:|---:|---:|---:|
| shelf, bookshelf | 694 | 0.169 | 0.169 | 0.334 |
| pillow, cushion | 220 | 0.055 | 0.055 | **0.973** |
| board, sign, poster | 163 | 0.037 | 0.037 | 0.227 |
| shower, shower wall/door/floor/head | 154 | 0.227 | 0.065 | 0.305 |
| towel, blanket | 137 | 0.321 | 0.321 | 0.832 |
| curtain, shower curtain, shower curtain rod | 134 | 0.567 | 0.410 | 0.761 |
| refrigerator, mini fridge | 114 | 0.254 | 0.254 | 0.482 |
| bottle, water bottle, case of water bottles | 93 | 0.011 | 0.011 | 0.505 |
| plate, cup, tray, bowl | 60 | 0.017 | 0.017 | **0.967** |
| plant, potted plant | 60 | 0.600 | 0.600 | 0.833 |
| clock, alarm clock | 24 | 0.333 | 0.333 | 0.917 |
| light switch, power outlet, power strip | 18 | 0.000 | 0.000 | **1.000** |
| column, pillar | 11 | 0.364 | 0.364 | 1.000 |

Baseline text naming lands on the wrong sibling; oracle naming is near-perfect for many groups
(pillow/cushion, plate/cup/bowl, light switch/outlet). Decorrelation is at best a no-op and sometimes
actively worse (shower, curtain, coffee-maker) because whitening near-collinear anchors amplifies noise.

## PRE-REGISTERED criteria — verdicts
- **Criterion ① (oracle near ceiling but unsupervised fails -> naming is the dead-knot, must train): CONFIRMED.**
  oracle_incluster fg-mIoU 0.2311 and unsup+oracle acc 0.62/0.73 sit far above every annotation-free
  naming method (all <= baseline). The gap is naming, not separability.
- **Criterion ② (an annotation-free method beats 0.17573 with per-class gains concentrated on sibling pairs): NOT met.**
  Best annotation-free renaming fg-mIoU = 0.1999 (= baseline instance readout, from instance-consistency alone,
  already the mask_text_vote axis); decorrelate/caption_proto/textname are all below it.

## Implication
Inference-only, annotation-free readout cannot exploit the (highly separable) instance features because the
text anchors are collinear and mis-oriented. The fix must change the visual<->text alignment during training
-> Method B (hard-negative caption loss). Code: `anchor_decorrelate` read-only mode added to
`_readout_sample` (`src/utils/readout_decorrelate.py`) for optional live confirmation, but it is a negative
result and is not recommended.
