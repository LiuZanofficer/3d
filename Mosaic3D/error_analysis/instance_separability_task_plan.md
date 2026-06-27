# Instance Separability Task Plan

> Purpose: lock down the next-stage experiments before implementation. Do not run a task unless its criteria, controls, inputs, and outputs are written here or in the task report first.

## Non-Negotiable Setup

- Setting: annotation-free Mosaic3D. Training uses captions/CLIP alignment only; never use ScanNet200 GT labels for training.
- Do not compare this line directly with fully supervised methods.
- Scored prediction: `pred_semantic` only, i.e. foreground-masked argmax.
- GT handling: map background classes `wall/floor/ceiling` to ignore; drop `ignore_label`; also drop invalid labels `(gt < 0) | (gt >= num_classes)`.
- Metric: `IoU_c = tp / (tp + fp + fn)` and `fg-mIoU = mean(IoU over fg_class_idx)`.
- Every new script must self-check baseline fg-mIoU near `0.1548`. If not, fix scoring before interpreting results.
- All conclusions must follow pre-registered criteria. Do not change thresholds after seeing results.

## Verified Current State

- Real baseline: ScanNet200 fg-mIoU ≈ `0.1548`; instance mAP ≈ `0.115`.
- Latest usable dump root: `Mosaic3D/logs/eval/runs/2026-06-26_10-17-27/eval_scene_dumps/scannet200/`.
- A sampled dump contains: `scene_name`, `class_names`, `gt_segment`, `gt_instance`, `pred_semantic`, `pred_point_classes`, `pred_mask_classes`, `pred_mask_scores`, `pred_masks`, `fg_class_idx`, `bg_class_idx`, `instance_ignore_class_idx`, `ignore_label`.
- `Mosaic3D/src/models/lightning_modules/language_module.py` currently dumps `pred_semantic` and per-class visual means, but not per-instance pooled features.
- `anno_sources` risk: `Mosaic3D/src/data/dataset_base.py` defaults to `["gsam2", "seem"]`; `configs/data/sc_segment3d_gathered.yaml` explicitly uses `["segment3d-gathered"]`. Task 2 must confirm the actual training configuration before analysis.

## Already Established Premise

The supported error mechanism is mask/instance-level sibling appearance confusion:
some large victim-class instances are confidently flipped as their top-1 sibling class.

Do not re-litigate these conclusions:

- V1b refuted text winner-take-all (`r_winner=0.32`).
- S1 refuted boundary-driven errors (`diff=+0.0066`).
- V2 showed part-whole geometry is weak, not the main cause.
- V3 showed caption coverage is not a clean dead-class explanation.
- S3 confirmed instance-level wholesale flips (`mean leak_in_flipped=0.697`).
- V4 confusion pairing was recoverable (`z=7.9`), but this is partly oracle and simplified-label-space evidence.

Important interpretation: V4's meaningful recoverable signal is roughly `0.2456 - 0.1949 = +0.0507`, not a guaranteed `+0.09` in the real 200-class space. The true ceiling is unknown until Task 0.

## Execution Order

1. Task 0: real 200-class oracle ceiling.
2. Task 1: per-instance visual feature separability.
3. Task 2: training caption granularity.

## Task 0: Real 200-Class Oracle Ceiling

### Goal

Estimate the best possible fg-mIoU if all S3-detected wholesale-flipped victim instances were corrected in the original 200-class label space.

### Method

1. Reuse S3 logic:
   - build full-val confusion matrix;
   - for each victim class `C`, find its top-1 absorber `J`;
   - select victim classes by cost, e.g. top-50 or all classes with valid `C -> J`.
2. Iterate all eval dumps.
3. For each GT instance of class `C`, compute `frac_J = mean(pred_semantic == J)`.
4. If `frac_J >= 0.5`, oracle-correct all points of that GT instance to `C`.
5. Recompute fg-mIoU with the exact scoring convention above.

### Pre-Registered Criteria

- `oracle_fgmiou <= 0.18`: warn that this problem's method ceiling is low.
- `oracle_fgmiou >= 0.21`: there is physical room to target 20+ fg-mIoU.
- Otherwise: continue, but treat expected improvement as modest.

### Required Controls And Self-Checks

- Report baseline fg-mIoU from the same script and require it to be near `0.1548`.
- Report results for top-k settings, at least top-25, top-50, and all valid pairs.
- Report per-class contribution so a few giant classes cannot hide the distribution.

### Output

- `Mosaic3D/error_analysis/reports/oracle_real_labelspace.md`

## Task 1: Instance Visual Feature Separability

### Goal

Determine whether flipped instances and correct instances are separable in the visual feature space used for CLIP-text classification.

### Step A: Dump Per-Instance Features

Modify `language_module.py` during eval dump only:

- Pass `out_dict["clip_feat"]` through the per-sample dump path using `batch["offset"]`.
- Do not store per-point features.
- For each GT instance, store:
  - `scene`
  - `gt_instance_id`
  - `true_class`
  - `n_points`
  - `pooled_feat = L2norm(mean(L2norm(clip_feat)))`
  - `pred_majority_class`
  - `frac_J`
- Skip invalid instance ids and ignore labels.
- Run eval with `MOSAIC3D_DUMP_EVAL=1`.

### Step B: Probe Separability

For each S3 pair `(C -> J)`:

- b1: train a k-fold logistic probe to separate flipped `C` instances from correct `C` instances.
- b2: train a k-fold logistic probe to separate true `C` instances from true `J` instances.
- Include random-label AUC control; it should be near 0.5.
- Aggregate by pair and optionally by coarse sibling family.

### Pre-Registered Criteria

- `AUC >= 0.80`: separable; points to decision/readout/aggregation or calibration.
- `AUC <= 0.60`: not separable; points to representation collapse or feature insufficiency.
- `0.60 < AUC < 0.80`: inconclusive.

### Required Controls And Self-Checks

- Random-label AUC near 0.5.
- Report instance counts per class/pair; mark low-sample pairs.
- Do not train on ScanNet200 labels for model improvement; this probe is diagnostic only.
- If a class has too few instances, exclude it from the main average and report separately.

### Output

- `Mosaic3D/error_analysis/reports/instance_separability.md`

## Task 2: Caption Granularity

### Goal

Determine whether instance flips are induced by overly fine-grained or sibling-biased training captions.

### Preconditions

- Confirm actual `anno_source` from the training configuration or checkpoint provenance.
- Verify `point_indices` align with `segment200.npy` on one training scene before full scan.
- Fix the caption term matching rule before scanning all captions.

### Method

1. Load `captions.<src>.npz` and `point_indices.<src>.npz` using `src.utils.io.unpack_list_of_np_arrays`.
2. Load `segment200.npy` for the same training scene.
3. For each caption region:
   - map `point_indices` to `segment200.npy`;
   - majority-vote valid GT class `g`;
   - parse caption terms `t` using longest matching over `CLASS_LABELS_200` plus a fixed alias table;
   - count `T[g][t] += 1`.
4. Row-normalize to estimate `P(term=t | GT=g)`.
5. For each S3 pair `(C -> J)`, report `P(caption=J | GT=C)` and `J` rank among non-C terms.
6. Compute Spearman correlation between training caption term distribution and eval FN destination distribution.

### Pre-Registered Criteria

- CONFIRM training over-specification if most eval `(C -> J)` pairs have `P(caption=J | GT=C) > 0`, `J` ranks first among non-C terms, and row-vector Spearman with eval-FN distribution is at least `0.4`.
- REFUTE if GT=C regions are mostly captioned as C and caption distribution is not correlated with eval-FN destinations.
- Otherwise: inconclusive.

### Required Controls And Self-Checks

- Longest match must prefer phrases such as `office chair` over `chair`.
- Captions with zero class terms must be counted under `<none>` or explicitly reported under a fixed rule.
- Captions with multiple class terms must follow one fixed rule, documented in the report.
- Report the chosen `anno_source`.

### Output

- `Mosaic3D/error_analysis/reports/caption_granularity.md`
- `Mosaic3D/error_analysis/reports/caption_granularity_T.npy`

## Final 2x2 Interpretation

| | caption biased fine | caption normal |
|---|---|---|
| feature separable | data/protocol granularity issue; consider caption normalization or hierarchy-aware training | decision/readout issue; consider mask-level calibration or reranking |
| feature not separable | representation damaged by fine labels; consider mask-to-text sibling contrast | true representation bottleneck; encoder/feature work required |

