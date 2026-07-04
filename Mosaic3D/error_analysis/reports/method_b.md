# TASK1 - Method B (hard-negative caption loss): training + eval plan

**Status:** training in progress (8-card NCCL, healthy). Final-checkpoint eval is
GPU-gated and runs automatically via `scripts/phase4_autoeval.sh` (see below).
No Method-B improvement is claimed yet: at epoch 1 the model is still below baseline.

## Training setup (running)
- run dir: `/root/runs/hardneg_full_8gpu_nccl277` (container overlay, writable; NFS is read-only).
- config: `experiment=train_spunet_multidata_ppt data=sc+ar+sc++ model/loss=caption_siglip_hardneg`
  `+model.loss_cfg.caption_loss.all_gather=true`, DDP/NCCL, `trainer.devices=8`, `max_epochs=128`,
  `+init_ckpt_path=./qz/sc+ar+sc++.ckpt`, ckpt every 200 steps + `save_last`.
- caption source = gsam2/seem (default). Annotation-free: no ScanNet200 GT labels in the loss.
- **Init caveat (honest):** loading qz init reported `missing=1 unexpected=588` — a large part of the
  checkpoint keys did not map onto the current graph, so B does not start exactly at the reproduced
  baseline weights and partly re-learns. This depresses early-epoch metrics.

## Health / stability (8-card NCCL)
- 8 GPUs at 100% util, ~29-31 GB/32 GB each; no `illegal memory access`, no `NCCL abort`, no Traceback.
- loss (CSV): `hard_negative_caption_loss` stable ~0.0027-0.0032 (not exploding); `caption_loss` low and
  batch-noisy (0.007-0.018, tracks num_objects 115-298); total loss ~0.010-0.021, no collapse/divergence.

| step | caption_loss | hardneg | total | num_obj |
|---:|---:|---:|---:|---:|
| 49 | 0.01417 | 0.00267 | 0.01684 | 168 |
| 99 | 0.00798 | 0.00272 | 0.01070 | 290 |
| 149 | 0.01323 | 0.00324 | 0.01647 | 242 |
| 199 | 0.01816 | 0.00278 | 0.02094 | 115 |
| 249 | 0.00711 | 0.00271 | 0.00982 | 298 |
| 299 | 0.01161 | 0.00311 | 0.01471 | 267 |

## Native validation metrics of the current B checkpoint (baseline readout, GPU-free from train loop)
| epoch | step | val fg-mIoU | head | common | tail | mAcc | mAP | mAP25 | mAP50 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 110 | 0.11586 | 0.2608 | 0.0676 | 0.0153 | 0.2225 | 0.0896 | 0.1312 | 0.1177 |
| 1 | 221 | 0.12734 | 0.2819 | 0.0778 | 0.0180 | 0.2406 | 0.0981 | 0.1472 | 0.1292 |

- Baseline reference (reproduced): point fg-mIoU **0.15475**, mAP **0.11486**.
- Read: epoch-1 B is **below** baseline and **rising** (fg-mIoU 0.1159 -> 0.1273). Too early (2/128 epochs) and
  affected by the partial init to conclude anything. NOT an improvement over baseline yet.

## PRE-REGISTERED criteria (locked, evaluated on the FINAL checkpoint)
1. B or B+mask_text_vote fg-mIoU **> 0.17573** (must beat the consistency axis, not just baseline).
2. `mean|cos(d_vis,d_txt)|` on the trained-B dump rises significantly from baseline ~0.03.
3. gain concentrates on the near-sibling pairs found in TASK2; non-sibling classes not materially hurt.
4. instance mAP stays ~0.115.
All four required to declare Method B closes the loop; otherwise record the failure point.

## Phase-4 eval (staged, auto-runs when a GPU frees or training ends)
`scripts/phase4_autoeval.sh` (running, PID logged in `method_b_autoeval/autoeval.log`) fires only when
`status.txt != RUNNING` or a GPU has >=20 GB free at <20% util for 3 polls (never mid-training, to protect
the run). It evaluates three readout modes on `last.ckpt` and computes fg-mIoU (fixed口径 self-check) + mAP:
- `MOSAIC3D_READOUT_MODE=baseline` -> B baseline readout
- `MOSAIC3D_READOUT_MODE=mask_text_vote` -> B + Method A consistency
- `MOSAIC3D_READOUT_MODE=anchor_decorrelate` (thr 0.90, cap 8) -> B + inference direction-fix
then reruns `analyze_readout_direction.py` on the B dump for the cos mechanism check.
Manual repro (single GPU):
```
CUDA_VISIBLE_DEVICES=<g> MOSAIC3D_DUMP_EVAL=1 MOSAIC3D_READOUT_MODE=<mode> \
python src/eval.py experiment=train_spunet_multidata_ppt data=sc+ar+sc++ \
  ckpt_path=/root/runs/hardneg_full_8gpu_nccl277/checkpoints/last.ckpt \
  trainer.devices=1 hydra.run.dir=/root/runs/phase4_eval/<mode>
```
Results will be appended to `error_analysis/reports/method_b_autoeval/summary.txt`.

## Cross-references
- TASK2 (`readout_inference_diag.md`): annotation-free inference readout cannot fix naming (dead-knot);
  motivates the *trained* direction fix that Method B attempts.
- TASK3 (`axes_summary.md`): mask_text_vote is the consistency axis (+0.02098); Method B must add on top of it.

---

## Phase-4 FINAL eval on trained B (`last.ckpt`, epoch 7, step 887)

Training was stopped at epoch 7/128 because native val fg-mIoU had plateaued since epoch 4
(0.1381/0.1381/0.1375/0.1369 at epochs 4/5/6/7) and init was verified healthy
(see `init_load_diagnosis.md`), so the qualitative verdict was already determined.

Three readout modes evaluated on the SAME B checkpoint, fixed口径 fg-mIoU (self-checked, == native val), mAP:

| model / readout | fg-mIoU | Δ vs baseline 0.15475 | Δ vs base+MTV 0.17573 | mAP | Δ vs 0.11486 | head | common | tail |
|---|---|---|---|---|---|---|---|---|
| baseline (reproduced) | 0.15475 | — | — | 0.11486 | — | — | — | — |
| baseline + mask_text_vote (Method A) | 0.17573 | +0.02098 | — | 0.11236 | −0.00250 | — | — | — |
| **B, baseline readout** | 0.1380 | **−0.0168** | −0.0377 | 0.1111 | −0.0038 | 0.302 | 0.087 | 0.020 |
| **B + mask_text_vote** | 0.1598 | +0.0050 | **−0.0159** | 0.1084 | −0.0065 | 0.340 | 0.107 | 0.026 |
| **B + anchor_decorrelate** | 0.1549 | +0.0001 | −0.0208 | 0.1061 | −0.0088 | 0.332 | 0.103 | 0.024 |

### cos mechanism (analyze_readout_direction on B baseline dump)
- B text anchors vs baseline text anchors: **max normed diff = 3.1e-7** → the CLIP text encoder is frozen,
  so Method B cannot and did not move the text readout side.
- `mean_cos(d_vis, d_txt) = 0.031`, median 0.020, frac_low = 1.000, random 0.025.
  Baseline was ~0.034 → **cos did NOT rise**; the visual-discriminant vs text-readout mismatch is unchanged.

### Pre-registered verdict
- **① B or B+mask_text_vote > 0.17573?** NO. Max is B+MTV = 0.1598. **Not met.**
- **② cos rises from ~0.03?** NO. cos = 0.031 ≈ baseline 0.034. **Refuted.**
- Trained B is **worse than the reproduced baseline at every readout** (fg-mIoU −0.017 at baseline readout;
  B+MTV 0.160 < baseline+MTV 0.176), and mAP is also slightly lower (0.111 vs 0.115).

### Conclusion — result (b): Method B (this config) does NOT improve; it slightly degrades
With `caption_siglip_hardneg` at this LR/weighting and a frozen CLIP text encoder, fine-tuning only
perturbs the 3D backbone away from the eval-favorable operating point without improving visual–text
alignment. The sibling-naming knot is NOT closed by this training. The only net-positive annotation-free
lever remains **Method A `mask_text_vote` on the reproduced baseline (+0.02098 fg-mIoU)**, which requires
no Method B training. Any future Method-B attempt must also unfreeze/adapt the text side (or use a
different objective), since a frozen-text hard-negative loss cannot change the readout direction.


---

## Phase-4 FINAL eval on trained B (`last.ckpt`, epoch 7, step 887)

Training was stopped at epoch 7/128 because native val fg-mIoU had plateaued since epoch 4
(0.1381/0.1381/0.1375/0.1369 at epochs 4/5/6/7) and init was verified healthy
(see `init_load_diagnosis.md`), so the qualitative verdict was already determined.

Three readout modes on the SAME B checkpoint; fixed-koujing fg-mIoU (self-checked, == native val) and mAP:

| model / readout | fg-mIoU | d vs baseline 0.15475 | d vs base+MTV 0.17573 | mAP | d vs 0.11486 | head | common | tail |
|---|---|---|---|---|---|---|---|---|
| baseline (reproduced) | 0.15475 | - | - | 0.11486 | - | - | - | - |
| baseline + mask_text_vote (Method A) | 0.17573 | +0.02098 | - | 0.11236 | -0.00250 | - | - | - |
| B, baseline readout | 0.1380 | -0.0168 | -0.0377 | 0.1111 | -0.0038 | 0.302 | 0.087 | 0.020 |
| B + mask_text_vote | 0.1598 | +0.0050 | -0.0159 | 0.1084 | -0.0065 | 0.340 | 0.107 | 0.026 |
| B + anchor_decorrelate | 0.1549 | +0.0001 | -0.0208 | 0.1061 | -0.0088 | 0.332 | 0.103 | 0.024 |

### cos mechanism (analyze_readout_direction on B baseline dump)
- B text anchors vs baseline text anchors: max normed diff = 3.1e-7 -> CLIP text encoder is frozen,
  so Method B cannot and did not move the text readout side.
- mean_cos(d_vis, d_txt) = 0.031, median 0.020, frac_low = 1.000, random 0.025.
  Baseline was ~0.034 -> cos did NOT rise; the visual-discriminant vs text-readout mismatch is unchanged.

### Pre-registered verdict
- (1) B or B+mask_text_vote > 0.17573?  NO. Max is B+MTV = 0.1598. Not met.
- (2) cos rises from ~0.03?  NO. cos = 0.031 ~= baseline 0.034. Refuted.
- Trained B is worse than the reproduced baseline at every readout (fg-mIoU -0.017 at baseline readout;
  B+MTV 0.160 < baseline+MTV 0.176), and mAP is also slightly lower (0.111 vs 0.115).

### Conclusion -- result (b): Method B (this config) does NOT improve; it slightly degrades
With caption_siglip_hardneg at this LR/weighting and a frozen CLIP text encoder, fine-tuning only
perturbs the 3D backbone away from the eval-favorable operating point without improving visual-text
alignment. The sibling-naming knot is NOT closed by this training. The only net-positive annotation-free
lever remains Method A mask_text_vote on the reproduced baseline (+0.02098 fg-mIoU), which requires
no Method B training. Any future Method-B attempt must also unfreeze/adapt the text side (or use a
different objective), since a frozen-text hard-negative loss cannot change the readout direction.
