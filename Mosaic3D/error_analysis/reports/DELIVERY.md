# DELIVERY -- one-line conclusion

**Result (b): trained Method B (caption_siglip_hardneg, this config) does NOT improve open-vocab 3D
segmentation -- it slightly degrades it.** On the same B checkpoint (epoch 7, plateaued): B baseline
readout fg-mIoU=0.1380 (< baseline 0.15475), B+mask_text_vote=0.1598 (< baseline+MTV 0.17573),
B+anchor_decorrelate=0.1549; mAP 0.106-0.111 (<= baseline 0.11486). The readout-direction mismatch is
unchanged (cos 0.031 vs 0.034), because the CLIP text encoder is frozen (B text anchors == baseline,
diff 3e-7), so a frozen-text hard-negative loss structurally cannot fix the naming knot.

The only validated annotation-free gain remains **Method A `mask_text_vote` on the reproduced baseline:
fg-mIoU 0.15475 -> 0.17573 (+0.02098), mAP 0.11486 -> 0.11236 (-0.00250)** -- readout-only, no training.

Init was verified healthy (missing=1/unexpected=588 is the frozen CLIP by design; backbone loaded from qz).
Details: method_b.md, init_load_diagnosis.md, readout_inference_diag.md, axes_summary.md.
