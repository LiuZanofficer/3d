# Init-load diagnosis: `missing=1, unexpected=588` is BENIGN by design

## Question
Method B validation fg-mIoU sits below baseline (plateau ~0.138 < 0.15475). Is the qz init
load broken (only partially loading the backbone)?

## Method
Diffed qz `state_dict` keys against the model's own `last.ckpt` (== true model param names),
and read the loading code path.

## Evidence
- qz keys = 1654, model keys = 1655; **matched = 1654, shape-mismatches = 0**.
- model-only (missing) = **1**: `caption_loss.logit_bias` (new hard-negative SigLIP param; correctly absent in qz).
- The `unexpected=588` at `on_fit_start` = exactly the `clip_encoder.*` keys
  (288 text `transformer.resblocks` + 288 `visual.transformer` + ~12 heads/embeds).
- Code confirms this is intentional (`language_module.py`):
  - `self.strict_loading = False` with comment "clip_encoder ... is saved into the checkpoint but is a
    non-registered object at load time ... ignores those extra clip_encoder.* keys".
  - `configure_model()`: clip_encoder is rebuilt from HF pretrained, **frozen** (`requires_grad=False`),
    removed from `self._modules`, and set via `object.__setattr__` → it is NOT in `self.state_dict()`,
    so qz's clip keys show up as "unexpected" and are skipped.

## Verdict
**Init is healthy.** The 3D segmentation backbone (`net.backbone.*`, 1044 keys) and `embedding_table`
loaded from the qz baseline correctly (missing=1 proves every model param except the new loss bias came
from qz). The 588 "unexpected" are the frozen CLIP encoder, which is rebuilt from the same HF pretrained
weights the baseline uses and is intentionally excluded from strict loading. The baseline eval (fg-mIoU
0.155) loads qz via Lightning's native strict path; training uses the documented non-strict path — both
end up with the same frozen CLIP.

## Consequence
The below-baseline plateau is **NOT** a loading bug. It is a genuine Method-B training-dynamics result:
with the current config (`caption_siglip_hardneg`, this LR/weighting), fine-tuning the baseline backbone
under the hard-negative caption loss settles fg-mIoU at ~0.138 (< baseline 0.155) and has plateaued since
epoch ~4 (0.1381/0.1381/0.1375 at epochs 4/5/6). Running to epoch 128 is very unlikely to change the
qualitative verdict. mAP creeps toward baseline (~0.112 at epoch 6).
