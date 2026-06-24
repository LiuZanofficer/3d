set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1 || true

PY=/root/autodl-tmp/conda-envs/llz/bin/python
ROOT=/root/lz
TS=$(date +%Y%m%d_%H%M%S)
OUT=/root/lz_outputs/sc_full_capacity_check_${TS}
TRAIN_DIR=$OUT/train
EVAL_DIR=$OUT/eval

mkdir -p "$OUT" "$TRAIN_DIR" "$EVAL_DIR"
cd "$ROOT"

# Step1: memory-check config (leakage: train/train, no strong aug)
cat > "$ROOT/configs/data/sc_memory_check.yaml" <<'YAML'
_target_: src.data.datamodule.DataModule

batch_size: 1
num_workers: 0
pin_memory: true
collate_fn: null
sampler: null
num_classes: 200
coord_norm_enable: false
coord_norm_center: true
coord_norm_scale_method: percentile_95
color_drop_prob: 0.0
color_drop_mode: gray

train_dataset:
  _target_: src.data.scannet.dataset.ScanNet200Dataset
  data_dir: ${paths.data_dir}/scannet
  split: train
  repeat: 1
  ignore_label: -100
  transforms:
    - type: NormalizeColor
    - type: ToTensor

val_datasets:
  - _target_: src.data.scannet.dataset.ScanNet200Dataset
    data_dir: ${paths.data_dir}/scannet
    split: train
    ignore_label: -100
    transforms:
      - type: NormalizeColor
      - type: ToTensor
YAML

# Step2: train (>=100 epochs)
CUDA_VISIBLE_DEVICES=0 "$PY" src/train.py \
  model=spunet34c_lz \
  data=sc_memory_check \
  model.net.backbone.in_channels=6 \
  model.net.backbone.norm_type=bn \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="$TRAIN_DIR" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=100 \
  +trainer.precision=16-mixed \
  data.batch_size=1 \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  model.loss.seg_loss.lovasz_weight=0.0 \
  optim.lr=0.001 \
  +model.eval_cfg.eval_bn_batch_stats=true \
  callbacks.model_checkpoint.save_last=true \
  callbacks.model_checkpoint.save_top_k=0 \
  > "$TRAIN_DIR/train.log" 2>&1

train_latest=$(find "$TRAIN_DIR" -mindepth 2 -maxdepth 2 -type d | sort | tail -n1)
if [ -n "$train_latest" ] && [ -d "$train_latest/.hydra" ]; then
  if [ -d "$TRAIN_DIR/.hydra" ]; then mv "$TRAIN_DIR/.hydra" "$TRAIN_DIR/.hydra_prev_$(date +%s)"; fi
  cp -r "$train_latest/.hydra" "$TRAIN_DIR/.hydra"
fi

CKPT="$TRAIN_DIR/checkpoints/last.ckpt"
if [ ! -f "$CKPT" ]; then
  echo "Missing checkpoint: $CKPT" >&2
  exit 3
fi

# Step3: eval (same config + same BN policy)
CUDA_VISIBLE_DEVICES=0 "$PY" src/eval.py \
  model=spunet34c_lz \
  data=sc_memory_check \
  ckpt_path="$CKPT" \
  model.net.backbone.in_channels=6 \
  model.net.backbone.norm_type=bn \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="$EVAL_DIR" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  data.batch_size=1 \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  model.loss.seg_loss.lovasz_weight=0.0 \
  optim.lr=0.001 \
  +model.eval_cfg.eval_bn_batch_stats=true \
  > "$EVAL_DIR/eval.log" 2>&1

eval_latest=$(find "$EVAL_DIR" -mindepth 2 -maxdepth 2 -type d | sort | tail -n1)
if [ -n "$eval_latest" ] && [ -d "$eval_latest/.hydra" ]; then
  if [ -d "$EVAL_DIR/.hydra" ]; then mv "$EVAL_DIR/.hydra" "$EVAL_DIR/.hydra_prev_$(date +%s)"; fi
  cp -r "$eval_latest/.hydra" "$EVAL_DIR/.hydra"
fi

# Step4: summarize
export CAPACITY_OUT="$OUT"
"$PY" - <<'PY'
import json
import os
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

out = Path(os.environ['CAPACITY_OUT'])
train_dir = out / 'train'
eval_dir = out / 'eval'
metric_key = 'val/miou_present_gt'


def choose_event(run_dir: Path):
    best = None
    for vdir in sorted((run_dir / 'logs' / 'lz').glob('version_*')):
        for f in sorted(vdir.glob('events.out.tfevents.*')):
            ea = event_accumulator.EventAccumulator(str(f))
            try:
                ea.Reload()
            except Exception:
                continue
            tags = set(ea.Tags().get('scalars', []))
            if metric_key not in tags:
                continue
            vals = ea.Scalars(metric_key)
            if not vals:
                continue
            score = (len(vals), vdir.name)
            if best is None or score > best['score']:
                best = {
                    'ea': ea,
                    'file': str(f),
                    'version': vdir.name,
                    'score': score,
                }
    if best is None:
        raise RuntimeError(f'No event with {metric_key} in {run_dir}')
    return best


def stat(ea, key):
    tags = set(ea.Tags().get('scalars', []))
    if key not in tags:
        return {'best': None, 'last': None, 'count': 0}
    vals = ea.Scalars(key)
    if not vals:
        return {'best': None, 'last': None, 'count': 0}
    arr = [float(x.value) for x in vals]
    return {'best': max(arr), 'last': arr[-1], 'count': len(arr)}


def collect(run_dir: Path):
    selected = choose_event(run_dir)
    ea = selected['ea']
    payload = {
        'event_file': selected['file'],
        'event_version': selected['version'],
        'val/miou': stat(ea, 'val/miou'),
        'val/miou_present_gt': stat(ea, 'val/miou_present_gt'),
        'val/miou_present_union': stat(ea, 'val/miou_present_union'),
        'val/miou_fg_mosaic': stat(ea, 'val/miou_fg_mosaic'),
        'debug/use_text_guidance': stat(ea, 'debug/use_text_guidance'),
    }
    (run_dir / 'metrics.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload

train = collect(train_dir)
eval_ = collect(eval_dir)

train_best = float(train['val/miou_present_gt']['best'] or 0.0)
eval_value = float(eval_['val/miou_present_gt']['last'] or 0.0)
train_tg_zero = (train['debug/use_text_guidance']['best'] == 0.0 and train['debug/use_text_guidance']['last'] == 0.0)
eval_tg_zero = (eval_['debug/use_text_guidance']['best'] == 0.0 and eval_['debug/use_text_guidance']['last'] == 0.0)
tg_all_zero = bool(train_tg_zero and eval_tg_zero)

passed = bool(train_best >= 0.8 and eval_value >= 0.8)

result = {
    'train_best_val_miou_present_gt': train_best,
    'eval_val_miou_present_gt': eval_value,
    'debug_use_text_guidance_all_zero': tg_all_zero,
    'pass_threshold': 0.8,
    'final_decision': 'PASS' if passed else 'FAIL',
    'augmentations_train': ['NormalizeColor', 'ToTensor'],
    'augmentations_val': ['NormalizeColor', 'ToTensor'],
}
(out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')

fail_reasons = []
if train_best < 0.8:
    fail_reasons.append('train split=train underfit: optimization horizon still insufficient for full-set memorization')
if eval_value < 0.8:
    fail_reasons.append('eval on same split still low: checkpoint/state mismatch or optimization not converged enough')
if not tg_all_zero:
    fail_reasons.append('text guidance unexpectedly enabled; violates CE-only gate assumptions')
if not fail_reasons:
    fail_reasons = ['none']

summary_lines = [
    '# CAPACITY CHECK SUMMARY',
    '',
    f"- FINAL_DECISION: {'PASS' if passed else 'FAIL'}",
    f"- train_best_val/miou_present_gt: {train_best:.10f}",
    f"- eval_val/miou_present_gt: {eval_value:.10f}",
    f"- debug/use_text_guidance all zero: {tg_all_zero}",
    '- augmentations(train): NormalizeColor, ToTensor',
    '- augmentations(val): NormalizeColor, ToTensor',
]
if not passed:
    summary_lines.extend([
        '',
        '## Top-2 likely reasons',
        f"1. {fail_reasons[0]}",
        f"2. {fail_reasons[1] if len(fail_reasons) > 1 else 'none'}",
    ])
(out / 'CAPACITY_CHECK_SUMMARY.md').write_text('\n'.join(summary_lines), encoding='utf-8')

print(json.dumps({
    'out': str(out),
    'final_decision': 'PASS' if passed else 'FAIL',
    'train_best': train_best,
    'eval_value': eval_value,
    'debug_use_text_guidance_all_zero': tg_all_zero,
}, indent=2))
PY

printf "%s\n" "$OUT" > /tmp/sc_full_capacity_check_out_path.txt
