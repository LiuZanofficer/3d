set -euo pipefail
source /etc/network_turbo >/dev/null 2>&1 || true

PY=/root/autodl-tmp/conda-envs/llz/bin/python
ROOT=/root/lz
TS=$(date +%Y%m%d_%H%M%S)
OUT=/root/lz_outputs/single_scene_095_${TS}
TRAIN_DIR=$OUT/train
EVAL_DIR=$OUT/eval

mkdir -p "$OUT" "$TRAIN_DIR" "$EVAL_DIR"
cd "$ROOT"

# Step A: train (single scene debug1)
CUDA_VISIBLE_DEVICES=0 "$PY" src/train.py \
  model=spunet34c_lz \
  data=sc_smoke \
  data.train_dataset.split=debug1 \
  data.val_datasets.0.split=debug1 \
  model.net.backbone.in_channels=6 \
  model.net.backbone.norm_type=bn \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="$TRAIN_DIR" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=200 \
  +trainer.max_steps=2000 \
  +trainer.precision=16-mixed \
  data.batch_size=1 \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  model.loss.seg_loss.lovasz_weight=0.0 \
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

# Step B: eval (same split/config)
CUDA_VISIBLE_DEVICES=0 "$PY" src/eval.py \
  model=spunet34c_lz \
  data=sc_smoke \
  ckpt_path="$CKPT" \
  data.train_dataset.split=debug1 \
  data.val_datasets.0.split=debug1 \
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
  +model.eval_cfg.eval_bn_batch_stats=true \
  > "$EVAL_DIR/eval.log" 2>&1

eval_latest=$(find "$EVAL_DIR" -mindepth 2 -maxdepth 2 -type d | sort | tail -n1)
if [ -n "$eval_latest" ] && [ -d "$eval_latest/.hydra" ]; then
  if [ -d "$EVAL_DIR/.hydra" ]; then mv "$EVAL_DIR/.hydra" "$EVAL_DIR/.hydra_prev_$(date +%s)"; fi
  cp -r "$eval_latest/.hydra" "$EVAL_DIR/.hydra"
fi

# Step C: collect metrics + verdict
export SINGLE_SCENE_OUT="$OUT"
"$PY" - <<'PY'
import json
import os
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

out = Path(os.environ['SINGLE_SCENE_OUT'])
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


def stats(ea, key: str):
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
        'val/miou': stats(ea, 'val/miou'),
        'val/miou_present_gt': stats(ea, 'val/miou_present_gt'),
        'val/miou_present_union': stats(ea, 'val/miou_present_union'),
        'val/miou_fg_mosaic': stats(ea, 'val/miou_fg_mosaic'),
        'debug/use_text_guidance': stats(ea, 'debug/use_text_guidance'),
    }
    (run_dir / 'metrics.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload

train_payload = collect(train_dir)
eval_payload = collect(eval_dir)

train_best = float(train_payload['val/miou_present_gt']['best'] or 0.0)
eval_value = float(eval_payload['val/miou_present_gt']['last'] or 0.0)
passed = bool(train_best >= 0.95 and eval_value >= 0.95)

result = {
    'train_best_val/miou_present_gt': train_best,
    'eval_val/miou_present_gt': eval_value,
    'final_decision': 'PASS' if passed else 'FAIL',
    'threshold': 0.95,
}
(out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
(out / 'SINGLE_SCENE_095_SUMMARY.md').write_text(
    '\n'.join([
        '# SINGLE_SCENE_095 SUMMARY',
        '',
        f"- FINAL_DECISION: {'PASS' if passed else 'FAIL'}",
        f"- train_best_val/miou_present_gt: {train_best:.10f}",
        f"- eval_val/miou_present_gt: {eval_value:.10f}",
    ]),
    encoding='utf-8',
)

print(json.dumps({
    'out': str(out),
    'final_decision': 'PASS' if passed else 'FAIL',
    'train_best': train_best,
    'eval_value': eval_value,
}, indent=2))
PY

printf "%s\n" "$OUT" > /tmp/single_scene_095_out_path.txt
