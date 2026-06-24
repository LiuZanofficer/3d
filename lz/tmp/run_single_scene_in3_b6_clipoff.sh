#!/usr/bin/env bash
set -euo pipefail

cd /root/lz
TS=$(date +%Y%m%d_%H%M%S)
OUT=/root/lz_outputs/single_scene_in3_b6_clipoff_${TS}
mkdir -p "${OUT}/train" "${OUT}/eval"
PY=/root/autodl-tmp/conda-envs/llz/bin/python

echo "[INFO] OUT=${OUT}" | tee "${OUT}/run.log"

echo "[STEP] train" | tee -a "${OUT}/run.log"
"${PY}" src/train.py \
  model=spunet34c_lz \
  data=sc_smoke \
  data.train_dataset.split=debug1 \
  data.val_datasets.0.split=debug1 \
  model.net.backbone.in_channels=3 \
  model.net.backbone.norm_type=bn \
  model.text_encoder_cfg.use_clip=false \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="${OUT}/train" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=200 \
  +trainer.max_steps=2000 \
  +trainer.precision=bf16-mixed \
  data.batch_size=6 \
  ++data.drop_last_train=false \
  data.num_workers=4 \
  data.pin_memory=true \
  ++data.persistent_workers=true \
  ++data.prefetch_factor=4 \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  model.loss.seg_loss.lovasz_weight=0.0 \
  +model.eval_cfg.eval_bn_batch_stats=true \
  callbacks.model_checkpoint.save_last=true \
  callbacks.model_checkpoint.save_top_k=0 \
  seed=42 \
  > "${OUT}/train.log" 2>&1

TRAIN_HYDRA_DIR="$(find "${OUT}/train" -type f -path '*/.hydra/config.yaml' | head -n 1 | xargs dirname || true)"
if [[ -n "${TRAIN_HYDRA_DIR}" && -f "${TRAIN_HYDRA_DIR}/config.yaml" ]]; then
  cp "${TRAIN_HYDRA_DIR}/config.yaml" "${OUT}/config.yaml"
  cp "${TRAIN_HYDRA_DIR}/overrides.yaml" "${OUT}/overrides.yaml"
fi

"${PY}" - <<PY2
import json
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

def dump_metrics(run_dir: Path, out_json: Path):
    evs = sorted(run_dir.glob('logs/lz/version_*/events.out.tfevents*'))
    if not evs:
        out_json.write_text(json.dumps({'error': 'no event file'}, indent=2), encoding='utf-8')
        return
    ev = evs[-1]
    ea = event_accumulator.EventAccumulator(str(ev), size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    keys = [
        'val/miou',
        'val/miou_present_gt',
        'val/miou_present_union',
        'val/miou_fg_mosaic',
        'debug/use_text_guidance',
        'train/loss_epoch',
    ]
    out = {'event_file': str(ev)}
    for key in keys:
        if key in tags:
            vals = ea.Scalars(key)
            out[key] = {
                'best': max(v.value for v in vals),
                'last': vals[-1].value,
                'count': len(vals),
            }
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

dump_metrics(Path('${OUT}/train'), Path('${OUT}/train/metrics.json'))
PY2

echo "[STEP] eval" | tee -a "${OUT}/run.log"
"${PY}" src/eval.py \
  model=spunet34c_lz \
  data=sc_smoke \
  ckpt_path="${OUT}/train/checkpoints/last.ckpt" \
  data.train_dataset.split=debug1 \
  data.val_datasets.0.split=debug1 \
  model.net.backbone.in_channels=3 \
  model.net.backbone.norm_type=bn \
  model.text_encoder_cfg.use_clip=false \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="${OUT}/eval" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  data.batch_size=6 \
  ++data.drop_last_train=false \
  data.num_workers=4 \
  data.pin_memory=true \
  ++data.persistent_workers=true \
  ++data.prefetch_factor=4 \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  model.loss.seg_loss.lovasz_weight=0.0 \
  +model.eval_cfg.eval_bn_batch_stats=true \
  seed=42 \
  > "${OUT}/eval.log" 2>&1

"${PY}" - <<PY3
import json
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

def dump_metrics(run_dir: Path, out_json: Path):
    evs = sorted(run_dir.glob('logs/lz/version_*/events.out.tfevents*'))
    if not evs:
        out_json.write_text(json.dumps({'error': 'no event file'}, indent=2), encoding='utf-8')
        return
    ev = evs[-1]
    ea = event_accumulator.EventAccumulator(str(ev), size_guidance={'scalars': 0})
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    keys = [
        'val/miou',
        'val/miou_present_gt',
        'val/miou_present_union',
        'val/miou_fg_mosaic',
        'debug/use_text_guidance',
    ]
    out = {'event_file': str(ev)}
    for key in keys:
        if key in tags:
            vals = ea.Scalars(key)
            out[key] = {
                'best': max(v.value for v in vals),
                'last': vals[-1].value,
                'count': len(vals),
            }
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')

dump_metrics(Path('${OUT}/eval'), Path('${OUT}/eval/metrics.json'))
PY3

"${PY}" - <<PY4
import json
from pathlib import Path

train_metrics = json.loads(Path('${OUT}/train/metrics.json').read_text(encoding='utf-8'))
eval_metrics = json.loads(Path('${OUT}/eval/metrics.json').read_text(encoding='utf-8'))

def get_best(data, key):
    node = data.get(key)
    if isinstance(node, dict):
        return node.get('best')
    return None

result = {
    'run': '${OUT}',
    'train_best_val_miou_present_gt': get_best(train_metrics, 'val/miou_present_gt'),
    'eval_best_val_miou_present_gt': get_best(eval_metrics, 'val/miou_present_gt'),
    'train_debug_use_text_guidance_best': get_best(train_metrics, 'debug/use_text_guidance'),
    'eval_debug_use_text_guidance_best': get_best(eval_metrics, 'debug/use_text_guidance'),
}

Path('${OUT}/result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
PY4

echo "[DONE] ${OUT}" | tee -a "${OUT}/run.log"
