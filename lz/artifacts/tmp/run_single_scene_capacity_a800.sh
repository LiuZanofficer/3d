#!/usr/bin/env bash
set -euo pipefail

cd /root/lz
TS=$(date +%Y%m%d_%H%M%S)
REF=/root/lz_outputs/single_scene_095_20260219_173644
FAST=/root/lz_outputs/a800_throughput_tune_economy_refactor_20260222_085718/one_epoch_verify
OUT=/root/lz_outputs/single_scene_capacity_a800_repro_${TS}
mkdir -p "${OUT}/train" "${OUT}/eval"
PY=/root/autodl-tmp/conda-envs/llz/bin/python

echo "[INFO] OUT=${OUT}" | tee "${OUT}/run.log"

echo "[STEP] train" | tee -a "${OUT}/run.log"
"${PY}" src/train.py \
  model=spunet34c_lz \
  data=sc_smoke \
  data.train_dataset.split=debug1 \
  data.val_datasets.0.split=debug1 \
  model.net.backbone.in_channels=6 \
  model.net.backbone.norm_type=bn \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="${OUT}/train" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  trainer.max_epochs=200 \
  +trainer.max_steps=2000 \
  +trainer.precision=bf16-mixed \
  data.batch_size=1 \
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
  model.net.backbone.in_channels=6 \
  model.net.backbone.norm_type=bn \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="${OUT}/eval" \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  data.batch_size=1 \
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

ref_train = json.loads(Path('${REF}/train/metrics.json').read_text(encoding='utf-8'))
ref_eval = json.loads(Path('${REF}/eval/metrics.json').read_text(encoding='utf-8'))
new_train = json.loads(Path('${OUT}/train/metrics.json').read_text(encoding='utf-8'))
new_eval = json.loads(Path('${OUT}/eval/metrics.json').read_text(encoding='utf-8'))

def get_best(data, key):
    node = data.get(key)
    if isinstance(node, dict):
        return node.get('best')
    return None

compare = {
    'reference_run': '${REF}',
    'fast_run': '${FAST}',
    'new_run': '${OUT}',
    'reference': {
        'train_best_val_miou_present_gt': get_best(ref_train, 'val/miou_present_gt'),
        'eval_best_val_miou_present_gt': get_best(ref_eval, 'val/miou_present_gt'),
        'train_debug_use_text_guidance_best': get_best(ref_train, 'debug/use_text_guidance'),
        'eval_debug_use_text_guidance_best': get_best(ref_eval, 'debug/use_text_guidance'),
    },
    'a800': {
        'train_best_val_miou_present_gt': get_best(new_train, 'val/miou_present_gt'),
        'eval_best_val_miou_present_gt': get_best(new_eval, 'val/miou_present_gt'),
        'train_debug_use_text_guidance_best': get_best(new_train, 'debug/use_text_guidance'),
        'eval_debug_use_text_guidance_best': get_best(new_eval, 'debug/use_text_guidance'),
    },
}

rt = compare['reference']['train_best_val_miou_present_gt']
re = compare['reference']['eval_best_val_miou_present_gt']
at = compare['a800']['train_best_val_miou_present_gt']
ae = compare['a800']['eval_best_val_miou_present_gt']

compare['delta'] = {
    'train_abs': None if (rt is None or at is None) else at - rt,
    'eval_abs': None if (re is None or ae is None) else ae - re,
}

pass_cond = (
    at is not None and ae is not None and
    at >= 0.95 and ae >= 0.95 and
    abs(compare['delta']['train_abs']) <= 0.03 and
    abs(compare['delta']['eval_abs']) <= 0.03
)
compare['final_decision'] = 'PASS' if pass_cond else 'FAIL'

Path('${OUT}/repro_compare.json').write_text(
    json.dumps(compare, ensure_ascii=False, indent=2),
    encoding='utf-8',
)

summary = "\\n".join([
    "# SINGLE_SCENE_CAPACITY_A800_SUMMARY",
    f"- FINAL_DECISION: {compare['final_decision']}",
    f"- train_best_val/miou_present_gt: {at}",
    f"- eval_best_val/miou_present_gt: {ae}",
    f"- debug/use_text_guidance(train/eval): {compare['a800']['train_debug_use_text_guidance_best']} / {compare['a800']['eval_debug_use_text_guidance_best']}",
    f"- delta_vs_ref(train/eval): {compare['delta']['train_abs']} / {compare['delta']['eval_abs']}",
    "- constraints: debug1/debug1, in_channels=6, CE-only, eval_bn_batch_stats=true, same seed.",
    "- speed inheritance: num_workers/pin_memory/persistent_workers/prefetch/precision from FAST_RUN; batch kept at 1 for single-scene dataloader validity.",
    "",
])
Path('${OUT}/SINGLE_SCENE_CAPACITY_A800_SUMMARY.md').write_text(summary, encoding='utf-8')
print(json.dumps({'OUT': '${OUT}', 'final_decision': compare['final_decision'], 'train_best': at, 'eval_best': ae}, ensure_ascii=False))
PY4

echo "[DONE] ${OUT}" | tee -a "${OUT}/run.log"
