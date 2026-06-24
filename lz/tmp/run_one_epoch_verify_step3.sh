#!/usr/bin/env bash
set -euo pipefail
OUT="/root/lz_outputs/a800_throughput_tune_economy_20260221_162740"
VOUT="$OUT/one_epoch_verify"
mkdir -p "$VOUT"
START=$(date +%s)
cd /root/lz
/root/autodl-tmp/conda-envs/llz/bin/python src/train.py \
  model=spunet34c_lz \
  data=sc_memory_check \
  paths.root_dir=/root/lz \
  paths.data_dir=/root/autodl-tmp/datasets/mosaic3d/data \
  paths.output_dir="$VOUT" \
  data.train_dataset.split=train \
  data.val_datasets.0.split=train \
  model.loss.weights.seg_loss=1.0 \
  model.loss.weights.instance_loss=0.0 \
  model.loss.weights.hpza_loss=0.0 \
  model.loss.seg_loss.lovasz_weight=0.0 \
  model.text_encoder_cfg.use_clip=false \
  +model.eval_cfg.eval_bn_batch_stats=true \
  trainer.accelerator=gpu \
  trainer.devices=1 \
  +trainer.precision=bf16-mixed \
  data.batch_size=2 \
  data.num_workers=12 \
  data.pin_memory=true \
  ++data.persistent_workers=true \
  ++data.prefetch_factor=4 \
  trainer.max_epochs=1 \
  +trainer.check_val_every_n_epoch=5 \
  +trainer.num_sanity_val_steps=0 \
  callbacks.model_checkpoint.save_last=true \
  callbacks.model_checkpoint.save_top_k=1 \
  +callbacks.model_checkpoint.every_n_train_steps=1000 \
  > "$VOUT/train.log" 2>&1
RC=$?
END=$(date +%s)
DUR=$((END-START))
/root/autodl-tmp/conda-envs/llz/bin/python - <<'PY'
import json, re
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator
out = Path('/root/lz_outputs/a800_throughput_tune_economy_20260221_162740/one_epoch_verify')
log = out / 'train.log'
text = log.read_text(encoding='utf-8', errors='ignore') if log.exists() else ''
matches = re.findall(r"(\d+)/(\d+)\s*\[", text)
steps_completed = int(matches[-1][0]) if matches else 0
steps_total = int(matches[-1][1]) if matches else 0
mean_step = None
events = list(out.glob('**/events.out.tfevents*'))
pts = []
for ef in events:
    try:
        ea = event_accumulator.EventAccumulator(str(ef), size_guidance={'scalars':0})
        ea.Reload()
        if 'train/loss_step' in ea.Tags().get('scalars', []):
            for ev in ea.Scalars('train/loss_step'):
                pts.append((int(ev.step), float(ev.wall_time)))
    except Exception:
        pass
if pts:
    pts.sort(key=lambda x:(x[0], x[1]))
    dedup = {}
    for s,t in pts:
        dedup[s]=t
    steps = sorted(dedup.keys())
    if len(steps) >= 5:
        s0 = next((s for s in steps if s>=20), steps[0])
        s1 = steps[-1]
        if s1 > s0:
            dt = dedup[s1]-dedup[s0]
            ds = s1-s0
            if dt>0 and ds>0:
                mean_step = dt/ds
duration = int((Path('/tmp/one_epoch_duration.txt').read_text().strip())) if Path('/tmp/one_epoch_duration.txt').exists() else None
if mean_step is None and duration and steps_completed>0:
    mean_step = duration / steps_completed
sps = (2/mean_step) if mean_step else None
result = {
  'return_code': int(Path('/tmp/one_epoch_rc.txt').read_text().strip()) if Path('/tmp/one_epoch_rc.txt').exists() else None,
  'duration_sec': duration,
  'one_epoch_time_min': (duration/60.0) if duration is not None else None,
  'steps_completed': steps_completed,
  'steps_total': steps_total,
  'batch_size': 2,
  'num_workers': 12,
  'precision': 'bf16-mixed',
  'pin_memory': True,
  'persistent_workers': True,
  'prefetch_factor': 4,
  'mean_step_sec': mean_step,
  'samples_per_sec': sps,
  'train_log': str(log)
}
(out / 'one_epoch_verify.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False))
PY
echo "$RC" > /tmp/one_epoch_rc.txt
echo "$DUR" > /tmp/one_epoch_duration.txt
# regenerate with duration/rc populated
/root/autodl-tmp/conda-envs/llz/bin/python - <<'PY'
import json
from pathlib import Path
out = Path('/root/lz_outputs/a800_throughput_tune_economy_20260221_162740/one_epoch_verify')
p = out / 'one_epoch_verify.json'
obj = json.loads(p.read_text(encoding='utf-8'))
obj['return_code'] = int(Path('/tmp/one_epoch_rc.txt').read_text().strip())
obj['duration_sec'] = int(Path('/tmp/one_epoch_duration.txt').read_text().strip())
obj['one_epoch_time_min'] = obj['duration_sec'] / 60.0
if obj.get('mean_step_sec') is None and obj.get('steps_completed',0)>0:
    obj['mean_step_sec'] = obj['duration_sec'] / obj['steps_completed']
if obj.get('mean_step_sec'):
    obj['samples_per_sec'] = obj['batch_size'] / obj['mean_step_sec']
p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(obj, ensure_ascii=False))
PY
# update top-level artifacts
/root/autodl-tmp/conda-envs/llz/bin/python - <<'PY'
import json
from pathlib import Path
root = Path('/root/lz_outputs/a800_throughput_tune_economy_20260221_162740')
probe = json.loads((root/'probe_results.json').read_text(encoding='utf-8'))
rows = probe['results']
stable = [r for r in rows if r.get('stable_200_steps') and r.get('batch_size')==2 and r.get('num_workers') in [4,8,12]]
stable = sorted(stable, key=lambda r: float(r.get('samples_per_sec') or 0), reverse=True)
best = stable[0] if stable else None
if best:
    (root/'best_config.json').write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding='utf-8')
    rec = [
      f"+trainer.precision={best['precision']}",
      f"data.batch_size={best['batch_size']}",
      f"data.num_workers={best['num_workers']}",
      "data.pin_memory=true",
      "++data.persistent_workers=true",
      "++data.prefetch_factor=4",
      "model=spunet34c_lz",
      "data=sc_memory_check",
      "data.train_dataset.split=train",
      "data.val_datasets.0.split=train",
      "model.loss.weights.seg_loss=1.0",
      "model.loss.weights.instance_loss=0.0",
      "model.loss.weights.hpza_loss=0.0",
      "model.loss.seg_loss.lovasz_weight=0.0",
      "model.text_encoder_cfg.use_clip=false",
      "+model.eval_cfg.eval_bn_batch_stats=true",
    ]
    (root/'recommended_overrides.txt').write_text('\n'.join(rec)+'\n', encoding='utf-8')
ver = json.loads((root/'one_epoch_verify'/'one_epoch_verify.json').read_text(encoding='utf-8'))
final = 'READY' if ver.get('return_code')==0 else 'BLOCKED'
summary = [
  '# THROUGHPUT_TUNING_SUMMARY',
  f"- FINAL_DECISION: {final}",
  f"- selected_precision: {best['precision'] if best else 'N/A'}",
  f"- selected_batch_size: {best['batch_size'] if best else 'N/A'}",
  f"- selected_num_workers: {best['num_workers'] if best else 'N/A'}",
  '- selected_pin_memory/persistent_workers/prefetch_factor: true/true/4',
  f"- best_samples_per_sec: {best['samples_per_sec'] if best else 'N/A'}",
  f"- one_epoch_time_min: {ver.get('one_epoch_time_min')}",
]
(root/'THROUGHPUT_TUNING_SUMMARY.md').write_text('\n'.join(summary)+'\n', encoding='utf-8')
print(json.dumps({'final_decision':final,'best':best,'verify':ver}, ensure_ascii=False))
PY
