#!/usr/bin/env bash
set -euo pipefail
OUT="/root/lz_outputs/a800_throughput_tune_economy_20260221_162740"
python_bin="/root/autodl-tmp/conda-envs/llz/bin/python"

"$python_bin" - <<'PY'
import json, re
from pathlib import Path

out = Path('/root/lz_outputs/a800_throughput_tune_economy_20260221_162740')
probe = json.loads((out / 'probe_results.json').read_text(encoding='utf-8'))
rows = probe.get('results', [])

stable = [r for r in rows if r.get('stable_200_steps') and r.get('samples_per_sec') is not None]
if not stable:
    best = {
        'status': 'BLOCKED',
        'reason': 'no stable_200_steps result in probe_results.json'
    }
else:
    best = max(stable, key=lambda r: float(r.get('samples_per_sec') or 0.0))

(out / 'best_config.json').write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding='utf-8')

if 'batch_size' in best:
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
    (out / 'recommended_overrides.txt').write_text("\n".join(rec) + "\n", encoding='utf-8')

log = out / 'one_epoch_verify' / 'train.log'
text = log.read_text(encoding='utf-8', errors='ignore') if log.exists() else ''

# ?? tqdm ????????? 458/600 [33:48<10:28, ...]
matches = re.findall(r"(\d+)/(\d+)\s*\[(\d+:\d+(?::\d+)?)<", text)

def parse_elapsed(s: str) -> float:
    parts = [int(x) for x in s.split(':')]
    if len(parts) == 2:
        m, sec = parts
        return m * 60 + sec
    if len(parts) == 3:
        h, m, sec = parts
        return h * 3600 + m * 60 + sec
    return 0.0

verify = {
    'status': 'estimated_from_partial_run',
    'source': 'option1_no_rerun',
    'train_log': str(log),
}

if matches and 'batch_size' in best:
    step_s, total_s, elapsed_s = matches[-1]
    step = int(step_s)
    total = int(total_s)
    elapsed_sec = parse_elapsed(elapsed_s)
    mean_step = (elapsed_sec / step) if step > 0 else None
    epoch_sec = (elapsed_sec * total / step) if step > 0 else None
    sps = (best['batch_size'] / mean_step) if mean_step and mean_step > 0 else None
    verify.update({
        'completed_steps': step,
        'total_steps': total,
        'elapsed_sec_at_interrupt': elapsed_sec,
        'mean_step_sec_est': mean_step,
        'samples_per_sec_est': sps,
        'one_epoch_time_sec_est': epoch_sec,
        'one_epoch_time_min_est': (epoch_sec / 60.0) if epoch_sec else None,
        'batch_size': best['batch_size'],
        'num_workers': best['num_workers'],
        'precision': best['precision'],
        'pin_memory': True,
        'persistent_workers': True,
        'prefetch_factor': 4,
    })
else:
    verify['reason'] = 'cannot parse progress from one_epoch_verify/train.log'

(out / 'one_epoch_verify.json').write_text(json.dumps(verify, indent=2, ensure_ascii=False), encoding='utf-8')
(out / 'one_epoch_verify' / 'one_epoch_verify.json').write_text(json.dumps(verify, indent=2, ensure_ascii=False), encoding='utf-8')

final_decision = 'READY' if ('samples_per_sec_est' in verify and verify.get('samples_per_sec_est')) else 'BLOCKED'
summary = [
    '# THROUGHPUT_TUNING_SUMMARY',
    f'- FINAL_DECISION: {final_decision}',
]
if 'batch_size' in best:
    summary += [
        f"- selected_precision: {best['precision']}",
        f"- selected_batch_size: {best['batch_size']}",
        f"- selected_num_workers: {best['num_workers']}",
        '- selected_pin_memory/persistent_workers/prefetch_factor: true/true/4',
        f"- best_samples_per_sec: {best.get('samples_per_sec')}",
    ]
if verify.get('one_epoch_time_min_est') is not None:
    summary += [
        f"- one_epoch_time_min (estimated): {verify['one_epoch_time_min_est']}",
        f"- samples_per_sec (estimated): {verify['samples_per_sec_est']}",
        f"- based_on_steps: {verify['completed_steps']}/{verify['total_steps']}",
    ]
summary += ['- note: option1 used partial-run estimation, no restart from step 0']
(out / 'THROUGHPUT_TUNING_SUMMARY.md').write_text('\n'.join(summary) + '\n', encoding='utf-8')

print(json.dumps({'final_decision': final_decision, 'best': best, 'verify': verify}, ensure_ascii=False))
PY
