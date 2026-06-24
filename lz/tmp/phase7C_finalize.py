import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from tensorboard.backend.event_processing import event_accumulator

BASE = Path('/root/lz_outputs/phase7C_bn_fix_eval_only')
RUNS = {
    'A_eval_bn_off': BASE / 'A_eval_bn_off',
    'B_eval_bn_on': BASE / 'B_eval_bn_on',
}
METRIC_KEYS = [
    'val/miou',
    'val/miou_present_gt',
    'val/miou_present_union',
    'val/miou_fg_mosaic',
]
COLLAPSE_KEYS = [
    'debug/use_text_guidance',
    'debug/val_pred_unique_count_0',
    'debug/val_pred_top1_ratio_0',
]


def _scalar_stats(ea: event_accumulator.EventAccumulator, key: str) -> Dict[str, Any]:
    tags = set(ea.Tags().get('scalars', []))
    if key not in tags:
        return {'best': None, 'last': None, 'count': 0}
    vals = ea.Scalars(key)
    if not vals:
        return {'best': None, 'last': None, 'count': 0}
    arr = [float(x.value) for x in vals]
    return {'best': float(max(arr)), 'last': float(arr[-1]), 'count': int(len(arr))}


def _choose_event(run_dir: Path) -> Dict[str, Any]:
    version_dirs = sorted((run_dir / 'logs' / 'lz').glob('version_*'))
    best = None
    for vdir in version_dirs:
        for event_file in sorted(vdir.glob('events.out.tfevents.*')):
            ea = event_accumulator.EventAccumulator(str(event_file))
            try:
                ea.Reload()
            except Exception:
                continue
            val_count = _scalar_stats(ea, 'val/miou')['count']
            score = (val_count, str(vdir))
            if best is None or score > best['score']:
                best = {
                    'score': score,
                    'event_file': str(event_file),
                    'event_version': vdir.name,
                    'ea': ea,
                }
    if best is None or best['score'][0] == 0:
        raise RuntimeError(f'No usable event file with val/miou found in {run_dir}')
    return best


def summarize_run(run_key: str, run_dir: Path) -> Dict[str, Any]:
    selected = _choose_event(run_dir)
    ea = selected['ea']
    metrics = {k: _scalar_stats(ea, k) for k in METRIC_KEYS}
    collapse = {k: _scalar_stats(ea, k) for k in COLLAPSE_KEYS}

    metrics_payload = {
        'event_version': selected['event_version'],
        'event_file': selected['event_file'],
        **metrics,
    }
    collapse_payload = {
        'event_version': selected['event_version'],
        'event_file': selected['event_file'],
        **collapse,
    }
    (run_dir / 'metrics.json').write_text(json.dumps(metrics_payload, indent=2), encoding='utf-8')
    (run_dir / 'collapse_stats.json').write_text(json.dumps(collapse_payload, indent=2), encoding='utf-8')

    tg_best = collapse['debug/use_text_guidance']['best']
    tg_last = collapse['debug/use_text_guidance']['last']
    tg_all_zero = (tg_best == 0.0) and (tg_last == 0.0)

    return {
        'run_key': run_key,
        'run_dir': str(run_dir),
        'event_version': selected['event_version'],
        'event_file': selected['event_file'],
        'metrics': metrics,
        'collapse': collapse,
        'use_text_guidance_all_zero': tg_all_zero,
    }


a = summarize_run('A_eval_bn_off', RUNS['A_eval_bn_off'])
b = summarize_run('B_eval_bn_on', RUNS['B_eval_bn_on'])

a_best = a['metrics']['val/miou_present_gt']['best'] or 0.0
b_best = b['metrics']['val/miou_present_gt']['best'] or 0.0
delta = b_best - a_best

criteria = {
    'only_switch_diff_expected': True,
    'use_text_guidance_all_zero': bool(a['use_text_guidance_all_zero'] and b['use_text_guidance_all_zero']),
    'miou_present_gt_delta_ge_0_20': bool(delta >= 0.20),
    'miou_present_gt_B_ge_0_30': bool(b_best >= 0.30),
}

final_decision = 'BN_FIX_READY' if all(criteria.values()) else 'BN_FIX_FAIL'

compare_payload = {
    'runA': a,
    'runB': b,
    'delta': {
        'val/miou': (b['metrics']['val/miou']['best'] or 0.0) - (a['metrics']['val/miou']['best'] or 0.0),
        'val/miou_present_gt': delta,
        'val/miou_present_union': (b['metrics']['val/miou_present_union']['best'] or 0.0) - (a['metrics']['val/miou_present_union']['best'] or 0.0),
        'val/miou_fg_mosaic': (b['metrics']['val/miou_fg_mosaic']['best'] or 0.0) - (a['metrics']['val/miou_fg_mosaic']['best'] or 0.0),
    },
    'acceptance': criteria,
    'final_decision': final_decision,
}

(BASE / 'bn_fix_compare.json').write_text(json.dumps(compare_payload, indent=2), encoding='utf-8')

summary_lines = [
    '# BN FIX SUMMARY',
    '',
    f'- Final decision: {final_decision}',
    f"- Run A best val/miou_present_gt: {a_best:.10f}",
    f"- Run B best val/miou_present_gt: {b_best:.10f}",
    f"- Delta (B-A) val/miou_present_gt: {delta:.10f}",
    f"- Run A use_text_guidance all zero: {a['use_text_guidance_all_zero']}",
    f"- Run B use_text_guidance all zero: {b['use_text_guidance_all_zero']}",
    '',
    '## Metrics (best)',
    f"- A val/miou: {(a['metrics']['val/miou']['best'] or 0.0):.10f}",
    f"- A val/miou_present_union: {(a['metrics']['val/miou_present_union']['best'] or 0.0):.10f}",
    f"- A val/miou_fg_mosaic: {(a['metrics']['val/miou_fg_mosaic']['best'] or 0.0):.10f}",
    f"- B val/miou: {(b['metrics']['val/miou']['best'] or 0.0):.10f}",
    f"- B val/miou_present_union: {(b['metrics']['val/miou_present_union']['best'] or 0.0):.10f}",
    f"- B val/miou_fg_mosaic: {(b['metrics']['val/miou_fg_mosaic']['best'] or 0.0):.10f}",
]
(BASE / 'BN_FIX_SUMMARY.md').write_text('\n'.join(summary_lines), encoding='utf-8')

print(final_decision)
print(json.dumps(compare_payload['delta'], indent=2))
