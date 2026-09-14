import json
from pathlib import Path
from typing import Dict, Any, Tuple

from tensorboard.backend.event_processing import event_accumulator

BASE = Path('/root/lz_outputs/phase7D_full15_bn_abc')
RUNS = {
    'A': BASE / 'A_baseline',
    'B': BASE / 'B_bn_batch_stats',
    'C': BASE / 'C_bn_recalib_eval',
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
    best = None
    for version_dir in sorted((run_dir / 'logs' / 'lz').glob('version_*')):
        for event_file in sorted(version_dir.glob('events.out.tfevents.*')):
            ea = event_accumulator.EventAccumulator(str(event_file))
            try:
                ea.Reload()
            except Exception:
                continue
            val_count = _scalar_stats(ea, 'val/miou')['count']
            score = (val_count, version_dir.name)
            if best is None or score > best['score']:
                best = {
                    'score': score,
                    'event_version': version_dir.name,
                    'event_file': str(event_file),
                    'ea': ea,
                }
    if best is None or best['score'][0] == 0:
        raise RuntimeError(f'No usable event file for {run_dir}')
    return best


def _read_overrides(run_dir: Path) -> Dict[str, str]:
    path = run_dir / '.hydra' / 'overrides.yaml'
    out: Dict[str, str] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding='utf-8').splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('- '):
            line = line[2:].strip()
        if '=' not in line:
            out[line] = ''
            continue
        key, value = line.split('=', 1)
        out[key.strip()] = value.strip()
    return out


def _write_run_files(run_key: str, run_dir: Path) -> Dict[str, Any]:
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

    tg = collapse.get('debug/use_text_guidance', {'best': None, 'last': None})
    use_tg_all_zero = bool(tg.get('best') == 0.0 and tg.get('last') == 0.0)

    return {
        'run_key': run_key,
        'run_dir': str(run_dir),
        'event_version': selected['event_version'],
        'event_file': selected['event_file'],
        'metrics': metrics,
        'collapse': collapse,
        'use_text_guidance_all_zero': use_tg_all_zero,
        'overrides': _read_overrides(run_dir),
    }


def _metric_best(run_info: Dict[str, Any], key: str) -> float:
    value = run_info['metrics'][key]['best']
    return float(value if value is not None else 0.0)


def _metric_last(run_info: Dict[str, Any], key: str) -> float:
    value = run_info['metrics'][key]['last']
    return float(value if value is not None else 0.0)


def _diff_keys(a: Dict[str, str], b: Dict[str, str]) -> Dict[str, Tuple[str, str]]:
    keys = set(a.keys()) | set(b.keys())
    out = {}
    for k in keys:
        av = a.get(k)
        bv = b.get(k)
        if av != bv:
            out[k] = (av, bv)
    return out


runA = _write_run_files('A', RUNS['A'])
runB = _write_run_files('B', RUNS['B'])
runC = _write_run_files('C', RUNS['C'])

metric_deltas = {}
for key in METRIC_KEYS:
    metric_deltas[key] = {
        'A_to_B_best': _metric_best(runB, key) - _metric_best(runA, key),
        'A_to_C_best': _metric_best(runC, key) - _metric_best(runA, key),
        'B_to_C_best': _metric_best(runC, key) - _metric_best(runB, key),
        'A_to_B_last': _metric_last(runB, key) - _metric_last(runA, key),
        'A_to_C_last': _metric_last(runC, key) - _metric_last(runA, key),
        'B_to_C_last': _metric_last(runC, key) - _metric_last(runB, key),
    }

ab_diff = _diff_keys(runA['overrides'], runB['overrides'])
allowed_ab_diff = {'paths.output_dir', '+model.eval_cfg.eval_bn_batch_stats'}
ab_only_bn_switch = set(ab_diff.keys()).issubset(allowed_ab_diff)
ab_same_ckpt = runA['overrides'].get('ckpt_path') == runB['overrides'].get('ckpt_path')

full15_consistent = True
for r in [runA, runB, runC]:
    ov = r['overrides']
    if ov.get('data.val_datasets.0.split') != 'val':
        full15_consistent = False
    if ov.get('data.train_dataset.split') != 'train':
        full15_consistent = False

use_tg_all_zero = bool(
    runA['use_text_guidance_all_zero']
    and runB['use_text_guidance_all_zero']
    and runC['use_text_guidance_all_zero']
)

# Decision based on primary metric val/miou_fg_mosaic (best)
a_fg = _metric_best(runA, 'val/miou_fg_mosaic')
b_fg = _metric_best(runB, 'val/miou_fg_mosaic')
c_fg = _metric_best(runC, 'val/miou_fg_mosaic')

close_to_b = (b_fg > 0 and c_fg >= 0.9 * b_fg) or (abs(b_fg - c_fg) <= max(0.0002, 0.1 * max(abs(b_fg), 1e-6)))
improve_over_a = (c_fg - a_fg) >= max(0.0005, 0.3 * max(abs(a_fg), 1e-6))
below_b = (c_fg < 0.9 * b_fg) and ((b_fg - c_fg) > max(0.0002, 0.1 * max(abs(b_fg), 1e-6)))
all_low = max(a_fg, b_fg, c_fg) < 0.002
small_spread = (max(a_fg, b_fg, c_fg) - min(a_fg, b_fg, c_fg)) <= 0.0005

if all_low and small_spread:
    final_decision = 'BN_POLICY_FAIL'
elif close_to_b and (c_fg > a_fg):
    final_decision = 'BN_POLICY_READY'
elif improve_over_a and below_b:
    final_decision = 'BN_POLICY_PARTIAL'
else:
    final_decision = 'BN_POLICY_FAIL'

compare_payload = {
    'runs': {'A': runA, 'B': runB, 'C': runC},
    'metric_deltas': metric_deltas,
    'consistency_checks': {
        'same_ckpt_A_B': ab_same_ckpt,
        'same_full15_split': full15_consistent,
        'ab_only_bn_switch_diff': ab_only_bn_switch,
        'use_text_guidance_all_zero': use_tg_all_zero,
        'ab_override_diff_keys': sorted(ab_diff.keys()),
    },
    'decision_basis': {
        'primary_metric': 'val/miou_fg_mosaic',
        'A_best': a_fg,
        'B_best': b_fg,
        'C_best': c_fg,
        'close_to_B': close_to_b,
        'improve_over_A': improve_over_a,
        'below_B': below_b,
        'all_low': all_low,
        'small_spread': small_spread,
    },
    'final_decision': final_decision,
}

(BASE / 'phase7D_bn_abc_compare.json').write_text(
    json.dumps(compare_payload, indent=2), encoding='utf-8'
)

summary_lines = [
    '# PHASE7D BN ABC SUMMARY',
    '',
    f'- Final decision: {final_decision}',
    f'- Primary metric val/miou_fg_mosaic (best): A={a_fg:.10f}, B={b_fg:.10f}, C={c_fg:.10f}',
    f"- A->B delta: {metric_deltas['val/miou_fg_mosaic']['A_to_B_best']:.10f}",
    f"- A->C delta: {metric_deltas['val/miou_fg_mosaic']['A_to_C_best']:.10f}",
    f"- B->C delta: {metric_deltas['val/miou_fg_mosaic']['B_to_C_best']:.10f}",
    f'- same_ckpt_A_B: {ab_same_ckpt}',
    f'- same_full15_split: {full15_consistent}',
    f'- A/B only BN switch diff: {ab_only_bn_switch}',
    f'- use_text_guidance all zero: {use_tg_all_zero}',
]

(BASE / 'PHASE7D_SUMMARY.md').write_text('\n'.join(summary_lines), encoding='utf-8')

print(final_decision)
print(json.dumps(compare_payload['decision_basis'], indent=2))
