import json
from pathlib import Path
from typing import Any, Dict, List

from tensorboard.backend.event_processing import event_accumulator

BASE = Path('/root/lz_outputs/phase8_spunet_gn')
GATE1_DIR = BASE / 'gate1_debug1_60'
GATE2_DIR = BASE / 'gate2_full15_ep15'

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

CODE_CHANGE_FILES = [
    '/root/lz/src/models/networks/spunet/spconv_unet_v1m1_base.py',
    '/root/lz/src/models/networks/lz_backbone/spunet34c_adapter.py',
    '/root/lz/configs/model/spunet34c_lz.yaml',
]


def _ensure_hydra(run_dir: Path) -> None:
    hydra_dir = run_dir / '.hydra'
    if hydra_dir.exists():
        return
    candidates = sorted(run_dir.glob('*/*/.hydra'))
    if not candidates:
        return
    src = candidates[-1]
    hydra_dir.mkdir(parents=True, exist_ok=True)
    for name in ['config.yaml', 'hydra.yaml', 'overrides.yaml']:
        src_file = src / name
        if src_file.exists():
            (hydra_dir / name).write_text(src_file.read_text(encoding='utf-8'), encoding='utf-8')


def _scalar_series(ea: event_accumulator.EventAccumulator, key: str) -> List[float]:
    tags = set(ea.Tags().get('scalars', []))
    if key not in tags:
        return []
    return [float(x.value) for x in ea.Scalars(key)]


def _scalar_stats(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {'best': None, 'last': None, 'count': 0}
    return {
        'best': float(max(values)),
        'last': float(values[-1]),
        'count': int(len(values)),
    }


def _select_event(run_dir: Path) -> Dict[str, Any] | None:
    version_dirs = sorted((run_dir / 'logs' / 'lz').glob('version_*'))
    best = None
    for vdir in version_dirs:
        for event_file in sorted(vdir.glob('events.out.tfevents.*')):
            ea = event_accumulator.EventAccumulator(str(event_file))
            try:
                ea.Reload()
            except Exception:
                continue
            count = len(_scalar_series(ea, 'val/miou'))
            score = (count, vdir.name)
            if best is None or score > best['score']:
                best = {
                    'score': score,
                    'event_version': vdir.name,
                    'event_file': str(event_file),
                    'ea': ea,
                }
    if best is None or best['score'][0] == 0:
        return None
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


def _trend_last5(values: List[float]) -> Dict[str, Any]:
    if len(values) < 5:
        return {'available': False, 'slope': None, 'delta': None, 'values': values}
    last5 = values[-5:]
    x = [0.0, 1.0, 2.0, 3.0, 4.0]
    x_mean = sum(x) / 5.0
    y_mean = sum(last5) / 5.0
    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, last5))
    denominator = sum((xi - x_mean) ** 2 for xi in x)
    slope = numerator / denominator if denominator > 0 else 0.0
    delta = last5[-1] - last5[0]
    return {
        'available': True,
        'slope': float(slope),
        'delta': float(delta),
        'values': [float(v) for v in last5],
    }


def summarize_run(run_dir: Path) -> Dict[str, Any]:
    _ensure_hydra(run_dir)
    selected = _select_event(run_dir)
    if selected is None:
        return {
            'executed': False,
            'run_dir': str(run_dir),
            'reason': 'missing_event_or_val_metrics',
        }

    ea = selected['ea']
    metric_series = {k: _scalar_series(ea, k) for k in METRIC_KEYS}
    collapse_series = {k: _scalar_series(ea, k) for k in COLLAPSE_KEYS}

    metrics = {k: _scalar_stats(v) for k, v in metric_series.items()}
    collapse = {k: _scalar_stats(v) for k, v in collapse_series.items()}

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

    tg = collapse['debug/use_text_guidance']
    use_tg_all_zero = bool(tg['best'] == 0.0 and tg['last'] == 0.0)

    trend = _trend_last5(metric_series['val/miou_fg_mosaic'])

    return {
        'executed': True,
        'run_dir': str(run_dir),
        'event_version': selected['event_version'],
        'event_file': selected['event_file'],
        'metrics': metrics,
        'collapse': collapse,
        'use_text_guidance_all_zero': use_tg_all_zero,
        'trend_last5_val_miou_fg_mosaic': trend,
        'overrides': _read_overrides(run_dir),
    }


def _best(run: Dict[str, Any], key: str) -> float:
    return float(run['metrics'][key]['best'] if run.get('executed') else 0.0)


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)

    gate1 = summarize_run(GATE1_DIR)
    gate2 = summarize_run(GATE2_DIR)

    gate1_best_present_gt = _best(gate1, 'val/miou_present_gt') if gate1.get('executed') else 0.0
    gate1_pass = bool(
        gate1.get('executed')
        and gate1_best_present_gt >= 0.30
        and gate1.get('use_text_guidance_all_zero', False)
    )

    gate2_best_fg = _best(gate2, 'val/miou_fg_mosaic') if gate2.get('executed') else 0.0
    gate2_trend = gate2.get('trend_last5_val_miou_fg_mosaic', {})
    trend_ok = True
    if gate2_trend.get('available'):
        slope = gate2_trend.get('slope', 0.0)
        delta = gate2_trend.get('delta', 0.0)
        trend_ok = bool(slope >= -1e-5 and delta >= -1e-4)

    gate2_pass = bool(gate2.get('executed') and gate2_best_fg >= 0.01 and trend_ok)

    final_decision = 'CONTINUE_SPUNET_GN' if (gate1_pass and gate2_pass) else 'STOP_SPUNET_GN'

    compare = {
        'code_change_files': CODE_CHANGE_FILES,
        'gate1': {
            **gate1,
            'pass_rule': {
                'best_val_miou_present_gt_ge_0_30': gate1_best_present_gt >= 0.30,
                'use_text_guidance_all_zero': gate1.get('use_text_guidance_all_zero', False),
            },
            'passed': gate1_pass,
        },
        'gate2': {
            **gate2,
            'pass_rule': {
                'best_val_miou_fg_mosaic_ge_0_01': gate2_best_fg >= 0.01,
                'trend_last5_not_declining': trend_ok,
            },
            'passed': gate2_pass,
        },
        'final_decision': final_decision,
    }

    (BASE / 'phase8_compare.json').write_text(json.dumps(compare, indent=2), encoding='utf-8')

    summary_lines = [
        '# PHASE8 SUMMARY',
        '',
        f'- Final decision: {final_decision}',
        f'- Gate-1 best val/miou_present_gt: {gate1_best_present_gt:.10f}',
        f"- Gate-1 use_text_guidance all zero: {gate1.get('use_text_guidance_all_zero', False)}",
        f'- Gate-2 executed: {gate2.get("executed", False)}',
        f'- Gate-2 best val/miou_fg_mosaic: {gate2_best_fg:.10f}',
        f"- Gate-2 trend last5: {gate2_trend}",
        '',
        '## Code changes',
        *[f'- {p}' for p in CODE_CHANGE_FILES],
    ]
    (BASE / 'PHASE8_SUMMARY.md').write_text('\n'.join(summary_lines), encoding='utf-8')

    print(final_decision)
    print(json.dumps({'gate1_passed': gate1_pass, 'gate2_passed': gate2_pass}, indent=2))


if __name__ == '__main__':
    main()
