import json
from pathlib import Path
from typing import Any, Dict, List

from tensorboard.backend.event_processing import event_accumulator

BASE = Path('/root/lz_outputs/phase8B_spunet_bn_recalib')
TRAIN_DIR = BASE / 'train_full15_ep15'
EVAL_A_DIR = BASE / 'eval_A_baseline'
EVAL_B_DIR = BASE / 'eval_B_recalib'

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
    best = None
    for vdir in sorted((run_dir / 'logs' / 'lz').glob('version_*')):
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

    (run_dir / 'metrics.json').write_text(
        json.dumps({'event_version': selected['event_version'], 'event_file': selected['event_file'], **metrics}, indent=2),
        encoding='utf-8',
    )
    (run_dir / 'collapse_stats.json').write_text(
        json.dumps({'event_version': selected['event_version'], 'event_file': selected['event_file'], **collapse}, indent=2),
        encoding='utf-8',
    )

    tg = collapse['debug/use_text_guidance']
    use_tg_all_zero = bool(tg['best'] == 0.0 and tg['last'] == 0.0)

    return {
        'executed': True,
        'run_dir': str(run_dir),
        'event_version': selected['event_version'],
        'event_file': selected['event_file'],
        'metrics': metrics,
        'collapse': collapse,
        'use_text_guidance_all_zero': use_tg_all_zero,
        'trend_last5_val_miou_fg_mosaic': _trend_last5(metric_series['val/miou_fg_mosaic']),
        'overrides': _read_overrides(run_dir),
    }


def _best(run: Dict[str, Any], key: str) -> float:
    if not run.get('executed'):
        return 0.0
    val = run['metrics'][key]['best']
    return float(val if val is not None else 0.0)


def main() -> None:
    BASE.mkdir(parents=True, exist_ok=True)

    smoke_path = BASE / 'smoke_check.json'
    smoke = json.loads(smoke_path.read_text(encoding='utf-8')) if smoke_path.exists() else None

    train = summarize_run(TRAIN_DIR)
    eval_a = summarize_run(EVAL_A_DIR)
    eval_b = summarize_run(EVAL_B_DIR)

    use_tg_all_zero = bool(
        train.get('use_text_guidance_all_zero', False)
        and eval_a.get('use_text_guidance_all_zero', False)
        and eval_b.get('use_text_guidance_all_zero', False)
    )

    train_trend = train.get('trend_last5_val_miou_fg_mosaic', {})
    trend_ok = True
    if train_trend.get('available'):
        slope = train_trend.get('slope', 0.0)
        delta = train_trend.get('delta', 0.0)
        trend_ok = bool(slope >= -1e-5 and delta >= -1e-4)

    eval_a_fg = _best(eval_a, 'val/miou_fg_mosaic')
    eval_b_fg = _best(eval_b, 'val/miou_fg_mosaic')

    final_decision = 'CONTINUE_SPUNET' if (eval_b_fg >= 0.01 and trend_ok) else 'SWITCH_TO_PTV3_NOW'

    compare = {
        'smoke': smoke,
        'train': train,
        'eval_A_baseline': eval_a,
        'eval_B_recalib': eval_b,
        'delta_A_to_B': {
            key: _best(eval_b, key) - _best(eval_a, key) for key in METRIC_KEYS
        },
        'use_text_guidance_all_zero': use_tg_all_zero,
        'final_decision': final_decision,
    }

    (BASE / 'phase8B_compare.json').write_text(json.dumps(compare, indent=2), encoding='utf-8')

    summary_lines = [
        '# PHASE8B SUMMARY',
        '',
        f'- Final decision: {final_decision}',
        f'- Smoke success: {bool(smoke and smoke.get("forward_backward_success"))}',
        f'- Train best val/miou_fg_mosaic: {_best(train, "val/miou_fg_mosaic"):.10f}',
        f'- Eval A best val/miou_fg_mosaic: {eval_a_fg:.10f}',
        f'- Eval B best val/miou_fg_mosaic: {eval_b_fg:.10f}',
        f'- Eval A->B delta val/miou_fg_mosaic: {eval_b_fg - eval_a_fg:.10f}',
        f'- use_text_guidance all zero: {use_tg_all_zero}',
        f'- Train trend last5 val/miou_fg_mosaic: {train_trend}',
    ]
    (BASE / 'PHASE8B_SUMMARY.md').write_text('\n'.join(summary_lines), encoding='utf-8')

    print(final_decision)
    print(json.dumps({'eval_a_fg': eval_a_fg, 'eval_b_fg': eval_b_fg, 'trend_ok': trend_ok}, indent=2))


if __name__ == '__main__':
    main()
