import json
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

base = Path('/root/lz_outputs')
runA = base / 'phase7A_recheck_debug1_aligned'
runB = base / 'phase7A_recheck_debug1_in6'

metric_keys = ['val/miou', 'val/miou_present_gt', 'val/miou_present_union', 'val/miou_fg_mosaic']


def select_event(run_dir: Path):
    versions = sorted((run_dir / 'logs' / 'lz').glob('version_*'))
    best = None
    for ver in versions:
        event_files = sorted(ver.glob('events.out.tfevents.*'))
        if not event_files:
            continue
        event_file = event_files[0]
        ea = event_accumulator.EventAccumulator(str(event_file))
        ea.Reload()
        tags = set(ea.Tags().get('scalars', []))
        if 'val/miou' not in tags:
            continue
        count = len(ea.Scalars('val/miou'))
        if best is None or count > best['count']:
            best = {'version': ver.name, 'file': str(event_file), 'ea': ea, 'count': count}
    if best is None:
        raise RuntimeError(f'No valid event found for {run_dir}')
    return best


def best_last(ea, key: str):
    vals = ea.Scalars(key)
    if not vals:
        return {'best': None, 'last': None, 'count': 0}
    return {
        'best': float(max(v.value for v in vals)),
        'last': float(vals[-1].value),
        'count': int(len(vals)),
    }


def summarize_run(run_dir: Path):
    ev = select_event(run_dir)
    ea = ev['ea']
    metrics = {k: best_last(ea, k) for k in metric_keys}
    collapse = {
        'debug/use_text_guidance': best_last(ea, 'debug/use_text_guidance'),
        'val_pred_unique_count_0': best_last(ea, 'debug/val_pred_unique_count_0'),
        'val_pred_top1_ratio_0': best_last(ea, 'debug/val_pred_top1_ratio_0'),
    }
    (run_dir / 'metrics.json').write_text(
        json.dumps(
            {
                'event_version': ev['version'],
                'event_file': ev['file'],
                **metrics,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    (run_dir / 'collapse_stats.json').write_text(
        json.dumps(
            {
                'event_version': ev['version'],
                'event_file': ev['file'],
                **collapse,
            },
            indent=2,
        ),
        encoding='utf-8',
    )
    use_tg_zero = bool(
        collapse['debug/use_text_guidance']['best'] == 0.0
        and collapse['debug/use_text_guidance']['last'] == 0.0
    )
    return {
        'output_dir': str(run_dir),
        'event_version': ev['version'],
        'event_file': ev['file'],
        'metrics': metrics,
        'collapse': collapse,
        'use_text_guidance_all_zero': use_tg_zero,
    }

runA_info = summarize_run(runA)
runB_info = summarize_run(runB)

best_present_gt = max(
    runA_info['metrics']['val/miou_present_gt']['best'] or 0.0,
    runB_info['metrics']['val/miou_present_gt']['best'] or 0.0,
)

if best_present_gt >= 0.30:
    final_decision = 'PASS'
elif best_present_gt >= 0.15:
    final_decision = 'GRAY'
else:
    final_decision = 'FAIL'

compare = {
    'runA': runA_info,
    'runB_triggered': True,
    'runB': runB_info,
    'final_decision': final_decision,
    'decision_basis_best_val_miou_present_gt': float(best_present_gt),
}
(base / 'phase7A_recheck_compare.json').write_text(json.dumps(compare, indent=2), encoding='utf-8')

summary_lines = [
    '# PHASE7A RECHECK SUMMARY',
    '',
    f'- Conclusion: {final_decision}',
    f"- RunA best val/miou_present_gt: {runA_info['metrics']['val/miou_present_gt']['best']:.10f}",
    f"- RunB best val/miou_present_gt: {runB_info['metrics']['val/miou_present_gt']['best']:.10f}",
    f"- RunA use_text_guidance all zero: {runA_info['use_text_guidance_all_zero']}",
    f"- RunB use_text_guidance all zero: {runB_info['use_text_guidance_all_zero']}",
    '',
    '## Next steps',
    '1. If staying with SpUNet, first fix collapse/label-learning path before any full split run.',
    '2. Keep only one variant for next gate (in_channels=3 or 6) based on better miou_present_gt.',
]
(base / 'PHASE7A_RECHECK_SUMMARY.md').write_text('\n'.join(summary_lines), encoding='utf-8')

print(final_decision)
print(runA_info['metrics']['val/miou_present_gt']['best'])
print(runB_info['metrics']['val/miou_present_gt']['best'])
