import json
from pathlib import Path
from tensorboard.backend.event_processing import event_accumulator

root = Path('/root/lz_outputs')
smoke_dir = root / 'phase7A_smoke'
gate_dir = root / 'phase7A_gate_debug1_60'

smoke_metrics = {}
smoke_file = smoke_dir / 'smoke_metrics.json'
if smoke_file.exists():
    smoke_metrics = json.loads(smoke_file.read_text())

logs_dir = gate_dir / 'logs' / 'lz'
versions = sorted(logs_dir.glob('version_*'))
chosen = None
for ver in reversed(versions):
    events = sorted(ver.glob('events.out.tfevents.*'))
    if not events:
        continue
    ea = event_accumulator.EventAccumulator(str(events[0]))
    ea.Reload()
    tags = ea.Tags().get('scalars', [])
    if 'val/miou' in tags and len(ea.Scalars('val/miou')) > 0:
        chosen = (ver, events[0], ea)
        break

if chosen is None:
    raise RuntimeError('No valid scalar events found for phase7A_gate_debug1_60')

ver, event_file, ea = chosen

def best_last(key: str):
    vals = ea.Scalars(key)
    if not vals:
        return {'best': None, 'last': None, 'count': 0}
    return {
        'best': float(max(v.value for v in vals)),
        'last': float(vals[-1].value),
        'count': int(len(vals)),
    }

metrics = {
    'val/miou': best_last('val/miou'),
    'val/miou_present_gt': best_last('val/miou_present_gt'),
    'val/miou_present_union': best_last('val/miou_present_union'),
    'val/miou_fg_mosaic': best_last('val/miou_fg_mosaic'),
}

use_text = best_last('debug/use_text_guidance')
collapse = {
    'val_pred_unique_count_0': best_last('debug/val_pred_unique_count_0'),
    'val_pred_top1_ratio_0': best_last('debug/val_pred_top1_ratio_0'),
}

best_gt = metrics['val/miou_present_gt']['best'] or 0.0
if best_gt >= 0.50:
    final_decision = 'PASS'
elif best_gt >= 0.30:
    final_decision = 'GRAY'
else:
    final_decision = 'FAIL'

compare = {
    'step_resumed_from': 'Step4',
    'dependency_status': {
        'spconv': True,
        'torch_scatter': True,
        'timm': True,
        'addict': True,
        'jaxtyping': True,
    },
    'smoke': {
        'output_dir': str(smoke_dir),
        **smoke_metrics,
    },
    'gate_debug1_60': {
        'executed': True,
        'output_dir': str(gate_dir),
        'event_version': ver.name,
        'event_file': str(event_file),
        'metrics': metrics,
        'debug_use_text_guidance': use_text,
        'collapse': collapse,
        'acbs_log_evidence': 'present_in_train_log',
    },
    'gate_full15': {
        'executed': False,
        'reason': 'Skipped because gate_debug1_60 best val/miou_present_gt < 0.30',
    },
    'debug_use_text_guidance_all_zero': bool((use_text['best'] == 0.0) and (use_text['last'] == 0.0)),
    'collapse_stop_triggered': False,
    'final_decision': final_decision,
}

(root / 'phase7A_compare.json').write_text(json.dumps(compare, indent=2), encoding='utf-8')

summary = (
    '# PHASE7A SUMMARY\n\n'
    f'- Decision: {final_decision}\n'
    '- Resume point: Step4 smoke (Step1/2 not rerun)\n'
    f"- Smoke success: {smoke_metrics.get('forward_backward_success', 0)} "
    f"(batch_size={smoke_metrics.get('used_batch_size', 0)}, fallback={smoke_metrics.get('fallback', 'unknown')})\n"
    '- Gate debug1 best metrics:\n'
    f"  - val/miou={metrics['val/miou']['best']:.10f}\n"
    f"  - val/miou_present_gt={metrics['val/miou_present_gt']['best']:.10f}\n"
    f"  - val/miou_present_union={metrics['val/miou_present_union']['best']:.10f}\n"
    f"  - val/miou_fg_mosaic={metrics['val/miou_fg_mosaic']['best']:.10f}\n"
    f"- debug/use_text_guidance all zero: {compare['debug_use_text_guidance_all_zero']}\n"
    f"- Collapse indicators (last): pred_unique_count_0={collapse['val_pred_unique_count_0']['last']}, "
    f"pred_top1_ratio_0={collapse['val_pred_top1_ratio_0']['last']}\n"
    '- Full15 gate: skipped (debug1 gate already FAIL)\n\n'
    '## Next steps\n'
    '1. Fix collapse first (increase pred diversity, reduce top1 dominance).\n'
    '2. Re-run gate_debug1, only proceed to full15 after GRAY/PASS.\n'
)
(root / 'PHASE7A_SUMMARY.md').write_text(summary, encoding='utf-8')
print('wrote', root / 'phase7A_compare.json')
print('wrote', root / 'PHASE7A_SUMMARY.md')
