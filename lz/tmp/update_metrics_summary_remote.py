from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

main_target = Path('/root/lz_outputs/diag_rootcause_gpu_20260211_225210/metrics_summary.txt')
backup_target = Path('/root/lz_outputs/phase2_metrics_20260213_0D_full/metrics_summary.txt')

m0c = json.loads(Path('/root/lz_outputs/phase2_metrics_20260213_0C/metrics_0C.json').read_text(encoding='utf-8'))['metrics']
m0d = json.loads(Path('/root/lz_outputs/phase2_metrics_20260213_0D_full/metrics_0D.json').read_text(encoding='utf-8'))['metrics']


def val(m: dict, key: str) -> float:
    return float(m[key]['last']) if m[key]['last'] is not None else float('nan')

rows = {
    '0-C': {
        'val/miou_present_gt': val(m0c, 'val/miou_present_gt'),
        'val/miou_present_union': val(m0c, 'val/miou_present_union'),
        'val/miou_fg_mosaic': val(m0c, 'val/miou_fg_mosaic'),
        'val/miou': val(m0c, 'val/miou'),
    },
    '0-D': {
        'val/miou_present_gt': val(m0d, 'val/miou_present_gt'),
        'val/miou_present_union': val(m0d, 'val/miou_present_union'),
        'val/miou_fg_mosaic': val(m0d, 'val/miou_fg_mosaic'),
        'val/miou': val(m0d, 'val/miou'),
    },
}

check_0c = rows['0-C']['val/miou_present_gt'] >= rows['0-C']['val/miou_present_union'] >= rows['0-C']['val/miou']
check_0d = rows['0-D']['val/miou_present_gt'] >= rows['0-D']['val/miou_present_union'] >= rows['0-D']['val/miou']

block = []
block.append(f"=== phase2_metric_rerun_{dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
block.append('run_0C=/root/lz_outputs/phase2_metrics_20260213_0C')
block.append('run_0D=/root/lz_outputs/phase2_metrics_20260213_0D_full')
block.append('format: run | val/miou_present_gt | val/miou_present_union | val/miou_fg_mosaic | val/miou')
for run in ('0-C', '0-D'):
    r = rows[run]
    block.append(
        f"{run} | {r['val/miou_present_gt']:.10f} | {r['val/miou_present_union']:.10f} | {r['val/miou_fg_mosaic']:.10f} | {r['val/miou']:.10f}"
    )
block.append(f"self_check_0C(miou_present_gt>=miou_present_union>=miou)={check_0c}")
block.append(f"self_check_0D(miou_present_gt>=miou_present_union>=miou)={check_0d}")
block.append('')
text = '\n'.join(block)

for target in (main_target, backup_target):
    target.parent.mkdir(parents=True, exist_ok=True)
    prev = target.read_text(encoding='utf-8') if target.exists() else ''
    if prev and not prev.endswith('\n'):
        prev += '\n'
    target.write_text(prev + text + '\n', encoding='utf-8')
    print(f'updated: {target}')

print('\n'.join(block))
