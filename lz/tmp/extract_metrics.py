import argparse
import json
import re
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', required=True)
    parser.add_argument('--exitcode', required=False)
    parser.add_argument('--out', required=True)
    parser.add_argument('--metric_key', default='val/miou')
    parser.add_argument('--loss_key', default='train/loss_step')
    args = parser.parse_args()

    log_path = Path(args.log)
    exit_path = Path(args.exitcode) if args.exitcode else None
    out_path = Path(args.out)

    log_text = log_path.read_text(encoding='utf-8', errors='ignore') if log_path.exists() else ''

    exitcode = None
    if exit_path and exit_path.exists():
        try:
            exitcode = int(exit_path.read_text(encoding='utf-8').strip())
        except Exception:
            exitcode = None

    metric_pat = re.compile(re.escape(args.metric_key) + r"[^0-9\-]*([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)")
    loss_pat = re.compile(re.escape(args.loss_key) + r"[^0-9\-]*([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)")

    metric_vals = [float(x) for x in metric_pat.findall(log_text)]
    loss_vals = [float(x) for x in loss_pat.findall(log_text)]

    obj = {
        'log': str(log_path),
        'exitcode': exitcode,
        'best_val_miou': max(metric_vals) if metric_vals else None,
        'last_val_miou': metric_vals[-1] if metric_vals else None,
        'first_loss_step': loss_vals[0] if loss_vals else None,
        'last_loss_step': loss_vals[-1] if loss_vals else None,
        'num_val_miou_points': len(metric_vals),
        'num_loss_step_points': len(loss_vals),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(obj, indent=2), encoding='utf-8')
    print(json.dumps(obj, indent=2))

if __name__ == '__main__':
    main()
