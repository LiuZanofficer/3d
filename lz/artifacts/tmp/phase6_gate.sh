#!/usr/bin/env bash
set -euo pipefail

PY=/root/autodl-tmp/conda-envs/llz/bin/python
ROOT=/root/lz
DATA=/root/autodl-tmp/datasets/mosaic3d/data

RUN_A=/root/lz_outputs/phase6_gate_A_coordnorm_ce60
RUN_B=/root/lz_outputs/phase6_gate_B_lovasz30
COMPARE_JSON=/root/lz_outputs/phase6_gate_compare.json
SUMMARY_MD=/root/lz_outputs/PHASE6_GATE_SUMMARY.md

extract_run_outputs() {
  local run_dir="$1"
  "$PY" - "$run_dir" <<'PY'
import glob
import json
import re
import sys
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

run_dir = Path(sys.argv[1])
event_files = sorted(glob.glob(str(run_dir / "logs" / "lz" / "version_*" / "events.out.tfevents.*")))
if not event_files:
    raise FileNotFoundError(f"No tensorboard events found under {run_dir}")
event_file = event_files[-1]
ea = event_accumulator.EventAccumulator(event_file)
ea.Reload()
tags = set(ea.Tags().get("scalars", []))

metric_tags = [
    "val/miou",
    "val/miou_present_gt",
    "val/miou_present_union",
    "val/miou_fg_mosaic",
]
collapse_tags = [
    "debug/use_text_guidance_epoch",
    "debug/use_text_guidance_0",
    "debug/train_pred_unique_count_epoch",
    "debug/train_pred_top1_ratio_epoch",
    "debug/val_pred_unique_count_0",
    "debug/val_pred_top1_ratio_0",
]

def series(tag):
    if tag not in tags:
        return []
    return [{"step": int(v.step), "value": float(v.value)} for v in ea.Scalars(tag)]

def summary(items):
    if not items:
        return {"count": 0, "best": None, "last": None, "min": None}
    values = [x["value"] for x in items]
    return {
        "count": len(values),
        "best": float(max(values)),
        "last": float(values[-1]),
        "min": float(min(values)),
    }

metric_series = {k: series(k) for k in metric_tags}
metric_summary = {k: summary(v) for k, v in metric_series.items()}
collapse_series = {k: series(k) for k in collapse_tags}
collapse_summary = {k: summary(v) for k, v in collapse_series.items()}

overrides = []
override_files = sorted(run_dir.glob("*/*/.hydra/overrides.yaml"))
if override_files:
    for line in override_files[-1].read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("- "):
            overrides.append(line[2:].strip())

log_path = run_dir / "train.log"
acbs = []
if log_path.exists():
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    seen = set()
    for m in re.finditer(r"ACBS epoch=(\d+) temperature=([0-9.]+)", text):
        rec = (int(m.group(1)), float(m.group(2)))
        if rec in seen:
            continue
        seen.add(rec)
        acbs.append({"epoch": rec[0], "temperature": rec[1]})

use_text_values = []
for key in ("debug/use_text_guidance_epoch", "debug/use_text_guidance_0"):
    use_text_values.extend([x["value"] for x in collapse_series.get(key, [])])
use_text_zero = all(abs(v) <= 1e-12 for v in use_text_values) if use_text_values else True

metrics_payload = {
    "run_dir": str(run_dir),
    "event_file": event_file,
    "metric_summary": metric_summary,
    "overrides": overrides,
    "acbs": acbs,
}
collapse_payload = {
    "run_dir": str(run_dir),
    "event_file": event_file,
    "summary": collapse_summary,
    "epoch_series": collapse_series,
    "use_text_guidance_all_zero": bool(use_text_zero),
}

(run_dir / "metrics.json").write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")
(run_dir / "collapse_stats.json").write_text(json.dumps(collapse_payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"[OK] wrote {run_dir / 'metrics.json'}")
print(f"[OK] wrote {run_dir / 'collapse_stats.json'}")
PY
}

run_train() {
  local name="$1"
  local out_dir="$2"
  local max_epochs="$3"
  local lovasz="$4"

  echo "[PHASE6] ===== ${name} start $(date '+%F %T') ====="
  mkdir -p "$out_dir"
  cd "$ROOT"
  set +e
  "$PY" src/train.py \
    data=sc_smoke \
    seed=42 \
    paths.data_dir="$DATA" \
    paths.root_dir="$ROOT" \
    paths.output_dir="$out_dir" \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.max_epochs="$max_epochs" \
    +trainer.max_steps=2000 \
    data.train_dataset.split=debug1 \
    data.val_datasets.0.split=debug1 \
    model.loss.weights.seg_loss=1.0 \
    model.loss.weights.instance_loss=0.0 \
    model.loss.weights.hpza_loss=0.0 \
    model.loss.seg_loss.lovasz_weight="$lovasz" \
    model.loss.seg_loss.use_class_weight=false \
    data.coord_norm_enable=true \
    data.color_drop_prob=0.0 \
    data.color_drop_mode=gray \
    > "$out_dir/train.log" 2>&1
  local exit_code=$?
  set -e
  echo "$exit_code" > "$out_dir/train.exitcode"
  if [[ "$exit_code" != "0" ]]; then
    echo "[PHASE6] ${name} failed with exit code=$exit_code"
    tail -n 120 "$out_dir/train.log" || true
    exit "$exit_code"
  fi
  extract_run_outputs "$out_dir"
  echo "[PHASE6] ===== ${name} done $(date '+%F %T') ====="
}

best_present_gt() {
  local run_dir="$1"
  "$PY" - "$run_dir" <<'PY'
import json
import sys
from pathlib import Path
metric_path = Path(sys.argv[1]) / "metrics.json"
payload = json.loads(metric_path.read_text(encoding="utf-8"))
best = payload["metric_summary"]["val/miou_present_gt"]["best"]
print(best if best is not None else 0.0)
PY
}

build_final_reports() {
  local run_a="$1"
  local run_b="$2"
  "$PY" - "$run_a" "$run_b" "$COMPARE_JSON" "$SUMMARY_MD" <<'PY'
import json
import sys
from pathlib import Path

run_a = Path(sys.argv[1])
run_b_arg = sys.argv[2]
run_b = Path(run_b_arg) if run_b_arg != "__NONE__" else None
compare_path = Path(sys.argv[3])
summary_path = Path(sys.argv[4])

metric_tags = ["val/miou", "val/miou_present_gt", "val/miou_present_union", "val/miou_fg_mosaic"]

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def metric_pack(run_dir: Path):
    payload = load_json(run_dir / "metrics.json")
    out = {}
    for t in metric_tags:
        s = payload["metric_summary"][t]
        out[t] = {"best": s["best"], "last": s["last"]}
    return out

def collapse_pack(run_dir: Path):
    payload = load_json(run_dir / "collapse_stats.json")
    return {
        "use_text_guidance_all_zero": bool(payload.get("use_text_guidance_all_zero", False)),
        "summary": payload.get("summary", {}),
    }

a_metrics = metric_pack(run_a)
a_collapse = collapse_pack(run_a)
a_best = float(a_metrics["val/miou_present_gt"]["best"] or 0.0)

runs = {
    "A": str(run_a),
}
metrics = {
    "A": a_metrics,
}
collapse = {
    "A": a_collapse,
}
final_decision = None
triggered_b = False
b_best = None

if a_best >= 0.70:
    final_decision = "PASS"
elif a_best < 0.50:
    final_decision = "FAIL"
else:
    if run_b is None:
        final_decision = "FAIL"
    else:
        triggered_b = True
        b_metrics = metric_pack(run_b)
        b_collapse = collapse_pack(run_b)
        b_best = float(b_metrics["val/miou_present_gt"]["best"] or 0.0)
        runs["B"] = str(run_b)
        metrics["B"] = b_metrics
        collapse["B"] = b_collapse
        final_decision = "PASS" if b_best >= 0.70 else "FAIL"

compare_payload = {
    "runs": runs,
    "metrics": metrics,
    "collapse": collapse,
    "triggered_B": triggered_b,
    "final_decision": final_decision,
    "decision_rule": {
        "pass_if_best_present_gt_gte": 0.70,
        "fail_if_best_present_gt_lt": 0.50,
        "conditional_B_range": [0.50, 0.70],
    },
}
compare_path.write_text(json.dumps(compare_payload, ensure_ascii=False, indent=2), encoding="utf-8")

summary = []
summary.append("# PHASE6 GATE SUMMARY")
summary.append("")
summary.append(f"- final_decision: {final_decision}")
summary.append(f"- run_A_best_val/miou_present_gt: {a_best:.10f}")
if triggered_b and b_best is not None:
    summary.append(f"- run_B_best_val/miou_present_gt: {b_best:.10f}")
summary.append(f"- use_text_guidance_all_zero_A: {a_collapse['use_text_guidance_all_zero']}")
if triggered_b and "B" in collapse:
    summary.append(f"- use_text_guidance_all_zero_B: {collapse['B']['use_text_guidance_all_zero']}")
summary.append("")
summary.append("## Next Step")
if final_decision == "PASS":
    summary.append("- 1) 进入全量训练前，固定本闸门配置并做一次可复现实验记录。")
    summary.append("- 2) 在同口径下启动全量训练（不改损失，不改split）。")
else:
    summary.append("- 1) 判定当前 backbone+管线在该闸门条件下学习性不足。")
    summary.append("- 2) 下一步仅推进 backbone 级几何能力增强，再回到同闸门复测。")
summary_path.write_text("\n".join(summary) + "\n", encoding="utf-8")
print(f"[OK] wrote {compare_path}")
print(f"[OK] wrote {summary_path}")
PY
}

echo "[PHASE6] GPU check"
nvidia-smi -L

run_train "phase6_gate_A_coordnorm_ce60" "$RUN_A" "60" "0.0"
A_BEST="$(best_present_gt "$RUN_A")"
echo "[PHASE6] A best val/miou_present_gt=${A_BEST}"

RUN_B_ACTUAL="__NONE__"
"$PY" - "$A_BEST" <<'PY'
import sys
v=float(sys.argv[1])
print("RUN_B" if (v >= 0.5 and v < 0.7) else "SKIP_B")
PY

IF_B="$("$PY" - "$A_BEST" <<'PY'
import sys
v=float(sys.argv[1])
print("1" if (v >= 0.5 and v < 0.7) else "0")
PY
)"

if [[ "$IF_B" == "1" ]]; then
  run_train "phase6_gate_B_lovasz30" "$RUN_B" "30" "0.5"
  RUN_B_ACTUAL="$RUN_B"
  B_BEST="$(best_present_gt "$RUN_B")"
  echo "[PHASE6] B best val/miou_present_gt=${B_BEST}"
else
  echo "[PHASE6] B not triggered (A outside [0.50,0.70))"
fi

build_final_reports "$RUN_A" "$RUN_B_ACTUAL"
echo "[PHASE6] gate pipeline done $(date '+%F %T')"
