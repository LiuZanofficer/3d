#!/usr/bin/env bash
set -euo pipefail

PY=/root/autodl-tmp/conda-envs/llz/bin/python
ROOT=/root/lz
DATA=/root/autodl-tmp/datasets/mosaic3d/data
CLASS_FREQ=/root/lz_outputs/class_freq_scannet200.json

RUN_A=/root/lz_outputs/phase4_A_baseline30
RUN_B=/root/lz_outputs/phase4_B_coord0
RUN_C=/root/lz_outputs/phase4_C_color0

extract_run_artifacts() {
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
scalar_tags = set(ea.Tags().get("scalars", []))

metric_tags = [
    "val/miou",
    "val/miou_present_gt",
    "val/miou_present_union",
    "val/miou_fg_mosaic",
]
collapse_tags = [
    "debug/use_text_guidance",
    "debug/use_text_guidance_epoch",
    "debug/use_text_guidance_0",
    "debug/train_pred_unique_count",
    "debug/train_pred_unique_count_epoch",
    "debug/train_pred_top1_ratio",
    "debug/train_pred_top1_ratio_epoch",
    "debug/train_ignore_ratio",
    "debug/train_ignore_ratio_epoch",
    "debug/val_pred_unique_count_0",
    "debug/val_pred_top1_ratio_0",
    "debug/val_ignore_ratio_0",
    "debug/input_coord_mean",
    "debug/input_coord_std",
    "debug/input_color_mean",
    "debug/input_color_std",
]

def series(tag: str):
    if tag not in scalar_tags:
        return []
    return [{"step": int(x.step), "value": float(x.value)} for x in ea.Scalars(tag)]

def summary(vals):
    if not vals:
        return {"count": 0, "last": None, "best": None, "min": None}
    arr = [x["value"] for x in vals]
    return {
        "count": len(arr),
        "last": float(arr[-1]),
        "best": float(max(arr)),
        "min": float(min(arr)),
    }

metric_series = {k: series(k) for k in metric_tags}
metric_summary = {k: summary(v) for k, v in metric_series.items()}
collapse_series = {k: series(k) for k in collapse_tags}
collapse_summary = {k: summary(v) for k, v in collapse_series.items()}

override_files = sorted(run_dir.glob("*/*/.hydra/overrides.yaml"))
overrides = []
if override_files:
    for line in override_files[-1].read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("- "):
            overrides.append(line[2:].strip())

acbs_epochs = []
log_path = run_dir / "train.log"
if log_path.exists():
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    seen = set()
    for m in re.finditer(r"ACBS epoch=(\d+) temperature=([0-9.]+)", text):
        rec = (int(m.group(1)), float(m.group(2)))
        if rec in seen:
            continue
        seen.add(rec)
        acbs_epochs.append({"epoch": rec[0], "temperature": rec[1]})

metrics_payload = {
    "run_dir": str(run_dir),
    "event_file": event_file,
    "metric_summary": metric_summary,
    "overrides": overrides,
    "acbs": {
        "count": len(acbs_epochs),
        "first": acbs_epochs[0] if acbs_epochs else None,
        "last": acbs_epochs[-1] if acbs_epochs else None,
    },
}

collapse_payload = {
    "run_dir": str(run_dir),
    "event_file": event_file,
    "summary": collapse_summary,
    "series": collapse_series,
    "acbs_epochs": acbs_epochs,
}

(run_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
(run_dir / "collapse_stats.json").write_text(json.dumps(collapse_payload, indent=2), encoding="utf-8")
print(f"[OK] wrote {run_dir / 'metrics.json'}")
print(f"[OK] wrote {run_dir / 'collapse_stats.json'}")
PY
}

run_one() {
  local name="$1"
  local out_dir="$2"
  local ablate_coord="$3"
  local ablate_color="$4"

  echo "[PHASE4] ===== ${name} start $(date '+%F %T') ====="
  mkdir -p "$out_dir"
  cd "$ROOT"
  "$PY" src/train.py \
    data=sc \
    paths.data_dir="$DATA" \
    paths.root_dir="$ROOT" \
    paths.output_dir="$out_dir" \
    sampler.class_freq_path="$CLASS_FREQ" \
    model.loss.weights.seg_loss=1.0 \
    model.loss.weights.instance_loss=0.0 \
    model.loss.weights.hpza_loss=0.0 \
    model.loss.seg_loss.lovasz_weight=0.0 \
    model.net.backbone.ablate_coord="$ablate_coord" \
    model.net.backbone.ablate_color="$ablate_color" \
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.max_epochs=30 \
    > "$out_dir/train.log" 2>&1
  echo "0" > "$out_dir/train.exitcode"
  extract_run_artifacts "$out_dir"
  echo "[PHASE4] ===== ${name} done $(date '+%F %T') ====="
}

build_compare() {
  "$PY" - "$RUN_A" "$RUN_B" "$RUN_C" <<'PY'
import json
import sys
from pathlib import Path

run_a = Path(sys.argv[1])
run_b = Path(sys.argv[2])
run_c = Path(sys.argv[3])

tags = ["val/miou", "val/miou_present_gt", "val/miou_present_union", "val/miou_fg_mosaic"]

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def pick_stats(metric_json, tag):
    stat = metric_json["metric_summary"][tag]
    return {
        "best": stat["best"],
        "last": stat["last"],
    }

def normalize_override_line(line: str):
    if "=" not in line:
        return line, None
    key, value = line.split("=", 1)
    return key.strip(), value.strip()

def map_overrides(lines):
    out = {}
    for line in lines:
        key, value = normalize_override_line(line)
        out[key] = value
    return out

def use_text_all_zero(collapse_json):
    keys = ["debug/use_text_guidance", "debug/use_text_guidance_epoch", "debug/use_text_guidance_0"]
    for key in keys:
        item = collapse_json["summary"].get(key, {})
        best = item.get("best")
        if best is None:
            continue
        if abs(float(best)) > 1e-12:
            return False
    return True

def acbs_ok(metric_json):
    return int(metric_json.get("acbs", {}).get("count", 0)) > 0

met_a = load_json(run_a / "metrics.json")
met_b = load_json(run_b / "metrics.json")
met_c = load_json(run_c / "metrics.json")
col_a = load_json(run_a / "collapse_stats.json")
col_b = load_json(run_b / "collapse_stats.json")
col_c = load_json(run_c / "collapse_stats.json")

runs = {"A": met_a, "B": met_b, "C": met_c}
cols = {"A": col_a, "B": col_b, "C": col_c}

best_last = {}
for key in ["A", "B", "C"]:
    best_last[key] = {tag: pick_stats(runs[key], tag) for tag in tags}

delta_ab = {tag: best_last["B"][tag]["best"] - best_last["A"][tag]["best"] for tag in tags}
delta_ac = {tag: best_last["C"][tag]["best"] - best_last["A"][tag]["best"] for tag in tags}

ov_a = map_overrides(met_a.get("overrides", []))
ov_b = map_overrides(met_b.get("overrides", []))
ov_c = map_overrides(met_c.get("overrides", []))

ignore_keys = {"paths.output_dir", "model.net.backbone.ablate_coord", "model.net.backbone.ablate_color"}
def same_except_ablation(base, other):
    base_filtered = {k: v for k, v in base.items() if k not in ignore_keys}
    other_filtered = {k: v for k, v in other.items() if k not in ignore_keys}
    return base_filtered == other_filtered

main_metric = "val/miou_fg_mosaic"
a = best_last["A"][main_metric]["best"]
b = best_last["B"][main_metric]["best"]
c = best_last["C"][main_metric]["best"]
eps = max(2e-4, 0.05 * a if a is not None else 0.0)

b_close = abs(b - a) <= eps
c_close = abs(c - a) <= eps
b_down = (a - b) > eps
c_down = (a - c) > eps

if b_close and c_down:
    verdict = "颜色依赖主导（几何盲）"
elif c_close and b_down:
    verdict = "几何主导"
elif b_down and c_down:
    verdict = "几何+颜色都在用（混合）"
else:
    verdict = "混合（差异不显著）"

compare = {
    "runs": {
        "A": str(run_a),
        "B": str(run_b),
        "C": str(run_c),
    },
    "metrics": best_last,
    "delta_best": {
        "A_to_B": delta_ab,
        "A_to_C": delta_ac,
        "A_to_B_abs": {k: abs(v) for k, v in delta_ab.items()},
        "A_to_C_abs": {k: abs(v) for k, v in delta_ac.items()},
    },
    "checks": {
        "use_text_guidance_all_zero": {
            "A": use_text_all_zero(col_a),
            "B": use_text_all_zero(col_b),
            "C": use_text_all_zero(col_c),
        },
        "acbs_active": {
            "A": acbs_ok(met_a),
            "B": acbs_ok(met_b),
            "C": acbs_ok(met_c),
        },
        "only_coord_color_varies": {
            "A_vs_B": same_except_ablation(ov_a, ov_b),
            "A_vs_C": same_except_ablation(ov_a, ov_c),
        },
    },
    "ablation_input_stats": {
        "A": {
            "coord_mean": col_a["summary"].get("debug/input_coord_mean", {}).get("last"),
            "coord_std": col_a["summary"].get("debug/input_coord_std", {}).get("last"),
            "color_mean": col_a["summary"].get("debug/input_color_mean", {}).get("last"),
            "color_std": col_a["summary"].get("debug/input_color_std", {}).get("last"),
        },
        "B": {
            "coord_mean": col_b["summary"].get("debug/input_coord_mean", {}).get("last"),
            "coord_std": col_b["summary"].get("debug/input_coord_std", {}).get("last"),
            "color_mean": col_b["summary"].get("debug/input_color_mean", {}).get("last"),
            "color_std": col_b["summary"].get("debug/input_color_std", {}).get("last"),
        },
        "C": {
            "coord_mean": col_c["summary"].get("debug/input_coord_mean", {}).get("last"),
            "coord_std": col_c["summary"].get("debug/input_coord_std", {}).get("last"),
            "color_mean": col_c["summary"].get("debug/input_color_mean", {}).get("last"),
            "color_std": col_c["summary"].get("debug/input_color_std", {}).get("last"),
        },
    },
    "verdict": {
        "rule_metric": main_metric,
        "eps": eps,
        "decision": verdict,
    },
}

out_compare = Path("/root/lz_outputs/phase4_compare.json")
out_compare.write_text(json.dumps(compare, indent=2), encoding="utf-8")

suggestions = []
if verdict == "颜色依赖主导（几何盲）":
    suggestions = [
        "优先修复几何分支：检查 coord 归一化/尺度与随机裁剪后坐标分布是否异常。",
        "在不改损失前提下先做几何特征通路单元测试（固定输入点云验证几何扰动敏感性）。",
    ]
elif verdict == "几何主导":
    suggestions = [
        "降低颜色增强扰动并检查 NormalizeColor 是否引入分布漂移。",
        "保留几何主干不变，后续只在颜色融合权重上做小幅调节。",
    ]
else:
    suggestions = [
        "先保留 coord+color 双输入，下一步只加每类 IoU 导出定位低效类别。",
        "在 CE-only 设定下做 1 个最小改动（例如 class-weight 校正）验证净增益。",
    ]

md_lines = []
md_lines.append("# PHASE4 Summary")
md_lines.append("")
md_lines.append("## Runs")
md_lines.append(f"- A baseline: `{run_a}`")
md_lines.append(f"- B coord0: `{run_b}`")
md_lines.append(f"- C color0: `{run_c}`")
md_lines.append("")
md_lines.append("## Acceptance Checks")
md_lines.append(f"- CE-only use_text_guidance=0: A={compare['checks']['use_text_guidance_all_zero']['A']}, B={compare['checks']['use_text_guidance_all_zero']['B']}, C={compare['checks']['use_text_guidance_all_zero']['C']}")
md_lines.append(f"- ACBS active evidence: A={compare['checks']['acbs_active']['A']}, B={compare['checks']['acbs_active']['B']}, C={compare['checks']['acbs_active']['C']}")
md_lines.append(f"- Only coord/color differs: A_vs_B={compare['checks']['only_coord_color_varies']['A_vs_B']}, A_vs_C={compare['checks']['only_coord_color_varies']['A_vs_C']}")
md_lines.append("")
md_lines.append("## Best Metrics")
for run_name in ["A", "B", "C"]:
    vals = compare["metrics"][run_name]
    md_lines.append(
        f"- {run_name}: val/miou={vals['val/miou']['best']:.10f}, "
        f"miou_present_gt={vals['val/miou_present_gt']['best']:.10f}, "
        f"miou_present_union={vals['val/miou_present_union']['best']:.10f}, "
        f"miou_fg_mosaic={vals['val/miou_fg_mosaic']['best']:.10f}"
    )
md_lines.append("")
md_lines.append("## Delta (Best)")
md_lines.append(f"- A→B ({main_metric}): {delta_ab[main_metric]:.10f} (abs={abs(delta_ab[main_metric]):.10f})")
md_lines.append(f"- A→C ({main_metric}): {delta_ac[main_metric]:.10f} (abs={abs(delta_ac[main_metric]):.10f})")
md_lines.append("")
md_lines.append("## Ablation Input Stats")
for run_name in ["A", "B", "C"]:
    st = compare["ablation_input_stats"][run_name]
    md_lines.append(
        f"- {run_name}: coord_mean={st['coord_mean']}, coord_std={st['coord_std']}, "
        f"color_mean={st['color_mean']}, color_std={st['color_std']}"
    )
md_lines.append("")
md_lines.append("## Verdict")
md_lines.append(f"- Decision: {verdict}")
md_lines.append(f"- Rule metric: {main_metric}, eps={eps}")
md_lines.append("")
md_lines.append("## Next Step (1-2 items)")
for item in suggestions[:2]:
    md_lines.append(f"- {item}")

out_md = Path("/root/lz_outputs/PHASE4_SUMMARY.md")
out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
print(f"[OK] wrote {out_compare}")
print(f"[OK] wrote {out_md}")
PY
}

run_one "A_baseline30" "$RUN_A" "false" "false"
run_one "B_coord0" "$RUN_B" "true" "false"
run_one "C_color0" "$RUN_C" "false" "true"
build_compare

echo "[PHASE4] pipeline done $(date '+%F %T')"
