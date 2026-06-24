#!/usr/bin/env bash
set -euo pipefail

PY=/root/autodl-tmp/conda-envs/llz/bin/python
ROOT=/root/lz
DATA=/root/autodl-tmp/datasets/mosaic3d/data
CLASS_FREQ=/root/lz_outputs/class_freq_scannet200.json
BASELINE=/root/lz_outputs/phase4_A_baseline30

SANITY=/root/lz_outputs/phase5_sanity_1ep
RUN_A=/root/lz_outputs/phase5_A_coordnorm
RUN_B=/root/lz_outputs/phase5_B_colordrop
RUN_C=/root/lz_outputs/phase5_C_weightedce

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
    raise FileNotFoundError(f"No tensorboard event file found in {run_dir}")
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
    "debug/use_text_guidance_epoch",
    "debug/use_text_guidance_0",
    "debug/train_pred_unique_count_epoch",
    "debug/train_pred_top1_ratio_epoch",
    "debug/val_pred_unique_count_0",
    "debug/val_pred_top1_ratio_0",
]

def series(tag):
    if tag not in scalar_tags:
        return []
    return [{"step": int(v.step), "value": float(v.value)} for v in ea.Scalars(tag)]

def summary(vals):
    if not vals:
        return {"count": 0, "best": None, "last": None, "min": None}
    arr = [x["value"] for x in vals]
    return {
        "count": len(arr),
        "best": float(max(arr)),
        "last": float(arr[-1]),
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

epochs = [x["epoch"] for x in acbs]
temps = [x["temperature"] for x in acbs]
acbs_monotonic = all(epochs[i] < epochs[i + 1] for i in range(len(epochs) - 1)) and all(
    temps[i] <= temps[i + 1] for i in range(len(temps) - 1)
)

use_text_vals = []
for key in ("debug/use_text_guidance_epoch", "debug/use_text_guidance_0"):
    use_text_vals.extend([x["value"] for x in collapse_series.get(key, [])])
use_text_zero = all(abs(v) <= 1e-12 for v in use_text_vals) if use_text_vals else True

metrics_payload = {
    "run_dir": str(run_dir),
    "event_file": event_file,
    "metric_summary": metric_summary,
    "overrides": overrides,
    "acbs": {
        "count": len(acbs),
        "epochs": acbs,
        "monotonic": bool(acbs_monotonic),
    },
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
  shift 3
  local extra=("$@")

  echo "[PHASE5] ===== ${name} start $(date '+%F %T') ====="
  mkdir -p "$out_dir"
  cd "$ROOT"
  set +e
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
    trainer.accelerator=gpu \
    trainer.devices=1 \
    trainer.max_epochs="$max_epochs" \
    "${extra[@]}" \
    > "$out_dir/train.log" 2>&1
  local exit_code=$?
  set -e
  echo "$exit_code" > "$out_dir/train.exitcode"
  if [[ "$exit_code" != "0" ]]; then
    echo "[PHASE5] ${name} failed with exit code=$exit_code"
    tail -n 120 "$out_dir/train.log" || true
    exit "$exit_code"
  fi

  extract_run_outputs "$out_dir"
  echo "[PHASE5] ===== ${name} done $(date '+%F %T') ====="
}

build_compare() {
  "$PY" - "$BASELINE" "$SANITY" "$RUN_A" "$RUN_B" "$RUN_C" <<'PY'
import json
from pathlib import Path
import sys

baseline_dir = Path(sys.argv[1])
sanity_dir = Path(sys.argv[2])
run_a = Path(sys.argv[3])
run_b = Path(sys.argv[4])
run_c = Path(sys.argv[5])
out_compare = Path("/root/lz_outputs/phase5_compare.json")
out_summary = Path("/root/lz_outputs/PHASE5_SUMMARY.md")

metric_tags = ["val/miou", "val/miou_present_gt", "val/miou_present_union", "val/miou_fg_mosaic"]

def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def load_metric(path: Path):
    return load_json(path / "metrics.json")

def load_collapse(path: Path):
    return load_json(path / "collapse_stats.json")

def metric_best_last(metric_json):
    out = {}
    for tag in metric_tags:
        s = metric_json["metric_summary"][tag]
        out[tag] = {"best": s["best"], "last": s["last"]}
    return out

def pct(delta, base):
    if base is None or abs(base) <= 1e-12:
        return None
    return (delta / base) * 100.0

def override_map(metrics_json):
    out = {}
    for line in metrics_json.get("overrides", []):
        if "=" not in line:
            out[line] = None
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out

def diff_keys(map_a, map_b):
    keys = set(map_a.keys()) | set(map_b.keys())
    diffs = {}
    for key in sorted(keys):
        va = map_a.get(key, None)
        vb = map_b.get(key, None)
        if va != vb:
            diffs[key] = [va, vb]
    return diffs

def load_dead(path: Path, split: str):
    fp = path / f"dead_classes_{split}.json"
    if not fp.exists():
        return {"count": None, "names": [], "path": str(fp), "exists": False}
    payload = load_json(fp)
    primary = payload.get("primary", [])
    names = []
    for item in primary:
        name = item.get("class_name", None)
        if name is None:
            class_id = item.get("class_id", "unknown")
            name = f"class_{class_id}"
        names.append(str(name))
    names = sorted(set(names))
    return {"count": len(names), "names": names, "path": str(fp), "exists": True}

baseline_metrics = load_metric(baseline_dir)
base = metric_best_last(baseline_metrics)

sanity_metrics = load_metric(sanity_dir)
sanity_collapse = load_collapse(sanity_dir)
sanity_ok = {
    "train_exitcode_zero": (sanity_dir / "train.exitcode").exists() and (sanity_dir / "train.exitcode").read_text(encoding="utf-8").strip() == "0",
    "use_text_guidance_all_zero": bool(sanity_collapse.get("use_text_guidance_all_zero", False)),
    "metrics_present": all(sanity_metrics["metric_summary"][k]["count"] > 0 for k in metric_tags),
}

runs = {"A": run_a, "B": run_b, "C": run_c}
run_metrics = {}
run_collapse = {}
run_stats = {}
for key, run_dir in runs.items():
    run_metrics[key] = load_metric(run_dir)
    run_collapse[key] = load_collapse(run_dir)
    run_stats[key] = metric_best_last(run_metrics[key])

delta_vs_base = {}
for key in ("A", "B", "C"):
    delta_vs_base[key] = {"best": {}, "last": {}}
    for tag in metric_tags:
        db = run_stats[key][tag]["best"] - base[tag]["best"]
        dl = run_stats[key][tag]["last"] - base[tag]["last"]
        delta_vs_base[key]["best"][tag] = {
            "delta": db,
            "pct": pct(db, base[tag]["best"]),
        }
        delta_vs_base[key]["last"][tag] = {
            "delta": dl,
            "pct": pct(dl, base[tag]["last"]),
        }

def delta_between(src, dst):
    out = {"best": {}, "last": {}}
    for tag in metric_tags:
        out["best"][tag] = run_stats[dst][tag]["best"] - run_stats[src][tag]["best"]
        out["last"][tag] = run_stats[dst][tag]["last"] - run_stats[src][tag]["last"]
    return out

delta_a_b = delta_between("A", "B")
delta_b_c = delta_between("B", "C")

over_a = override_map(run_metrics["A"])
over_b = override_map(run_metrics["B"])
over_c = override_map(run_metrics["C"])

diff_ab = diff_keys(over_a, over_b)
diff_bc = diff_keys(over_b, over_c)
allowed_ab = {"paths.output_dir", "data.color_drop_prob", "data.color_drop_mode"}
allowed_bc = {
    "paths.output_dir",
    "model.loss.seg_loss.use_class_weight",
    "model.loss.seg_loss.class_weight_mode",
    "model.loss.seg_loss.class_weight_clip_min",
    "model.loss.seg_loss.class_weight_clip_max",
    "model.loss.seg_loss.class_freq_path",
}

override_checks = {
    "A_vs_B_only_specified": set(diff_ab.keys()).issubset(allowed_ab),
    "B_vs_C_only_specified": set(diff_bc.keys()).issubset(allowed_bc),
    "A_vs_B_diff_keys": diff_ab,
    "B_vs_C_diff_keys": diff_bc,
}

dead = {}
for key, run_dir in runs.items():
    dead[key] = {
        "best": load_dead(run_dir, "best"),
        "last": load_dead(run_dir, "last"),
    }

def dead_delta(src, dst, split):
    src_set = set(dead[src][split]["names"])
    dst_set = set(dead[dst][split]["names"])
    return {
        "count_delta": dead[dst][split]["count"] - dead[src][split]["count"] if (dead[src][split]["count"] is not None and dead[dst][split]["count"] is not None) else None,
        "added": sorted(dst_set - src_set),
        "removed": sorted(src_set - dst_set),
    }

dead_changes = {
    "A_to_B_best": dead_delta("A", "B", "best"),
    "B_to_C_best": dead_delta("B", "C", "best"),
    "A_to_B_last": dead_delta("A", "B", "last"),
    "B_to_C_last": dead_delta("B", "C", "last"),
}

use_text_zero = {k: bool(run_collapse[k].get("use_text_guidance_all_zero", False)) for k in ("A", "B", "C")}
acbs = {
    k: {
        "count": int(run_metrics[k].get("acbs", {}).get("count", 0)),
        "monotonic": bool(run_metrics[k].get("acbs", {}).get("monotonic", False)),
        "epochs": run_metrics[k].get("acbs", {}).get("epochs", []),
    }
    for k in ("A", "B", "C")
}

baseline_miou = base["val/miou"]["best"]
run0_pass = abs((sanity_metrics["metric_summary"]["val/miou"]["best"] - baseline_miou) / baseline_miou) <= 0.50 if baseline_miou else False

threshold_hits = {}
for key in ("A", "B", "C"):
    d_fg = delta_vs_base[key]["best"]["val/miou_fg_mosaic"]["delta"]
    p_fg = delta_vs_base[key]["best"]["val/miou_fg_mosaic"]["pct"] or 0.0
    d_miou = delta_vs_base[key]["best"]["val/miou"]["delta"]
    p_miou = delta_vs_base[key]["best"]["val/miou"]["pct"] or 0.0
    d_pg = delta_vs_base[key]["best"]["val/miou_present_gt"]["delta"]
    p_pg = delta_vs_base[key]["best"]["val/miou_present_gt"]["pct"] or 0.0
    c1 = (d_fg >= 0.0005) and (p_fg >= 30.0)
    c2 = (d_miou >= 0.0014) and (p_miou >= 20.0)
    c3 = (d_pg >= 0.0011) and (p_pg >= 15.0)
    threshold_hits[key] = {"fg_rule": c1, "miou_rule": c2, "present_gt_rule": c3, "any": c1 or c2 or c3}

coordnorm_effective = delta_vs_base["A"]["best"]["val/miou_fg_mosaic"]["delta"] > 0.0
colordrop_effective = delta_a_b["best"]["val/miou_fg_mosaic"] > 0.0
weightedce_effective = delta_b_c["best"]["val/miou_fg_mosaic"] > 0.0

compare = {
    "baseline_phase4_A": str(baseline_dir),
    "sanity_run": {
        "path": str(sanity_dir),
        "checks": sanity_ok,
        "metric_summary": metric_best_last(sanity_metrics),
    },
    "runs": {k: str(v) for k, v in runs.items()},
    "metrics": run_stats,
    "delta_vs_baseline_phase4_A": delta_vs_base,
    "delta_A_to_B": delta_a_b,
    "delta_B_to_C": delta_b_c,
    "dead_classes": {
        "counts": {
            k: {"best": dead[k]["best"]["count"], "last": dead[k]["last"]["count"]}
            for k in ("A", "B", "C")
        },
        "changes": dead_changes,
    },
    "use_text_guidance_all_zero": use_text_zero,
    "acbs": acbs,
    "override_checks": override_checks,
    "threshold_hits": threshold_hits,
    "effectiveness": {
        "coordnorm": coordnorm_effective,
        "colordrop": colordrop_effective,
        "weightedce": weightedce_effective,
    },
    "run0_replaced_by_sanity": {
        "phase4_baseline_reused": True,
        "sanity_pass": all(sanity_ok.values()),
        "sanity_miou_not_for_performance": True,
    },
}
out_compare.write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8")

any_hit = any(threshold_hits[k]["any"] for k in ("A", "B", "C"))
best_run = max(("A", "B", "C"), key=lambda x: run_stats[x]["val/miou_fg_mosaic"]["best"])

summary = []
summary.append("# PHASE5 SUMMARY")
summary.append("")
summary.append("## Sanity")
summary.append(f"- phase4 baseline reused: True (`{baseline_dir}`)")
summary.append(f"- sanity checks: {sanity_ok}")
summary.append("")
summary.append("## Best Metrics (vs phase4_A)")
for k in ("A", "B", "C"):
    m = run_stats[k]
    d = delta_vs_base[k]["best"]
    summary.append(
        f"- {k}: miou={m['val/miou']['best']:.10f} (Δ={d['val/miou']['delta']:.10f}), "
        f"present_gt={m['val/miou_present_gt']['best']:.10f} (Δ={d['val/miou_present_gt']['delta']:.10f}), "
        f"fg_mosaic={m['val/miou_fg_mosaic']['best']:.10f} (Δ={d['val/miou_fg_mosaic']['delta']:.10f})"
    )
summary.append("")
summary.append("## Effect Attribution")
summary.append(f"- CoordNorm effective: {coordnorm_effective} (A vs baseline)")
summary.append(f"- ColorDrop effective: {colordrop_effective} (B vs A)")
summary.append(f"- WeightedCE effective: {weightedce_effective} (C vs B)")
summary.append(f"- Dead class counts (best): A={dead['A']['best']['count']}, B={dead['B']['best']['count']}, C={dead['C']['best']['count']}")
summary.append(f"- Dead class changes A→B(best): +{len(dead_changes['A_to_B_best']['added'])} / -{len(dead_changes['A_to_B_best']['removed'])}")
summary.append(f"- Dead class changes B→C(best): +{len(dead_changes['B_to_C_best']['added'])} / -{len(dead_changes['B_to_C_best']['removed'])}")
summary.append("")
summary.append("## Compliance")
summary.append(f"- use_text_guidance all zero (A/B/C): {use_text_zero}")
summary.append(f"- override consistency: A_vs_B={override_checks['A_vs_B_only_specified']}, B_vs_C={override_checks['B_vs_C_only_specified']}")
summary.append(f"- ACBS monotonic (A/B/C): A={acbs['A']['monotonic']}, B={acbs['B']['monotonic']}, C={acbs['C']['monotonic']}")
summary.append("")
summary.append("## Threshold Result")
summary.append(f"- threshold hits: {threshold_hits}")
if not any_hit:
    summary.append("- verdict: 在不换 backbone 条件下，本阶段策略不足以达到既定阈值。")
    summary.append("- next (keep 1-2):")
    summary.append("  1) 保留 CoordNorm + WeightedCE，移除 ColorDrop。")
    summary.append("  2) 转向 backbone 级几何特征增强（保持其余训练协议不变）。")
else:
    summary.append(f"- verdict: 达标（best run={best_run}）。")
    summary.append("- next (keep 1-2):")
    summary.append("  1) 保留带来净增益的改动组合。")
    summary.append("  2) 仅在该组合上继续做一次30epoch复验。")

out_summary.write_text("\n".join(summary) + "\n", encoding="utf-8")
print(f"[OK] wrote {out_compare}")
print(f"[OK] wrote {out_summary}")
PY
}

echo "[PHASE5] GPU check"
nvidia-smi -L

run_train "phase5_sanity_1ep" "$SANITY" "1" \
  data.coord_norm_enable=false \
  data.color_drop_prob=0.0 \
  data.color_drop_mode=gray \
  model.loss.seg_loss.use_class_weight=false

run_train "phase5_A_coordnorm" "$RUN_A" "30" \
  data.coord_norm_enable=true \
  data.color_drop_prob=0.0 \
  data.color_drop_mode=gray \
  model.loss.seg_loss.use_class_weight=false

run_train "phase5_B_colordrop" "$RUN_B" "30" \
  data.coord_norm_enable=true \
  data.color_drop_prob=0.3 \
  data.color_drop_mode=gray \
  model.loss.seg_loss.use_class_weight=false

run_train "phase5_C_weightedce" "$RUN_C" "30" \
  data.coord_norm_enable=true \
  data.color_drop_prob=0.3 \
  data.color_drop_mode=gray \
  model.loss.seg_loss.use_class_weight=true \
  model.loss.seg_loss.class_weight_mode=inv_sqrt \
  model.loss.seg_loss.class_weight_clip_min=0.2 \
  model.loss.seg_loss.class_weight_clip_max=5.0 \
  model.loss.seg_loss.class_freq_path="$CLASS_FREQ"

build_compare
echo "[PHASE5] pipeline done $(date '+%F %T')"
