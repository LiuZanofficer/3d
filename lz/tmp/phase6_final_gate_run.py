import glob
import json
import shutil
import subprocess
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator

PYTHON = "/root/autodl-tmp/conda-envs/llz/bin/python"
ROOT = Path("/root/lz")
DATA_DIR = "/root/autodl-tmp/datasets/mosaic3d/data"
OLD_OVERRIDES = Path("/root/lz_outputs/phase6_recheck_0C_repro/.hydra/overrides.yaml")
OUT = Path("/root/lz_outputs/phase6_final_gate_full15")
DECISION_JSON = Path("/root/lz_outputs/phase6_final_gate_decision.json")
SUMMARY_MD = Path("/root/lz_outputs/PHASE6_FINAL_GATE_SUMMARY.md")

OUT.mkdir(parents=True, exist_ok=True)
if not OLD_OVERRIDES.exists():
    raise FileNotFoundError(f"Missing old overrides: {OLD_OVERRIDES}")

old_items = []
for line in OLD_OVERRIDES.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line.startswith("- "):
        old_items.append(line[2:].strip())


def parse_key(item: str):
    if "=" in item:
        return item.split("=", 1)[0].strip()
    return item.strip()


remove_keys = {
    "data",
    "paths.data_dir",
    "paths.root_dir",
    "paths.output_dir",
    "trainer.accelerator",
    "trainer.devices",
    "trainer.max_epochs",
    "+trainer.max_steps",
    "trainer.max_steps",
    "data.train_dataset.split",
    "data.val_datasets.0.split",
    "seed",
    "model.loss.weights.seg_loss",
    "model.loss.weights.instance_loss",
    "model.loss.weights.hpza_loss",
    "model.loss.seg_loss.lovasz_weight",
    "data.coord_norm_enable",
    "data.color_drop_prob",
    "data.color_drop_mode",
    "model.loss.seg_loss.use_class_weight",
    "sampler.class_freq_path",
}
filtered = [item for item in old_items if parse_key(item) not in remove_keys]

new_overrides = filtered + [
    "data=sc",
    f"paths.data_dir={DATA_DIR}",
    "paths.root_dir=/root/lz",
    f"paths.output_dir={OUT.as_posix()}",
    "trainer.accelerator=gpu",
    "trainer.devices=1",
    "trainer.max_epochs=15",
    "seed=42",
    "sampler.class_freq_path=/root/lz_outputs/class_freq_scannet200.json",
    "model.loss.weights.seg_loss=1.0",
    "model.loss.weights.instance_loss=0.0",
    "model.loss.weights.hpza_loss=0.0",
    "model.loss.seg_loss.lovasz_weight=0.0",
    "data.coord_norm_enable=true",
    "data.color_drop_prob=0.0",
    "data.color_drop_mode=gray",
    "model.loss.seg_loss.use_class_weight=false",
]

train_log = OUT / "train.log"
cmd = [PYTHON, "src/train.py", *new_overrides]
with train_log.open("w", encoding="utf-8") as f:
    proc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
(OUT / "train.exitcode").write_text(str(proc.returncode), encoding="utf-8")
if proc.returncode != 0:
    raise SystemExit(proc.returncode)

# Flatten hydra files
hydra_cfgs = sorted(OUT.glob("*/*/.hydra/config.yaml"))
if not hydra_cfgs:
    raise FileNotFoundError(f"No .hydra/config.yaml found under {OUT}")
sel_hydra = hydra_cfgs[-1].parent
flat_hydra = OUT / ".hydra"
flat_hydra.mkdir(parents=True, exist_ok=True)
shutil.copy2(sel_hydra / "config.yaml", flat_hydra / "config.yaml")
shutil.copy2(sel_hydra / "overrides.yaml", flat_hydra / "overrides.yaml")

# Parse events
event_files = sorted(glob.glob(str(OUT / "logs" / "lz" / "version_*" / "events.out.tfevents.*")))
if not event_files:
    raise FileNotFoundError(f"No event file under {OUT}")
ea = event_accumulator.EventAccumulator(event_files[-1])
ea.Reload()
scalar_tags = set(ea.Tags().get("scalars", []))

metric_tags = ["val/miou", "val/miou_present_gt", "val/miou_present_union", "val/miou_fg_mosaic"]
collapse_tags = [
    "debug/use_text_guidance_epoch",
    "debug/use_text_guidance_0",
    "debug/train_pred_unique_count_epoch",
    "debug/train_pred_top1_ratio_epoch",
    "debug/val_pred_unique_count_0",
    "debug/val_pred_top1_ratio_0",
]

metrics = {}
metric_series = {}
for tag in metric_tags:
    if tag not in scalar_tags:
        metrics[tag] = {"best": None, "last": None, "count": 0}
        metric_series[tag] = []
        continue
    vals = [float(x.value) for x in ea.Scalars(tag)]
    metric_series[tag] = vals
    metrics[tag] = {
        "best": float(max(vals)) if vals else None,
        "last": float(vals[-1]) if vals else None,
        "count": len(vals),
    }

collapse_series = {}
collapse_summary = {}
for tag in collapse_tags:
    if tag not in scalar_tags:
        collapse_series[tag] = []
        collapse_summary[tag] = {"best": None, "last": None, "count": 0}
        continue
    vals = [float(x.value) for x in ea.Scalars(tag)]
    collapse_series[tag] = vals
    collapse_summary[tag] = {
        "best": float(max(vals)) if vals else None,
        "last": float(vals[-1]) if vals else None,
        "count": len(vals),
    }

use_text_values = []
for key in ("debug/use_text_guidance_epoch", "debug/use_text_guidance_0"):
    use_text_values.extend(collapse_series.get(key, []))
use_text_zero = all(abs(v) <= 1e-12 for v in use_text_values) if use_text_values else True

collapse_payload = {
    "run_dir": OUT.as_posix(),
    "event_file": event_files[-1],
    "summary": collapse_summary,
    "series": collapse_series,
    "use_text_guidance_all_zero": bool(use_text_zero),
}
(OUT / "collapse_stats.json").write_text(
    json.dumps(collapse_payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)


def trend_last5(values):
    if not values:
        return {"n": 0, "slope": None, "delta": None, "nondecreasing": None}
    arr = values[-5:]
    n = len(arr)
    if n == 1:
        return {"n": 1, "slope": 0.0, "delta": 0.0, "nondecreasing": True}
    slope = (arr[-1] - arr[0]) / (n - 1)
    delta = arr[-1] - arr[0]
    nondecreasing = all(arr[i + 1] >= arr[i] for i in range(n - 1))
    return {"n": n, "slope": float(slope), "delta": float(delta), "nondecreasing": bool(nondecreasing)}


trend = {k: trend_last5(metric_series.get(k, [])) for k in metric_tags}

best_fg = float(metrics["val/miou_fg_mosaic"]["best"] or 0.0)
best_pg = float(metrics["val/miou_present_gt"]["best"] or 0.0)

no_rise = True
for key in ("val/miou_fg_mosaic", "val/miou_present_gt"):
    t = trend[key]
    s = t["slope"] if t["slope"] is not None else -1.0
    d = t["delta"] if t["delta"] is not None else -1.0
    if not (s <= 1e-4 or d <= 1e-4):
        no_rise = False

sustained_up = True
for key in ("val/miou_fg_mosaic", "val/miou_present_gt"):
    t = trend[key]
    s = t["slope"] if t["slope"] is not None else 0.0
    nd = bool(t["nondecreasing"]) if t["nondecreasing"] is not None else False
    if not (s > 0 and nd):
        sustained_up = False

if (best_fg < 0.003) and (best_pg < 0.03) and no_rise:
    final_decision = "SWITCH_BACKBONE_NOW"
    recommendation = "当前backbone在全量15epoch未见起量迹象，建议立即切换backbone。"
elif (best_fg >= 0.01) and sustained_up:
    final_decision = "KEEP_BACKBONE_CONTINUE"
    recommendation = "当前backbone出现明确起量趋势，可继续在现backbone上训练。"
else:
    final_decision = "GRAY_EXTEND_TO_30"
    recommendation = "指标处于灰区，建议先延长到30epoch再做是否换backbone决策。"

metrics_payload = {
    "run_dir": OUT.as_posix(),
    "event_file": event_files[-1],
    "best_last": {k: {"best": metrics[k]["best"], "last": metrics[k]["last"]} for k in metric_tags},
    "counts": {k: metrics[k]["count"] for k in metric_tags},
    "trend_last5": trend,
    "use_text_guidance_all_zero": bool(use_text_zero),
}
(OUT / "metrics.json").write_text(
    json.dumps(metrics_payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

decision_payload = {
    "run_dir": OUT.as_posix(),
    "best_last": metrics_payload["best_last"],
    "trend_last5": trend,
    "use_text_guidance_all_zero": bool(use_text_zero),
    "final_decision": final_decision,
    "recommendation": recommendation,
}
DECISION_JSON.write_text(
    json.dumps(decision_payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

summary_lines = [
    "# PHASE6 FINAL GATE SUMMARY",
    "",
    f"- final_decision: {final_decision}",
    f"- best val/miou={metrics['val/miou']['best']}",
    f"- best val/miou_present_gt={metrics['val/miou_present_gt']['best']}",
    f"- best val/miou_present_union={metrics['val/miou_present_union']['best']}",
    f"- best val/miou_fg_mosaic={metrics['val/miou_fg_mosaic']['best']}",
    f"- use_text_guidance_all_zero: {use_text_zero}",
    "",
    "## Next",
    f"- {recommendation}",
    "- 保持本次设置不变再执行下一步动作（最多一个动作），避免变量漂移。",
]
SUMMARY_MD.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print(
    json.dumps(
        {
            "final_decision": final_decision,
            "best_last": metrics_payload["best_last"],
            "use_text_guidance_all_zero": use_text_zero,
            "out_run": OUT.as_posix(),
            "decision_json": DECISION_JSON.as_posix(),
            "summary_md": SUMMARY_MD.as_posix(),
        },
        ensure_ascii=False,
        indent=2,
    )
)
