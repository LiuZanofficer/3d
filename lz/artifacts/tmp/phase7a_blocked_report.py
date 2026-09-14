import importlib
import json
from pathlib import Path

import torch

out_dir = Path("/root/lz_outputs")
out_dir.mkdir(parents=True, exist_ok=True)

# Step0
env_payload = {
    "host": "connect.nmb1.seetacloud.com",
    "gpu_visible": bool(torch.cuda.is_available()),
    "gpu_device_count": int(torch.cuda.device_count()),
    "torch_cuda_available": bool(torch.cuda.is_available()),
}
(out_dir / "phase7A_env_check.json").write_text(
    json.dumps(env_payload, ensure_ascii=False, indent=2), encoding="utf-8"
)

# Step1
source_candidates = [
    Path("/root/Mosaic3D"),
    Path("/root/mosaic3d"),
    Path("/root/autodl-tmp/Mosaic3D"),
]
selected_source = None
for path in source_candidates:
    if path.exists():
        selected_source = path
        break

required_rel = [
    "src/models/networks/spunet/spconv_unet_v1m1_base.py",
    "src/models/utils/structure.py",
    "src/models/utils/misc.py",
    "src/models/utils/serialization/default.py",
    "configs/model/spunet34c.yaml",
]
required_status = {}
if selected_source is not None:
    for rel in required_rel:
        required_status[rel] = bool((selected_source / rel).exists())
else:
    for rel in required_rel:
        required_status[rel] = False

modules = ["spconv", "torch_scatter", "timm", "addict"]
dep_status = {}
for module in modules:
    try:
        importlib.import_module(module)
        dep_status[module] = {"ok": True, "error": None}
    except Exception as exc:
        dep_status[module] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

dep_ok = all(v["ok"] for v in dep_status.values())
required_ok = all(required_status.values())
source_ok = selected_source is not None

if not source_ok:
    final = "BLOCKED_MISSING_SPUNET_SOURCE"
elif not required_ok:
    final = "BLOCKED_MISSING_SPUNET_SOURCE"
elif not dep_ok:
    final = "BLOCKED_MISSING_DEPENDENCY"
else:
    final = "UNBLOCKED"

dep_payload = {
    "source_priority": [str(path) for path in source_candidates],
    "selected_source": str(selected_source) if selected_source else None,
    "source_found": source_ok,
    "required_files": required_status,
    "required_files_all_present": required_ok,
    "dependency_imports": dep_status,
    "dependencies_all_present": dep_ok,
    "step1_result": final,
}
(out_dir / "phase7A_dependency_check.json").write_text(
    json.dumps(dep_payload, ensure_ascii=False, indent=2), encoding="utf-8"
)

sync_lines = [
    "# Phase7A Code Sync",
    "",
    f"- selected_source: `{selected_source}`",
    f"- required_files_all_present: {required_ok}",
    f"- dependencies_all_present: {dep_ok}",
]
if final == "BLOCKED_MISSING_DEPENDENCY":
    sync_lines.append("- sync_status: skipped_due_to_missing_dependency")
    sync_lines.append("- reason: spconv/torch_scatter/addict import failed in llz env.")
elif final == "BLOCKED_MISSING_SPUNET_SOURCE":
    sync_lines.append("- sync_status: skipped_due_to_missing_source")
else:
    sync_lines.append("- sync_status: pending (not executed in this run)")
(out_dir / "phase7A_code_sync.md").write_text("\n".join(sync_lines) + "\n", encoding="utf-8")

compare = {
    "env_check": env_payload,
    "dependency_check": dep_payload,
    "smoke": {"executed": False, "reason": final},
    "gate_debug1_60": {"executed": False, "reason": final},
    "gate_full15": {"executed": False, "reason": final},
    "debug_use_text_guidance_all_zero": None,
    "collapse_stop_triggered": None,
    "oom_fallback_triggered": None,
    "final_decision": final,
}
(out_dir / "phase7A_compare.json").write_text(
    json.dumps(compare, ensure_ascii=False, indent=2), encoding="utf-8"
)

missing_dep = [key for key, val in dep_status.items() if not val["ok"]]
summary_lines = [
    "# PHASE7A SUMMARY",
    "",
    f"- final_decision: {final}",
    f"- source_found: {source_ok}, required_files_all_present: {required_ok}",
    f"- dependencies_all_present: {dep_ok}",
]
if missing_dep:
    summary_lines.append("- missing_dependencies: " + ", ".join(missing_dep))
summary_lines.extend(
    [
        "- next: install missing dependencies in llz env, then rerun Phase7A from Step1.4.",
        "- next: after dependency pass, execute smoke -> gate_debug1_60 -> gate_full15.",
    ]
)
(out_dir / "PHASE7A_SUMMARY.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

print(
    json.dumps(
        {"final_decision": final, "missing_dependencies": missing_dep},
        ensure_ascii=False,
        indent=2,
    )
)
