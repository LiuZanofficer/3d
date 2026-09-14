import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path("/root/lz")
DATA_ROOT = Path("/root/autodl-tmp/datasets/mosaic3d/data")
DATA_SCANNET = DATA_ROOT / "scannet"
OUT_DIR = Path("/root/lz_outputs")

OUT_PROTOCOL = OUT_DIR / "phase7_precheck_protocol.json"
OUT_DATA_JSON = OUT_DIR / "phase7_precheck_data.json"
OUT_DATA_MD = OUT_DIR / "phase7_precheck_data.md"
OUT_CONTRACT = OUT_DIR / "phase7_precheck_contract.json"
OUT_SMOKE = OUT_DIR / "phase7_precheck_smoke.json"
OUT_PLAN_YAML = OUT_DIR / "phase7_precheck_4090_plan.yaml"
OUT_PLAN_MD = OUT_DIR / "phase7_precheck_4090_plan.md"
OUT_DECISION = OUT_DIR / "phase7_precheck_decision.json"
OUT_SUMMARY = OUT_DIR / "PHASE7_PRECHECK_SUMMARY.md"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_split(split_file: Path) -> List[str]:
    if not split_file.exists():
        return []
    lines = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if (not line) or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def flatten_mapper_stats(mapper: np.ndarray, ignore_index: int, num_classes: int) -> Dict[str, Any]:
    mapper = np.asarray(mapper).astype(np.int64, copy=False)
    valid = mapper[mapper != ignore_index]
    unique_valid = np.unique(valid) if valid.size > 0 else np.asarray([], dtype=np.int64)
    invalid_values = unique_valid[(unique_valid < 0) | (unique_valid >= num_classes)]
    return {
        "mapper_length": int(mapper.shape[0]),
        "ignore_index": int(ignore_index),
        "num_classes": int(num_classes),
        "valid_unique_count": int(unique_valid.size),
        "valid_min": int(unique_valid.min()) if unique_valid.size else None,
        "valid_max": int(unique_valid.max()) if unique_valid.size else None,
        "invalid_value_count": int(invalid_values.size),
        "invalid_values_sample": [int(x) for x in invalid_values[:20]],
        "in_range_or_ignore": bool(invalid_values.size == 0),
    }


def analyze_data_split(split_name: str, scenes: List[str]) -> Dict[str, Any]:
    total = len(scenes)
    if total == 0:
        return {
            "split": split_name,
            "total_scenes": 0,
            "scene_dir_exists": 0,
            "scene_dir_exists_rate": 0.0,
            "coord_color_exists": 0,
            "coord_color_exists_rate": 0.0,
            "segment200_exists": 0,
            "segment200_exists_rate": 0.0,
            "caption_point_pair_any": 0,
            "caption_point_pair_any_rate": 0.0,
            "caption_point_pair_segment3d_gathered": 0,
            "caption_point_pair_segment3d_gathered_rate": 0.0,
            "missing_samples_top100": [],
        }

    scene_dir_exists = 0
    coord_color_exists = 0
    segment200_exists = 0
    caption_point_pair_any = 0
    caption_point_pair_gathered = 0
    missing = []

    for scene in scenes:
        scene_dir = DATA_SCANNET / scene
        exists_scene = scene_dir.exists()
        coord_ok = (scene_dir / "coord.npy").exists()
        color_ok = (scene_dir / "color.npy").exists()
        seg_ok = (scene_dir / "segment200.npy").exists()

        cap_any = False
        if exists_scene:
            for cap_file in scene_dir.glob("captions.*.npz"):
                source = cap_file.name[len("captions.") : -len(".npz")]
                if (scene_dir / f"point_indices.{source}.npz").exists():
                    cap_any = True
                    break

        cap_gathered = (
            (scene_dir / "captions.segment3d-gathered.npz").exists()
            and (scene_dir / "point_indices.segment3d-gathered.npz").exists()
        )

        if exists_scene:
            scene_dir_exists += 1
        if coord_ok and color_ok:
            coord_color_exists += 1
        if seg_ok:
            segment200_exists += 1
        if cap_any:
            caption_point_pair_any += 1
        if cap_gathered:
            caption_point_pair_gathered += 1

        miss = []
        if not exists_scene:
            miss.append("scene_dir")
        if not coord_ok:
            miss.append("coord.npy")
        if not color_ok:
            miss.append("color.npy")
        if not seg_ok:
            miss.append("segment200.npy")
        if not cap_any:
            miss.append("caption_point_pair_any")
        if not cap_gathered:
            miss.append("caption_point_pair_segment3d_gathered")
        if miss and len(missing) < 100:
            missing.append({"scene": scene, "missing": miss})

    def rate(x: int) -> float:
        return float(x / total * 100.0)

    return {
        "split": split_name,
        "total_scenes": int(total),
        "scene_dir_exists": int(scene_dir_exists),
        "scene_dir_exists_rate": rate(scene_dir_exists),
        "coord_color_exists": int(coord_color_exists),
        "coord_color_exists_rate": rate(coord_color_exists),
        "segment200_exists": int(segment200_exists),
        "segment200_exists_rate": rate(segment200_exists),
        "caption_point_pair_any": int(caption_point_pair_any),
        "caption_point_pair_any_rate": rate(caption_point_pair_any),
        "caption_point_pair_segment3d_gathered": int(caption_point_pair_gathered),
        "caption_point_pair_segment3d_gathered_rate": rate(caption_point_pair_gathered),
        "missing_samples_top100": missing,
    }


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(x, device) for x in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(x, device) for x in obj)
    return obj


def main():
    # ---------------------------
    # Step 0: protocol lock
    # ---------------------------
    split_dir = ROOT / "src/data/metadata/split_files"
    split_scannet_val = split_dir / "scannet_val.txt"
    split_scannet_train = split_dir / "scannet_train.txt"
    eval_entry = ROOT / "src/eval.py"
    lz_module_path = ROOT / "src/models/lightning_modules/lz_module.py"

    data_cfg = OmegaConf.load(ROOT / "configs/data/sc.yaml")
    loss_cfg = OmegaConf.load(ROOT / "configs/model/loss/lz.yaml")
    default_num_classes = int(OmegaConf.select(data_cfg, "num_classes", default=200))
    default_ignore_index = int(OmegaConf.select(loss_cfg, "ignore_label", default=-100))

    metric_keys = sorted(
        {
            m.group(1)
            for m in re.finditer(
                r'self\.log\("((?:val|debug)/[^"]+)"',
                lz_module_path.read_text(encoding="utf-8"),
            )
        }
    )
    metric_keys_focus = [k for k in metric_keys if k.startswith("val/miou") or "use_text_guidance" in k]

    protocol_payload = {
        "val_split_files": {
            "scannet": split_scannet_val.as_posix(),
            "scannet200": split_scannet_val.as_posix(),
        },
        "train_split_file": split_scannet_train.as_posix(),
        "eval_script_entry": eval_entry.as_posix(),
        "metric_keys_focus": metric_keys_focus,
        "required_metric_key_present": "val/miou_fg_mosaic" in metric_keys,
        "default_ignore_index": default_ignore_index,
        "default_num_classes": default_num_classes,
    }
    write_json(OUT_PROTOCOL, protocol_payload)

    # ---------------------------
    # Step 1: data precheck
    # ---------------------------
    train_scenes = read_split(split_scannet_train)
    val_scenes = read_split(split_scannet_val)
    train_stats = analyze_data_split("train", train_scenes)
    val_stats = analyze_data_split("val", val_scenes)

    data_payload = {
        "data_root": DATA_SCANNET.as_posix(),
        "split_files": {
            "train": split_scannet_train.as_posix(),
            "val": split_scannet_val.as_posix(),
        },
        "train": train_stats,
        "val": val_stats,
    }
    write_json(OUT_DATA_JSON, data_payload)

    data_md_lines = [
        "# Phase7 Data Precheck",
        "",
        f"- Data root: `{DATA_SCANNET}`",
        f"- Train split scenes: {train_stats['total_scenes']}",
        f"- Val split scenes: {val_stats['total_scenes']}",
        "",
        "## Train",
        f"- scene_dir_exists_rate: {train_stats['scene_dir_exists_rate']:.4f}%",
        f"- coord_color_exists_rate: {train_stats['coord_color_exists_rate']:.4f}%",
        f"- segment200_exists_rate: {train_stats['segment200_exists_rate']:.4f}%",
        f"- caption_point_pair_any_rate: {train_stats['caption_point_pair_any_rate']:.4f}%",
        f"- caption_point_pair_segment3d_gathered_rate: {train_stats['caption_point_pair_segment3d_gathered_rate']:.4f}%",
        "",
        "## Val",
        f"- scene_dir_exists_rate: {val_stats['scene_dir_exists_rate']:.4f}%",
        f"- coord_color_exists_rate: {val_stats['coord_color_exists_rate']:.4f}%",
        f"- segment200_exists_rate: {val_stats['segment200_exists_rate']:.4f}%",
        f"- caption_point_pair_any_rate: {val_stats['caption_point_pair_any_rate']:.4f}%",
        f"- caption_point_pair_segment3d_gathered_rate: {val_stats['caption_point_pair_segment3d_gathered_rate']:.4f}%",
        "",
        "## Missing samples (top 100)",
        f"- train_missing_count_reported: {len(train_stats['missing_samples_top100'])}",
        f"- val_missing_count_reported: {len(val_stats['missing_samples_top100'])}",
    ]
    OUT_DATA_MD.write_text("\n".join(data_md_lines) + "\n", encoding="utf-8")

    # ---------------------------
    # Step 2: contract checks
    # ---------------------------
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base=None):
        cfg = compose(
            config_name="train",
            overrides=[
                "data=sc",
                f"paths.data_dir={DATA_ROOT.as_posix()}",
                "paths.root_dir=/root/lz",
                "paths.output_dir=/root/lz_outputs/phase7_precheck_tmp",
                "seed=42",
                "trainer.accelerator=auto",
                "trainer.devices=1",
                "trainer.max_epochs=1",
                "model.loss.weights.seg_loss=1.0",
                "model.loss.weights.instance_loss=0.0",
                "model.loss.weights.hpza_loss=0.0",
                "model.loss.seg_loss.lovasz_weight=0.0",
                "model.loss.seg_loss.use_class_weight=false",
                "sampler.class_freq_path=/root/lz_outputs/class_freq_scannet200.json",
                "data.coord_norm_enable=true",
                "data.color_drop_prob=0.0",
                "data.color_drop_mode=gray",
            ],
        )

    backbone_out_dim = int(cfg.model.net.backbone.out_dim)
    decoder_in_dim = int(cfg.model.net.decoder.in_dim)
    num_classes_cfg = int(cfg.data.num_classes)
    ignore_idx_cfg = int(cfg.model.loss.ignore_label)
    dim_match = backbone_out_dim == decoder_in_dim

    train_dataset = hydra.utils.instantiate(cfg.data.train_dataset)
    mapper_stats = flatten_mapper_stats(
        mapper=np.asarray(train_dataset.valid_class_mapper),
        ignore_index=int(train_dataset.ignore_label),
        num_classes=num_classes_cfg,
    )

    contract_payload = {
        "current_backbone": {
            "name": str(cfg.model.net.backbone._target_),
            "input_contract": {"coord": "(N,3)", "color": "(N,3)"},
            "output_contract": {"point_feats": f"(N,{backbone_out_dim})"},
            "backbone_out_dim": backbone_out_dim,
            "decoder_in_dim": decoder_in_dim,
            "dim_match": bool(dim_match),
        },
        "class_contract": {
            "num_classes_cfg": num_classes_cfg,
            "ignore_index_cfg": ignore_idx_cfg,
            "dataset_ignore_index": int(train_dataset.ignore_label),
            "valid_class_mapper_check": mapper_stats,
        },
        "candidate_backbone_static_notes": {
            "candidate": "PointNet++ (not integrated yet)",
            "must_preserve": [
                "input keys: coord/color",
                "output key: point_feats",
                "point_feats channel C must match decoder.in_dim",
                "num_classes=200 and ignore_index chain unchanged",
            ],
            "likely_touch_files": [
                "/root/lz/src/models/networks/lz_backbone/mssoe_fpn.py",
                "/root/lz/src/models/networks/lz_model.py",
                "/root/lz/configs/model/lz.yaml",
            ],
        },
    }
    write_json(OUT_CONTRACT, contract_payload)

    # dynamic smoke (1 batch)
    smoke_payload: Dict[str, Any] = {
        "forward_backward_success": False,
        "error": None,
        "device": None,
        "single_step_time_sec": None,
        "gpu_peak_memory_mb": None,
        "new_backbone_smoke": "not_integrated",
        "new_backbone_memory_budget_suggestion_mb": {
            "target_peak_upper_bound_mb": 20000,
            "hard_cap_mb": 23000,
        },
    }

    try:
        datamodule = hydra.utils.instantiate(cfg.data)
        datamodule.setup("fit")
        loader = datamodule.train_dataloader()
        batch = next(iter(loader))

        model = hydra.utils.instantiate(cfg.model, _recursive_=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        smoke_payload["device"] = str(device)
        model.to(device)
        batch = move_to_device(batch, device)
        model.train()

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device=device)

        start = time.perf_counter()
        outputs = model(batch)
        seg_logits = outputs["seg_logits"]
        seg_labels = batch.get("segment")
        loss = model.seg_loss(seg_logits, seg_labels)
        loss.backward()
        end = time.perf_counter()

        smoke_payload["forward_backward_success"] = True
        smoke_payload["single_step_time_sec"] = float(end - start)
        smoke_payload["loss_value"] = float(loss.detach().cpu().item())
        if device.type == "cuda":
            smoke_payload["gpu_peak_memory_mb"] = float(
                torch.cuda.max_memory_allocated(device=device) / 1024.0 / 1024.0
            )
        else:
            smoke_payload["gpu_peak_memory_mb"] = None
            smoke_payload["gpu_note"] = "CUDA unavailable in current (no-card) session."
    except Exception as e:
        smoke_payload["error"] = f"{type(e).__name__}: {e}"

    write_json(OUT_SMOKE, smoke_payload)

    # ---------------------------
    # Step 3: single-GPU plan
    # ---------------------------
    plan_yaml = {
        "backbone_candidate": "SpUNet34C (first) / PointNet++ (backup)",
        "train_first_run": {
            "batch_size": 2,
            "accumulate_grad_batches": 1,
            "lr": 0.001,
            "amp": True,
            "grad_clip_val": 1.0,
            "max_epochs": 15,
        },
        "point_budget": {
            "max_points_per_sample_hint": 120000,
            "voxel_size_hint_if_sparseconv": 0.02,
            "max_voxels_hint_if_sparseconv": 250000,
        },
        "memory_budget_strategy": {
            "target_peak_mb": 20000,
            "hard_cap_mb": 23000,
            "oom_fallback_order": [
                "batch_size 2 -> 1",
                "enable/keep AMP",
                "reduce max_points_per_sample by 20%",
            ],
        },
    }
    OUT_PLAN_YAML.write_text(OmegaConf.to_yaml(plan_yaml), encoding="utf-8")

    plan_md_lines = [
        "# Phase7 4090 Plan",
        "",
        "- 目标：换骨干后的首跑避免OOM并保持可比较口径。",
        "- 首跑建议：batch_size=2, accumulate_grad_batches=1, lr=1e-3, amp=true, grad_clip=1.0。",
        "- 点数预算：max_points≈120k；若稀疏卷积可用 voxel_size≈0.02 与 max_voxels≈250k。",
        "- 显存策略：目标峰值<20GB，硬上限23GB；OOM时按 batch→points 顺序回退。",
    ]
    OUT_PLAN_MD.write_text("\n".join(plan_md_lines) + "\n", encoding="utf-8")

    # ---------------------------
    # Step 4: GO/NO-GO
    # ---------------------------
    no_go_reasons = []

    train_hit = float(train_stats["scene_dir_exists_rate"])
    val_hit = float(val_stats["scene_dir_exists_rate"])
    if train_hit < 99.0 or val_hit < 99.0:
        no_go_reasons.append(
            f"Split命中率不足99%（train={train_hit:.4f}%, val={val_hit:.4f}%）"
        )

    train_seg = float(train_stats["segment200_exists_rate"])
    val_seg = float(val_stats["segment200_exists_rate"])
    if train_seg < 99.0 or val_seg < 99.0:
        no_go_reasons.append(
            f"segment200覆盖率异常（train={train_seg:.4f}%, val={val_seg:.4f}%）"
        )

    if (not dim_match) or (not mapper_stats["in_range_or_ignore"]):
        no_go_reasons.append("契约检查未通过（维度或类别映射不一致）")

    if not smoke_payload["forward_backward_success"]:
        no_go_reasons.append(f"1-batch smoke失败：{smoke_payload['error']}")
    else:
        gpu_peak = smoke_payload.get("gpu_peak_memory_mb", None)
        if gpu_peak is None:
            no_go_reasons.append("当前无GPU，无法确认峰值显存是否可控")
        elif gpu_peak > 23000.0:
            no_go_reasons.append(f"1-batch峰值显存超预算：{gpu_peak:.2f} MB")

    decision = "GO" if len(no_go_reasons) == 0 else "NO-GO"
    decision_payload = {
        "decision": decision,
        "checks": {
            "protocol_locked": True,
            "data_split_hit_rate_train": train_hit,
            "data_split_hit_rate_val": val_hit,
            "segment200_coverage_train": train_seg,
            "segment200_coverage_val": val_seg,
            "contract_dim_match": bool(dim_match),
            "mapper_in_range_or_ignore": bool(mapper_stats["in_range_or_ignore"]),
            "smoke_forward_backward_success": bool(smoke_payload["forward_backward_success"]),
            "smoke_gpu_peak_memory_mb": smoke_payload.get("gpu_peak_memory_mb"),
        },
        "no_go_reasons": no_go_reasons,
    }
    write_json(OUT_DECISION, decision_payload)

    summary_lines = [
        "# PHASE7 PRECHECK SUMMARY",
        "",
        f"- decision: {decision}",
        f"- train split命中率: {train_hit:.4f}%",
        f"- val split命中率: {val_hit:.4f}%",
        f"- train segment200覆盖率: {train_seg:.4f}%",
        f"- val segment200覆盖率: {val_seg:.4f}%",
        f"- contract dim_match: {dim_match}",
        f"- mapper_in_range_or_ignore: {mapper_stats['in_range_or_ignore']}",
        f"- smoke_forward_backward_success: {smoke_payload['forward_backward_success']}",
        f"- smoke_gpu_peak_memory_mb: {smoke_payload.get('gpu_peak_memory_mb')}",
        "",
        "## Risks",
    ]
    if no_go_reasons:
        for r in no_go_reasons[:5]:
            summary_lines.append(f"- {r}")
    else:
        summary_lines.append("- 无阻塞项，可进入换骨干执行阶段。")

    OUT_SUMMARY.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "decision": decision,
                "no_go_reasons": no_go_reasons,
                "outputs": [
                    OUT_PROTOCOL.as_posix(),
                    OUT_DATA_JSON.as_posix(),
                    OUT_DATA_MD.as_posix(),
                    OUT_CONTRACT.as_posix(),
                    OUT_SMOKE.as_posix(),
                    OUT_PLAN_YAML.as_posix(),
                    OUT_PLAN_MD.as_posix(),
                    OUT_DECISION.as_posix(),
                    OUT_SUMMARY.as_posix(),
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
