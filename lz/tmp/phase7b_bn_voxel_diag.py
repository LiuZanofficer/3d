#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from src.models.utils.metrics import compute_iou_from_conf, update_confusion_matrix
from src.models.utils.structure import Point

OUT_DIR = Path('/root/lz_outputs/phase7B_bn_voxel_diag')
RUNS = {
    'runA': Path('/root/lz_outputs/phase7A_recheck_debug1_aligned'),
    'runB': Path('/root/lz_outputs/phase7A_recheck_debug1_in6'),
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def to_jsonable(x: Any) -> Any:
    if isinstance(x, (np.generic,)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    return x


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=to_jsonable), encoding='utf-8')


def has_gpu() -> bool:
    try:
        out = subprocess.run(['nvidia-smi'], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if out.returncode != 0:
            return False
    except Exception:
        return False
    return bool(torch.cuda.is_available())


@dataclass
class RunAssets:
    name: str
    run_dir: Path
    ckpt_path: Path
    cfg_path: Path
    cfg: Any


def find_latest_cfg(run_dir: Path) -> Path:
    cfg_paths = list(run_dir.glob('**/.hydra/config.yaml'))
    if not cfg_paths:
        raise FileNotFoundError(f'No .hydra/config.yaml under {run_dir}')
    cfg_paths.sort(key=lambda p: p.stat().st_mtime)
    return cfg_paths[-1]


def build_assets(name: str, run_dir: Path) -> RunAssets:
    ckpt = run_dir / 'checkpoints' / 'last.ckpt'
    if not ckpt.exists():
        raise FileNotFoundError(f'Missing checkpoint: {ckpt}')
    cfg_path = find_latest_cfg(run_dir)
    cfg = OmegaConf.load(str(cfg_path))
    return RunAssets(name=name, run_dir=run_dir, ckpt_path=ckpt, cfg_path=cfg_path, cfg=cfg)


def move_to_device(obj: Any, device: torch.device) -> Any:
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    return obj


def get_dataset_meta(val_loader: DataLoader) -> Tuple[int, int, List[str], List[int]]:
    ds = val_loader.dataset
    class_names = list(getattr(ds, 'CLASS_LABELS', []))
    num_classes = len(class_names)
    ignore_label = int(getattr(ds, 'ignore_label', -100))
    fg_idx = [int(x) for x in getattr(ds, 'fg_class_idx', []) if 0 <= int(x) < num_classes]
    if not fg_idx:
        fg_idx = [
            idx
            for idx, cname in enumerate(class_names)
            if str(cname) not in {'wall', 'floor', 'ceiling'} and 'other' not in str(cname)
        ]
    if not fg_idx:
        fg_idx = list(range(num_classes))
    return num_classes, ignore_label, class_names, fg_idx


def pred_stats(preds: torch.Tensor, labels: torch.Tensor, ignore_label: int) -> Tuple[float, float]:
    preds = preds.view(-1).long()
    labels = labels.view(-1).long()
    valid = labels != int(ignore_label)
    vp = preds[valid]
    if vp.numel() == 0:
        return 0.0, 0.0
    unique_cnt = float(torch.unique(vp).numel())
    max_id = int(vp.max().item())
    bc = torch.bincount(vp, minlength=max_id + 1)
    top1 = float(bc.max().item() / vp.numel())
    return unique_cnt, top1


def compute_metrics_from_conf(conf: torch.Tensor, fg_idx: List[int]) -> Dict[str, float]:
    conf_f = conf.float()
    iou = compute_iou_from_conf(conf_f)
    row_sum = conf_f.sum(dim=1)
    col_sum = conf_f.sum(dim=0)
    union = row_sum + col_sum - torch.diag(conf_f)

    miou = float(iou.mean().item())
    gt_mask = row_sum > 0
    union_mask = union > 0
    miou_present_gt = float(iou[gt_mask].mean().item()) if gt_mask.any() else 0.0
    miou_present_union = float(iou[union_mask].mean().item()) if union_mask.any() else 0.0
    miou_fg_mosaic = float(iou[fg_idx].mean().item()) if fg_idx else miou

    return {
        'miou': miou,
        'miou_present_gt': miou_present_gt,
        'miou_present_union': miou_present_union,
        'miou_fg_mosaic': miou_fg_mosaic,
    }


def set_bn_eval_only(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def instantiate_datamodule_and_model(cfg: Any):
    datamodule = hydra.utils.instantiate(cfg.data)
    datamodule.setup('fit')
    val_loader = datamodule.val_dataloader()[0]

    model = hydra.utils.instantiate(cfg.model, _recursive_=False)
    return datamodule, val_loader, model


def evaluate_mode(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    val_loader: DataLoader,
    mode: str,
    device: torch.device,
) -> Dict[str, Any]:
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if mode == 'eval':
        model.eval()
    elif mode == 'train':
        model.train()
    elif mode == 'train_bn_eval':
        model.train()
        set_bn_eval_only(model)
    else:
        raise ValueError(f'Unknown mode: {mode}')

    model.to(device)

    num_classes, ignore_label, class_names, fg_idx = get_dataset_meta(val_loader)
    conf = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)

    uniq_list: List[float] = []
    top1_list: List[float] = []
    use_tg_values: List[float] = []
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch = move_to_device(batch, device)
            outputs = model(batch)
            logits = outputs['seg_logits']
            labels = batch['segment'].long()
            preds = torch.argmax(logits, dim=1)

            conf = update_confusion_matrix(
                conf=conf,
                preds=preds,
                labels=labels,
                num_classes=num_classes,
                ignore_label=ignore_label,
            )

            unique_cnt, top1 = pred_stats(preds, labels, ignore_label)
            uniq_list.append(unique_cnt)
            top1_list.append(top1)
            dbg = outputs.get('debug_use_text_guidance', None)
            if dbg is not None:
                use_tg_values.append(float(dbg.detach().float().mean().item()))
            num_batches += 1

    m = compute_metrics_from_conf(conf, fg_idx)
    m.update(
        {
            'pred_unique_mean': float(np.mean(uniq_list) if uniq_list else 0.0),
            'pred_top1_ratio_mean': float(np.mean(top1_list) if top1_list else 0.0),
            'num_batches': int(num_batches),
            'debug_use_text_guidance_mean': float(np.mean(use_tg_values) if use_tg_values else 0.0),
            'load_state_missing_keys': [str(x) for x in missing],
            'load_state_unexpected_keys': [str(x) for x in unexpected],
        }
    )
    return m


def bn_running_stats_from_ckpt(ckpt_path: Path) -> Dict[str, Any]:
    ckpt = torch.load(str(ckpt_path), map_location='cpu')
    state = ckpt.get('state_dict', ckpt)

    running_var_all: List[np.ndarray] = []
    running_mean_abs_all: List[np.ndarray] = []
    num_batches_all: List[float] = []
    abnormal_layers: List[Dict[str, Any]] = []

    prefixes = set()
    for k in state.keys():
        if k.endswith('running_mean'):
            prefixes.add(k[:-len('running_mean')])

    for pref in sorted(prefixes):
        mean_key = pref + 'running_mean'
        var_key = pref + 'running_var'
        nbt_key = pref + 'num_batches_tracked'
        mean_t = state.get(mean_key)
        var_t = state.get(var_key)
        nbt_t = state.get(nbt_key)
        if mean_t is None or var_t is None:
            continue

        mean_np = mean_t.detach().cpu().float().numpy().reshape(-1)
        var_np = var_t.detach().cpu().float().numpy().reshape(-1)
        running_mean_abs_all.append(np.abs(mean_np))
        running_var_all.append(var_np)

        nbt = float(nbt_t.detach().cpu().item()) if nbt_t is not None else 0.0
        num_batches_all.append(nbt)

        var_min = float(np.nanmin(var_np)) if var_np.size else 0.0
        var_max = float(np.nanmax(var_np)) if var_np.size else 0.0
        nan_count = int(np.isnan(var_np).sum())

        is_abnormal = bool(nan_count > 0 or var_min < 1e-8)
        if is_abnormal:
            abnormal_layers.append(
                {
                    'layer': pref.rstrip('.'),
                    'var_min': var_min,
                    'var_max': var_max,
                    'var_nan_count': nan_count,
                    'num_batches_tracked': nbt,
                }
            )

    if running_var_all:
        all_var = np.concatenate(running_var_all)
        all_mean_abs = np.concatenate(running_mean_abs_all)
    else:
        all_var = np.array([], dtype=np.float32)
        all_mean_abs = np.array([], dtype=np.float32)

    if num_batches_all:
        nbt_arr = np.array(num_batches_all, dtype=np.float32)
    else:
        nbt_arr = np.array([], dtype=np.float32)

    def pct(arr: np.ndarray, p: float, default: float = 0.0) -> float:
        if arr.size == 0:
            return default
        return float(np.nanpercentile(arr, p))

    def arr_min(arr: np.ndarray) -> float:
        return float(np.nanmin(arr)) if arr.size else 0.0

    def arr_max(arr: np.ndarray) -> float:
        return float(np.nanmax(arr)) if arr.size else 0.0

    payload = {
        'ckpt_path': str(ckpt_path),
        'bn_layer_count': int(len(prefixes)),
        'running_var': {
            'min': arr_min(all_var),
            'p1': pct(all_var, 1),
            'median': pct(all_var, 50),
            'p99': pct(all_var, 99),
            'max': arr_max(all_var),
            'nan_count': int(np.isnan(all_var).sum()) if all_var.size else 0,
        },
        'running_mean_abs': {
            'mean': float(np.nanmean(all_mean_abs)) if all_mean_abs.size else 0.0,
            'p99': pct(all_mean_abs, 99),
            'max': arr_max(all_mean_abs),
        },
        'num_batches_tracked': {
            'min': arr_min(nbt_arr),
            'median': pct(nbt_arr, 50),
            'max': arr_max(nbt_arr),
        },
        'abnormal_layers': abnormal_layers,
    }
    return payload


def voxel_consistency_stats(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(device)

    backbone = model.net.backbone
    if not hasattr(backbone, 'grid_size'):
        raise RuntimeError('Backbone does not expose grid_size')

    batches: List[Dict[str, Any]] = []
    unique_voxels_list: List[float] = []
    ppv_mean_list: List[float] = []
    ppv_p95_list: List[float] = []
    ppv_max_list: List[float] = []
    ratio_list: List[float] = []

    with torch.no_grad():
        for idx, batch in enumerate(val_loader):
            batch = move_to_device(batch, device)
            coord = batch['coord'].float()
            color = batch.get('color', None)
            offset = batch.get('offset')
            if offset is None:
                offset = torch.tensor([0, coord.shape[0]], device=device, dtype=torch.long)
            else:
                offset = offset.to(device=device, dtype=torch.long)

            feat = backbone._build_feat(coord, color)
            point = Point(coord=coord, feat=feat, offset=offset, grid_size=float(backbone.grid_size))
            point.sparsify(pad=int(getattr(backbone, 'pad', 128)))

            indices = point.sparse_conv_feat.indices[:, 1:].long()
            unique_voxels = int(indices.shape[0])
            n_points = int(coord.shape[0])

            v2p = point.v2p_map.long()
            v2p_min = int(v2p.min().item()) if v2p.numel() > 0 else -1
            v2p_max = int(v2p.max().item()) if v2p.numel() > 0 else -1
            v2p_oob = bool((v2p < 0).any().item() or (v2p >= max(unique_voxels, 1)).any().item())

            bincount = torch.bincount(v2p, minlength=max(unique_voxels, 1)).float()
            ppv_mean = float(bincount.mean().item()) if bincount.numel() else 0.0
            ppv_p95 = float(torch.quantile(bincount, 0.95).item()) if bincount.numel() else 0.0
            ppv_max = float(bincount.max().item()) if bincount.numel() else 0.0

            coord_mean = coord.mean(dim=0)
            coord_std = coord.std(dim=0, unbiased=False)
            coord_min = coord.min(dim=0).values
            coord_max = coord.max(dim=0).values
            voxel_min = indices.min(dim=0).values if indices.numel() else torch.zeros(3, device=device)
            voxel_max = indices.max(dim=0).values if indices.numel() else torch.zeros(3, device=device)

            offset_vals = offset.detach().cpu().tolist()
            offset_ok = (
                len(offset_vals) >= 2
                and int(offset_vals[0]) == 0
                and int(offset_vals[-1]) == n_points
                and all(int(offset_vals[i]) < int(offset_vals[i + 1]) for i in range(len(offset_vals) - 1))
            )

            ratio = float(n_points / max(unique_voxels, 1))

            entry = {
                'batch_index': idx,
                'coord_mean': [float(x) for x in coord_mean.detach().cpu().tolist()],
                'coord_std': [float(x) for x in coord_std.detach().cpu().tolist()],
                'coord_min': [float(x) for x in coord_min.detach().cpu().tolist()],
                'coord_max': [float(x) for x in coord_max.detach().cpu().tolist()],
                'grid_size': float(backbone.grid_size),
                'voxel_index_min': [int(x) for x in voxel_min.detach().cpu().tolist()],
                'voxel_index_max': [int(x) for x in voxel_max.detach().cpu().tolist()],
                'unique_voxels': unique_voxels,
                'n_points': n_points,
                'n_points_to_unique_voxels_ratio': ratio,
                'points_per_voxel_mean': ppv_mean,
                'points_per_voxel_p95': ppv_p95,
                'points_per_voxel_max': ppv_max,
                'offset_shape': list(offset.shape),
                'offset_values': [int(x) for x in offset_vals],
                'offset_valid': bool(offset_ok),
                'v2p_map_len': int(v2p.numel()),
                'v2p_map_min': v2p_min,
                'v2p_map_max': v2p_max,
                'v2p_map_oob': bool(v2p_oob),
            }
            batches.append(entry)
            unique_voxels_list.append(float(unique_voxels))
            ppv_mean_list.append(ppv_mean)
            ppv_p95_list.append(ppv_p95)
            ppv_max_list.append(ppv_max)
            ratio_list.append(ratio)

    payload = {
        'num_batches': len(batches),
        'batches': batches,
        'summary': {
            'unique_voxels_mean': float(np.mean(unique_voxels_list) if unique_voxels_list else 0.0),
            'points_per_voxel_mean': float(np.mean(ppv_mean_list) if ppv_mean_list else 0.0),
            'points_per_voxel_p95_mean': float(np.mean(ppv_p95_list) if ppv_p95_list else 0.0),
            'points_per_voxel_max_mean': float(np.mean(ppv_max_list) if ppv_max_list else 0.0),
            'n_points_to_unique_voxels_ratio_mean': float(np.mean(ratio_list) if ratio_list else 0.0),
            'has_offset_invalid': bool(any(not b['offset_valid'] for b in batches)),
            'has_v2p_oob': bool(any(b['v2p_map_oob'] for b in batches)),
        },
    }
    return payload


def bn_confirmed(run_payload: Dict[str, Any]) -> bool:
    eval_m = run_payload['modes']['eval']['miou_present_gt']
    train_m = run_payload['modes']['train']['miou_present_gt']
    train_bn_eval_m = run_payload['modes']['train_bn_eval']['miou_present_gt']
    gain_ok = (train_m >= max(2.0 * eval_m, eval_m + 0.02))
    delta_train = abs(train_m - eval_m)
    delta_bn_eval = abs(train_bn_eval_m - eval_m)
    fallback_ok = delta_bn_eval <= max(0.005, 0.5 * delta_train)
    return bool(gain_ok and fallback_ok)


def voxel_confirmed(voxel_payload: Dict[str, Any]) -> bool:
    s = voxel_payload.get('summary', {})
    if s.get('has_offset_invalid', False) or s.get('has_v2p_oob', False):
        return True
    ratio = float(s.get('n_points_to_unique_voxels_ratio_mean', 0.0))
    ppv_mean = float(s.get('points_per_voxel_mean', 0.0))
    ppv_p95 = float(s.get('points_per_voxel_p95_mean', 0.0))
    unique_mean = float(s.get('unique_voxels_mean', 0.0))
    if unique_mean <= 1:
        return True
    if ratio > 50 or ppv_mean > 20 or ppv_p95 > 50:
        return True
    return False


def main() -> None:
    ensure_dir(OUT_DIR)

    if not has_gpu():
        decision = {'final_decision': 'BLOCKED_NO_GPU', 'reason': 'GPU not available'}
        write_json(OUT_DIR / 'phase7B_diagnosis_compare.json', decision)
        (OUT_DIR / 'PHASE7B_DIAG_SUMMARY.md').write_text(
            '# PHASE7B DIAG SUMMARY\n\n- FINAL_DECISION: BLOCKED_NO_GPU\n',
            encoding='utf-8',
        )
        print('BLOCKED_NO_GPU')
        return

    device = torch.device('cuda:0')
    run_results: Dict[str, Any] = {}

    for run_name, run_dir in RUNS.items():
        assets = build_assets(run_name, run_dir)
        datamodule, val_loader, model = instantiate_datamodule_and_model(assets.cfg)

        ckpt = torch.load(str(assets.ckpt_path), map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        mode_payload = {
            'run_name': run_name,
            'run_dir': str(run_dir),
            'ckpt_path': str(assets.ckpt_path),
            'cfg_path': str(assets.cfg_path),
            'modes': {},
        }
        for mode in ['eval', 'train', 'train_bn_eval']:
            mode_payload['modes'][mode] = evaluate_mode(
                model=model,
                state_dict=state_dict,
                val_loader=val_loader,
                mode=mode,
                device=device,
            )

        write_json(OUT_DIR / f'bn_mode_swap_{run_name}.json', mode_payload)

        bn_payload = bn_running_stats_from_ckpt(assets.ckpt_path)
        write_json(OUT_DIR / f'bn_running_stats_{run_name}.json', bn_payload)

        voxel_payload = voxel_consistency_stats(
            model=model,
            state_dict=state_dict,
            val_loader=val_loader,
            device=device,
        )
        voxel_payload.update(
            {
                'run_name': run_name,
                'run_dir': str(run_dir),
                'ckpt_path': str(assets.ckpt_path),
                'cfg_path': str(assets.cfg_path),
            }
        )
        write_json(OUT_DIR / f'voxel_stats_{run_name}.json', voxel_payload)

        run_results[run_name] = {
            'bn_mode_swap': mode_payload,
            'bn_running_stats': bn_payload,
            'voxel_stats': voxel_payload,
        }

    bn_hit = any(bn_confirmed(run_results[r]['bn_mode_swap']) for r in run_results)
    voxel_hit = any(voxel_confirmed(run_results[r]['voxel_stats']) for r in run_results)

    if bn_hit and voxel_hit:
        final_decision = 'BOTH_CONFIRMED'
    elif bn_hit:
        final_decision = 'BN_CONFIRMED'
    elif voxel_hit:
        final_decision = 'VOXEL_CONFIRMED'
    else:
        final_decision = 'NEITHER_CONFIRMED'

    compare = {
        'final_decision': final_decision,
        'bn_confirmed_by_run': {
            r: bool(bn_confirmed(run_results[r]['bn_mode_swap'])) for r in run_results
        },
        'voxel_confirmed_by_run': {
            r: bool(voxel_confirmed(run_results[r]['voxel_stats'])) for r in run_results
        },
        'runs': {
            r: {
                'bn_mode_swap_file': str(OUT_DIR / f'bn_mode_swap_{r}.json'),
                'bn_running_stats_file': str(OUT_DIR / f'bn_running_stats_{r}.json'),
                'voxel_stats_file': str(OUT_DIR / f'voxel_stats_{r}.json'),
                'mode_metrics': run_results[r]['bn_mode_swap']['modes'],
                'voxel_summary': run_results[r]['voxel_stats']['summary'],
            }
            for r in run_results
        },
    }
    write_json(OUT_DIR / 'phase7B_diagnosis_compare.json', compare)

    lines = [
        '# PHASE7B DIAG SUMMARY',
        '',
        f'- FINAL_DECISION: {final_decision}',
        '',
        '## BN Mode Swap (miou_present_gt)',
    ]
    for r in ['runA', 'runB']:
        mode = run_results[r]['bn_mode_swap']['modes']
        lines.append(
            f"- {r}: eval={mode['eval']['miou_present_gt']:.6f}, "
            f"train={mode['train']['miou_present_gt']:.6f}, "
            f"train_bn_eval={mode['train_bn_eval']['miou_present_gt']:.6f}"
        )
    lines += ['', '## Voxel Summary']
    for r in ['runA', 'runB']:
        s = run_results[r]['voxel_stats']['summary']
        lines.append(
            f"- {r}: unique_voxels_mean={s['unique_voxels_mean']:.2f}, "
            f"ppv_mean={s['points_per_voxel_mean']:.4f}, ppv_p95_mean={s['points_per_voxel_p95_mean']:.4f}, "
            f"ratio_mean={s['n_points_to_unique_voxels_ratio_mean']:.4f}, "
            f"offset_invalid={s['has_offset_invalid']}, v2p_oob={s['has_v2p_oob']}"
        )
    (OUT_DIR / 'PHASE7B_DIAG_SUMMARY.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(final_decision)


if __name__ == '__main__':
    main()
