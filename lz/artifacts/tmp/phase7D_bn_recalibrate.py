import json
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import lightning as L
import torch
from lightning import LightningDataModule, LightningModule
from lightning.fabric.utilities.apply_func import move_data_to_device
from omegaconf import DictConfig
from torch.nn.modules.batchnorm import _BatchNorm


def _extract_state_dict_from_checkpoint(checkpoint: Any) -> Optional[Dict[str, torch.Tensor]]:
    if not isinstance(checkpoint, dict):
        return None
    for key in ["state_dict", "model_state_dict", "model", "net", "network"]:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    return None


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    ckpt_path = cfg.get("ckpt_path")
    if not ckpt_path:
        raise ValueError("ckpt_path is required for BN recalibration")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict_from_checkpoint(checkpoint)
    if state_dict is None:
        raise ValueError(f"Unsupported checkpoint format: {ckpt_path}")

    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"strict load failed, missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available for recalibration")

    device = torch.device("cuda:0")
    model = model.to(device)
    model.eval()

    bn_modules = []
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.train()
            bn_modules.append(module)

    datamodule.setup(stage="fit")
    train_loader = datamodule.train_dataloader()

    max_iters = int(cfg.get("bn_recalib_iters", 300))
    processed = 0

    with torch.no_grad():
        for batch in train_loader:
            if processed >= max_iters:
                break
            batch = move_data_to_device(batch, device)
            _ = model(batch)
            processed += 1

    for module in bn_modules:
        module.eval()

    out_ckpt = Path(cfg.get("recalib_ckpt_out"))
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    checkpoint["state_dict"] = model.state_dict()
    torch.save(checkpoint, out_ckpt)

    bn_running_var = []
    bn_running_mean_abs = []
    for module in bn_modules:
        if getattr(module, "running_var", None) is not None:
            bn_running_var.extend(module.running_var.detach().float().cpu().tolist())
        if getattr(module, "running_mean", None) is not None:
            bn_running_mean_abs.extend(module.running_mean.detach().float().abs().cpu().tolist())

    def _safe_stats(values):
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None}
        tensor = torch.tensor(values, dtype=torch.float32)
        return {
            "count": int(tensor.numel()),
            "min": float(tensor.min().item()),
            "max": float(tensor.max().item()),
            "mean": float(tensor.mean().item()),
        }

    stats = {
        "base_ckpt": str(ckpt_path),
        "recalib_ckpt": str(out_ckpt),
        "max_iters": max_iters,
        "processed_iters": processed,
        "bn_module_count": len(bn_modules),
        "bn_running_var_stats": _safe_stats(bn_running_var),
        "bn_running_mean_abs_stats": _safe_stats(bn_running_mean_abs),
    }

    stats_path = Path(cfg.get("bn_recalib_stats_out"))
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
