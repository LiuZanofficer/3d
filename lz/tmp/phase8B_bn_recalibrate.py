import json
import time
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
    for key in ['state_dict', 'model_state_dict', 'model', 'net', 'network']:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    return None


def _batch_size_from_batch(batch: dict[str, Any]) -> int:
    offsets = batch.get('offset')
    if offsets is None:
        return 1
    if isinstance(offsets, torch.Tensor):
        return max(int(offsets.numel()) - 1, 1)
    return max(len(offsets) - 1, 1)


@hydra.main(version_base='1.3', config_path='../configs', config_name='eval')
def main(cfg: DictConfig) -> None:
    if cfg.get('seed'):
        L.seed_everything(int(cfg.seed), workers=True)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available for BN recalibration')

    ckpt_path = str(cfg.get('ckpt_path', ''))
    if not ckpt_path:
        raise ValueError('ckpt_path is required')

    out_ckpt = Path(str(cfg.get('recalib_ckpt_out')))
    out_stats = Path(str(cfg.get('bn_recalib_stats_out')))
    max_iters = int(cfg.get('bn_recalib_iters', 300))

    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    out_stats.parent.mkdir(parents=True, exist_ok=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = _extract_state_dict_from_checkpoint(checkpoint)
    if state_dict is None:
        raise ValueError(f'Unsupported checkpoint format: {ckpt_path}')

    model.load_state_dict(state_dict, strict=True)

    device = torch.device('cuda:0')
    model = model.to(device)
    model.eval()

    bn_layers = []
    for module in model.modules():
        if isinstance(module, _BatchNorm):
            module.train()
            bn_layers.append(module)

    datamodule.setup(stage='fit')
    train_loader = datamodule.train_dataloader()

    iter_count = 0
    sample_count = 0
    start = time.time()

    with torch.no_grad():
        for batch in train_loader:
            if iter_count >= max_iters:
                break
            batch = move_data_to_device(batch, device)
            _ = model(batch)
            iter_count += 1
            sample_count += _batch_size_from_batch(batch)

    duration = float(time.time() - start)

    for module in bn_layers:
        module.eval()

    checkpoint['state_dict'] = model.state_dict()
    torch.save(checkpoint, out_ckpt)

    stats = {
        'base_ckpt': ckpt_path,
        'recalib_ckpt': str(out_ckpt),
        'bn_layers': int(len(bn_layers)),
        'iter_count': int(iter_count),
        'sample_count': int(sample_count),
        'duration_sec': duration,
    }
    out_stats.write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
