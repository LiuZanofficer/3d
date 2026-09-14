import json
import time
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import torch
from lightning import LightningDataModule, LightningModule
from lightning.fabric.utilities.apply_func import move_data_to_device
from omegaconf import DictConfig


def _batch_size_from_batch(batch: dict[str, Any]) -> int:
    offsets = batch.get('offset')
    if offsets is None:
        return 1
    if isinstance(offsets, torch.Tensor):
        return max(int(offsets.numel()) - 1, 1)
    return max(len(offsets) - 1, 1)


@hydra.main(version_base='1.3', config_path='../configs', config_name='train')
def main(cfg: DictConfig) -> None:
    out_path = Path(str(cfg.get('smoke_out', '/root/lz_outputs/phase8B_spunet_bn_recalib/smoke_check.json')))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        'forward_backward_success': False,
        'gpu_peak_memory_mb': None,
        'used_batch_size': None,
        'duration_sec': None,
        'error': None,
    }

    start = time.time()
    try:
        if cfg.get('seed'):
            L.seed_everything(int(cfg.seed), workers=True)

        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is not available')

        datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
        model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

        datamodule.setup(stage='fit')
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        used_batch_size = _batch_size_from_batch(batch)

        device = torch.device('cuda:0')
        model = model.to(device)
        model.train()

        batch = move_data_to_device(batch, device)

        torch.cuda.reset_peak_memory_stats(device)

        outputs = model(batch)
        seg_logits = outputs['seg_logits']
        loss = seg_logits.float().mean()
        loss.backward()

        peak_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 ** 2))

        result['forward_backward_success'] = True
        result['gpu_peak_memory_mb'] = peak_mem_mb
        result['used_batch_size'] = int(used_batch_size)
    except Exception as exc:  # noqa: BLE001
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        result['duration_sec'] = float(time.time() - start)
        out_path.write_text(json.dumps(result, indent=2), encoding='utf-8')

    print(json.dumps(result, indent=2))
    if not result['forward_backward_success']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
