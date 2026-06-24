from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src.utils import (
    RankedLogger,
    extras,
    get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


def _extract_state_dict_from_checkpoint(checkpoint: Any) -> Optional[Dict[str, torch.Tensor]]:
    if not isinstance(checkpoint, dict):
        return None

    candidate_keys = ["state_dict", "model_state_dict", "model", "net", "network"]
    for key in candidate_keys:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value

    if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    return None


def _candidate_model_keys(ckpt_key: str) -> List[str]:
    candidates = [ckpt_key]
    prefix_rules = ["module.", "model.", "net."]
    for prefix in prefix_rules:
        if ckpt_key.startswith(prefix):
            candidates.append(ckpt_key[len(prefix) :])
    return candidates


def _load_init_weights(model: LightningModule, ckpt_path: str) -> Dict[str, int]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = _extract_state_dict_from_checkpoint(checkpoint)
    if state_dict is None:
        raise ValueError(f"Unsupported checkpoint format for init load: {ckpt_path}")

    model_state = model.state_dict()
    matched: Dict[str, torch.Tensor] = {}
    skipped_shape = 0
    skipped_missing = 0

    for ckpt_key, ckpt_tensor in state_dict.items():
        if not torch.is_tensor(ckpt_tensor):
            continue
        found_match = False
        for model_key in _candidate_model_keys(ckpt_key):
            model_tensor = model_state.get(model_key)
            if model_tensor is None:
                continue
            if tuple(model_tensor.shape) != tuple(ckpt_tensor.shape):
                skipped_shape += 1
                found_match = True
                break
            matched[model_key] = ckpt_tensor
            found_match = True
            break
        if not found_match:
            skipped_missing += 1

    load_result = model.load_state_dict(matched, strict=False)
    return {
        "matched_keys": len(matched),
        "missing_model_keys": len(load_result.missing_keys),
        "unexpected_model_keys": len(load_result.unexpected_keys),
        "skipped_shape": skipped_shape,
        "skipped_missing": skipped_missing,
    }


@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    extras(cfg)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, _recursive_=False)

    init_ckpt_path = cfg.get("init_ckpt_path")
    if init_ckpt_path and not cfg.get("ckpt_path"):
        log.info(f"Loading initial weights from <{init_ckpt_path}> (weights-only mode)")
        stats = _load_init_weights(model, init_ckpt_path)
        log.info(
            "Init load stats | "
            f"matched={stats['matched_keys']} "
            f"missing_model={stats['missing_model_keys']} "
            f"unexpected_model={stats['unexpected_model_keys']} "
            f"skipped_shape={stats['skipped_shape']} "
            f"skipped_missing={stats['skipped_missing']}"
        )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train", True):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        log.info("Finished training!")

    train_metrics = trainer.callback_metrics
    test_metrics = {}

    if cfg.get("test", False):
        log.info("Starting testing!")
        trainer.validate(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))
        test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}
    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> Optional[float]:
    metric_dict, _ = train(cfg)
    metric_value = get_metric_value(metric_dict, cfg.get("optimized_metric", "val/miou"))
    return metric_value


if __name__ == "__main__":
    main()
