from __future__ import annotations

import functools
import os
import random
from typing import Any, Dict, Iterable, List, Optional

import hydra
import numpy as np
import torch
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def seed_everything(seed: int, workers: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if workers:
        os.environ["PYTHONHASHSEED"] = str(seed)


def task_wrapper(task_func):
    @functools.wraps(task_func)
    def wrapped(*args, **kwargs):
        try:
            return task_func(*args, **kwargs)
        except Exception:
            log.exception("Task failed.")
            raise

    return wrapped


def extras(cfg: DictConfig) -> None:
    # Placeholder for future optional extras (e.g., warnings filters).
    _ = cfg


def get_metric_value(metric_dict: Dict[str, Any], metric_name: str) -> Optional[float]:
    if metric_name in metric_dict:
        value = metric_dict[metric_name]
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)
    return None


def instantiate_callbacks(callbacks_cfg: Optional[DictConfig]) -> List[Callback]:
    if not callbacks_cfg:
        return []
    callbacks: List[Callback] = []
    for _, cb_conf in callbacks_cfg.items():
        if "_target_" in cb_conf:
            callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def instantiate_loggers(logger_cfg: Optional[DictConfig]) -> List[Logger]:
    if not logger_cfg:
        return []
    loggers: List[Logger] = []
    for _, lg_conf in logger_cfg.items():
        if "_target_" in lg_conf:
            loggers.append(hydra.utils.instantiate(lg_conf))
    return loggers


def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    cfg = object_dict.get("cfg")
    if cfg is None:
        return
    logger_list: List[Logger] = object_dict.get("logger", [])
    if not logger_list:
        return
    hparams = OmegaConf.to_container(cfg, resolve=True)
    for logger in logger_list:
        try:
            logger.log_hyperparams(hparams)  # type: ignore[attr-defined]
        except Exception:
            continue
