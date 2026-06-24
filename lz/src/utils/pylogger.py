from __future__ import annotations

import logging
import os
from typing import Optional

import torch


def _is_rank_zero() -> bool:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


class RankedLogger(logging.LoggerAdapter):
    """Logger that only emits on rank 0 by default."""

    def __init__(self, name: str, rank_zero_only: bool = True) -> None:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                fmt="[%(asctime)s][%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))
        super().__init__(logger, {})
        self.rank_zero_only = rank_zero_only

    def log(self, level, msg, *args, **kwargs) -> None:  # type: ignore[override]
        if self.rank_zero_only and not _is_rank_zero():
            return
        super().log(level, msg, *args, **kwargs)
