"""Drop-in adapter wrapping ``MSDSpUNet34C`` for use inside ``LZModel``.

The adapter mirrors the public surface of ``SpUNet34CAdapter``: it consumes a
batch dict (``coord`` / ``color`` / ``offset``) and produces a single
``point_feats`` tensor on the original point order. ``LZModel.text_proj`` and
all decoders downstream therefore work without any modification.
"""

from __future__ import annotations

from typing import Dict, Literal, Optional, Sequence

import torch
import torch.nn as nn

from src.models.networks.spunet.msd_spunet34c import MSDSpUNet34C


class MSDSpUNet34CAdapter(nn.Module):
    """Wrap MSD-SpUNet34C and expose point-level features for LZModel."""

    def __init__(
        self,
        in_channels: int = 6,
        out_dim: int = 128,
        grid_size: float = 0.02,
        pad: int = 128,
        base_channels: int = 32,
        channels: Sequence[int] = (32, 64, 128, 256, 256, 128, 96, 96),
        layers: Sequence[int] = (2, 3, 4, 6, 2, 2, 2, 2),
        # Fine branch
        fine_grid_size: float = 0.01,
        fine_channels: Sequence[int] = (16, 32, 64, 64, 32, 16),
        fine_layers: Sequence[int] = (1, 1, 1, 1, 1, 1),
        fine_out_dim: int = 64,
        # MSD-specific
        lkcb_kernel: int = 9,
        csas_hidden_dim: int = 64,
        # Misc
        hash_method: Literal["fnv", "ravel"] = "fnv",
        pooling_method: Literal["mean", "random"] = "mean",
        norm_type: Literal["bn", "gn"] = "bn",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.grid_size = float(grid_size)
        self.pad = int(pad)

        self.backbone = MSDSpUNet34C(
            in_channels=self.in_channels,
            out_dim=self.out_dim,
            coarse_grid_size=self.grid_size,
            fine_grid_size=fine_grid_size,
            coarse_pad=self.pad,
            coarse_base_channels=base_channels,
            coarse_channels=list(channels),
            coarse_layers=list(layers),
            fine_channels=list(fine_channels),
            fine_layers=list(fine_layers),
            fine_out_dim=fine_out_dim,
            lkcb_kernel=lkcb_kernel,
            csas_hidden_dim=csas_hidden_dim,
            norm_type=norm_type,
            hash_method=hash_method,
            pooling_method=pooling_method,
        )

    def _build_feat(self, coord: torch.Tensor, color: Optional[torch.Tensor]) -> torch.Tensor:
        if color is None:
            color = torch.zeros_like(coord)
        else:
            color = color.float()

        if self.in_channels == 3:
            feat = color
        elif self.in_channels == 6:
            feat = torch.cat([coord, color], dim=1)
        else:
            feat = torch.cat([coord, color], dim=1)
            if feat.shape[1] < self.in_channels:
                pad = torch.zeros(
                    feat.shape[0],
                    self.in_channels - feat.shape[1],
                    dtype=feat.dtype,
                    device=feat.device,
                )
                feat = torch.cat([feat, pad], dim=1)
            elif feat.shape[1] > self.in_channels:
                feat = feat[:, : self.in_channels]
        return feat

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        text_guidance: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del text_guidance

        coord = batch["coord"].float()
        color = batch.get("color")
        feat = self._build_feat(coord, color)

        offset = batch.get("offset")
        if offset is None:
            offset = torch.tensor([0, coord.shape[0]], device=coord.device, dtype=torch.long)
        else:
            offset = offset.to(device=coord.device, dtype=torch.long)

        point_feats = self.backbone(coord=coord, feat=feat, offset=offset)

        if color is None:
            color_for_stats = torch.zeros_like(coord)
        else:
            color_for_stats = color.float()

        return {
            "point_feats": point_feats,
            "debug_input_coord_mean": coord.mean(),
            "debug_input_coord_std": coord.std(unbiased=False),
            "debug_input_color_mean": color_for_stats.mean(),
            "debug_input_color_std": color_for_stats.std(unbiased=False),
        }
