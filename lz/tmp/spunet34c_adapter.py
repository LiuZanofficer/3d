from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

from src.models.networks.spunet.spconv_unet_v1m1_base import SpUNetBase
from src.models.utils.structure import Point


class SpUNet34CAdapter(nn.Module):
    """Wrap SpUNet34C and expose point-level features for LZ decoder."""

    def __init__(
        self,
        in_channels: int = 3,
        out_dim: int = 128,
        grid_size: float = 0.02,
        pad: int = 128,
        base_channels: int = 32,
        channels: Sequence[int] = (32, 64, 128, 256, 256, 128, 96, 96),
        layers: Sequence[int] = (2, 3, 4, 6, 2, 2, 2, 2),
        hash_method: str = "fnv",
        pooling_method: str = "mean",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.grid_size = float(grid_size)
        self.pad = int(pad)

        self.backbone = SpUNetBase(
            in_channels=self.in_channels,
            out_channels=self.out_dim,
            base_channels=int(base_channels),
            channels=list(channels),
            layers=list(layers),
            out_fpn=False,
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

        point = Point(
            coord=coord,
            feat=feat,
            offset=offset,
            grid_size=self.grid_size,
        )
        point_out = self.backbone(point)

        voxel_feats = point_out.sparse_conv_feat.features
        v2p_map = point_out.v2p_map.long()
        point_feats = voxel_feats[v2p_map]

        if color is None:
            color = torch.zeros_like(coord)
        else:
            color = color.float()

        return {
            "point_feats": point_feats,
            "debug_input_coord_mean": coord.mean(),
            "debug_input_coord_std": coord.std(unbiased=False),
            "debug_input_color_mean": color.mean(),
            "debug_input_color_std": color.std(unbiased=False),
        }
