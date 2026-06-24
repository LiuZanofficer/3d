"""MSD-SpUNet34C: Multi-Scale Detail-Enhanced SpUNet34C.

Three components stacked on the vanilla SpUNetBase:

1. **LKCB** (Large-Kernel Context Block): a stack of ``depth`` 3x3x3
   sub-manifold sparse convolutions plus a 1x1 residual branch, inserted at
   the U-Net bottleneck. The installed spconv build does not support
   ``groups>1``, so a true depthwise 9x9x9 conv is impossible; instead we
   stack ``depth = (kernel - 1)/2`` full 3x3x3 SubMConv3d layers (so kernel=9
   maps to depth=4) which yields the same effective receptive field at lower
   parameter cost. The paper's ``9^3`` notation refers to the equivalent
   receptive field, not a single literal kernel.

2. **CSAS** (Cross-Scale Attention Skip): per-point skip gates on the U-Net
   skip connections. ``g_l = sigmoid(MLP([F_enc; up(F_dec)]))`` modulates the
   encoder skip features before concatenation with the upsampled decoder
   features (paper Eq. 2 / 3).

3. **Fine-Detail Branch**: an independent shallow sparse U-Net with
   ``voxel=0.01m`` that runs in parallel with the main 0.02m backbone. Each
   main point looks up its containing fine voxel via the fine branch's
   ``v2p_map``; the resulting per-point feature is concatenated with the
   coarse feature and fused through a 1x1 MLP (paper Eq. 4-6).

The class ``MSDSpUNet34C`` is intended to drop into LZModel through the
``MSDSpUNet34CAdapter`` wrapper (see lz_backbone/).
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import spconv.pytorch as spconv
import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from src.models.networks.spunet.spconv_unet_v1m1_base import (
    BasicBlock,
    SpUNetBase,
    _build_norm_fn,
)
from src.models.utils.misc import offset2batch
from src.models.utils.structure import Custom1x1Subm3d, Point


class LargeKernelContextBlock(spconv.SparseModule):
    """Stacked 3x3x3 sparse convs simulating a large receptive field with a
    1x1 residual branch.

    The original design called for a depthwise 9x9x9 sparse conv to capture
    long-range bottleneck context. The installed spconv build, however, does
    not implement ``groups>1`` for SubMConv3d (it raises
    ``AssertionError: don't support groups for now``), so a literal depthwise
    9x9x9 layer is impossible. We approximate the same effective receptive
    field by stacking ``depth = (kernel-1)/2`` full 3x3x3 sub-manifold
    convolutions: each layer extends each voxel's reach by one hop in the
    voxel graph, so depth=4 reaches voxels up to 4 hops away (effective RF
    9x9x9). The tail 1x1 projection plus identity residual preserve output
    dimensionality and the input topology (sub-manifold = no new voxels).
    """

    # kernel_size -> equivalent stack depth of 3x3x3 layers.
    _STACK_DEPTH_BY_KERNEL = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5}

    def __init__(
        self,
        channels: int,
        kernel_size: int = 9,
        norm_fn=None,
        indice_key: Optional[str] = None,
    ) -> None:
        super().__init__()
        assert norm_fn is not None
        assert kernel_size in self._STACK_DEPTH_BY_KERNEL, (
            f"LKCB kernel must be one of {sorted(self._STACK_DEPTH_BY_KERNEL.keys())}, got {kernel_size}"
        )
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        depth = self._STACK_DEPTH_BY_KERNEL[self.kernel_size]
        self.stack_depth = depth

        base_key = indice_key or f"lkcb{self.kernel_size}"

        # Stack of (3x3x3 SubMConv3d -> Norm -> ReLU) layers; each layer adds
        # 2 to the receptive field. We share an indice_key so the SpConv
        # backend reuses the indice_dict for all stacked kernels.
        layers: List[nn.Module] = []
        for i in range(depth):
            layers.append(
                spconv.SubMConv3d(
                    channels,
                    channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                    indice_key=f"{base_key}_s{i}",
                )
            )
            layers.append(_SparseFeatureWrapper(norm_fn(channels)))
            if i < depth - 1:
                layers.append(_SparseFeatureWrapper(nn.ReLU(inplace=True)))
        self.large_kernel = spconv.SparseSequential(*layers)

        # 1x1 residual projection branch.
        self.pwconv = Custom1x1Subm3d(channels, channels, kernel_size=1, bias=False)
        self.bn_pw = norm_fn(channels)

        self.act = nn.ReLU(inplace=True)

    def forward(self, x: spconv.SparseConvTensor) -> spconv.SparseConvTensor:
        residual = x

        # Large effective-RF branch via stacked 3x3x3 sub-manifold convs.
        large = self.large_kernel(x)

        # Pointwise (1x1) projection branch sharing the input indices.
        pw = self.pwconv(x)
        pw = pw.replace_feature(self.bn_pw(pw.features))

        # Residual fusion.
        out = large.replace_feature(
            large.features + pw.features + residual.features
        )
        out = out.replace_feature(self.act(out.features))
        return out


class _SparseFeatureWrapper(nn.Module):
    """Apply a dense module (BN/ReLU/etc.) to ``SparseConvTensor.features``."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, x):
        if isinstance(x, spconv.SparseConvTensor):
            return x.replace_feature(self.module(x.features))
        return self.module(x)


class CrossScaleAttentionGate(nn.Module):
    """Per-point gate for U-Net skip connections.

    Given encoder features ``F_enc`` and the upsampled decoder features
    ``up(F_dec)`` at the same level, produce a gate ``g`` per point and apply
    it to ``F_enc`` before the standard ``cat([up(F_dec), g * F_enc])`` fusion.
    """

    def __init__(self, enc_channels: int, dec_channels: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.enc_channels = int(enc_channels)
        self.dec_channels = int(dec_channels)
        self.hidden_dim = int(hidden_dim)

        self.gate = nn.Sequential(
            nn.Linear(self.enc_channels + self.dec_channels, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dim, self.enc_channels),
            nn.Sigmoid(),
        )

    def forward(
        self,
        enc_feat: torch.Tensor,
        dec_feat: torch.Tensor,
    ) -> torch.Tensor:
        g = self.gate(torch.cat([enc_feat, dec_feat], dim=1))
        return enc_feat * g


class MSDSpUNetBase(SpUNetBase):
    """SpUNetBase with LKCB inserted at the bottleneck and CSAS on skips."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        base_channels: int = 32,
        channels: List[int] = (32, 64, 128, 256, 256, 128, 96, 96),
        layers: List[int] = (2, 3, 4, 6, 2, 2, 2, 2),
        norm_type: Literal["bn", "gn"] = "bn",
        hash_method: Literal["fnv", "ravel"] = "fnv",
        pooling_method: Literal["mean", "random"] = "mean",
        lkcb_kernel: int = 9,
        csas_hidden_dim: int = 64,
        **kwargs,
    ) -> None:
        assert len(channels) % 2 == 0, (
            f"channels must have even length (encoder + decoder stages); got {len(channels)}"
        )
        assert len(channels) == len(layers), (
            f"channels and layers must have the same length; got {len(channels)} vs {len(layers)}"
        )
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            channels=list(channels),
            layers=list(layers),
            out_fpn=False,
            hash_method=hash_method,
            pooling_method=pooling_method,
            norm_type=norm_type,
            **kwargs,
        )
        self.lkcb_kernel = int(lkcb_kernel)
        self.csas_hidden_dim = int(csas_hidden_dim)
        norm_fn = _build_norm_fn(norm_type)

        # LKCB sits at the bottleneck (deepest encoder output, channels[num_stages-1]).
        bottleneck_channels = self.channels[self.num_stages - 1]
        self.lkcb = LargeKernelContextBlock(
            channels=bottleneck_channels,
            kernel_size=self.lkcb_kernel,
            norm_fn=norm_fn,
            indice_key=f"lkcb{self.num_stages}",
        )

        # CSAS gates: one per up-step (we have ``num_stages`` decoder stages).
        # At decoder stage ``s``, we fuse ``enc[s]`` (channel = enc_channels_at_stage[s])
        # with the upsampled decoder feature at the same resolution.
        # Encoder channels at each stage:
        # stage 0 -> base_channels (= conv_input output, used as the deepest skip)
        # stage s>0 -> channels[s-1]
        self.csas_gates = nn.ModuleList()
        for s in reversed(range(self.num_stages)):
            # Decoder up step at stage s: takes channels[len-s-2] -> dec_channels.
            # When s==0 the output of self.up[0] has channel channels[len-1]
            # (the last entry of `channels` after going through all stages).
            # We compute ``dec_channels`` mirror to SpUNetBase loop.
            dec_channels = self.channels[len(self.channels) - s - 1]
            if s == 0:
                enc_channels_at_skip = self.base_channels
            else:
                enc_channels_at_skip = self.channels[s - 1]
            self.csas_gates.append(
                CrossScaleAttentionGate(
                    enc_channels=enc_channels_at_skip,
                    dec_channels=dec_channels,
                    hidden_dim=self.csas_hidden_dim,
                )
            )
        # csas_gates[i] corresponds to up step i where i=0 is the deepest,
        # matching ``for s in reversed(range(num_stages))`` iteration order.

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):  # type: ignore[override]
        SpUNetBase._init_weights(m)
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, input_dict):  # type: ignore[override]
        if "grid_coord" in input_dict:
            grid_coord = input_dict["grid_coord"]
            feat = input_dict["feat"]
            offset = input_dict["offset"]
            batch = offset2batch(offset)
            sparse_shape = torch.add(torch.max(grid_coord, dim=0).values, 128).tolist()
            x = spconv.SparseConvTensor(
                features=feat,
                indices=torch.cat(
                    [batch.unsqueeze(-1).int(), grid_coord.int()], dim=1
                ).contiguous(),
                spatial_shape=sparse_shape,
                batch_size=batch[-1].tolist() + 1,
            )
            point = None
        else:
            point = Point(input_dict)
            point.sparsify(
                pad=128,
                hash_method=self.hash_method,
                pooling_method=self.pooling_method,
            )
            x = point.sparse_conv_feat

        x = self.conv_input(x)
        skips = [x]

        # Encoder forward.
        for s in range(self.num_stages):
            x = self.down[s](x)
            x = self.enc[s](x)
            if s == self.num_stages - 1:
                # Paper Eq. (1): LKCB at the bottleneck (deepest encoder output).
                x = self.lkcb(x)
            skips.append(x)

        # Decoder forward with CSAS gating on skips.
        x = skips.pop(-1)

        for gate_idx, s in enumerate(reversed(range(self.num_stages))):
            x = self.up[s](x)
            skip = skips.pop(-1)

            # Paper Eq. (2)/(3): per-point gate on encoder skip features
            # using upsampled decoder features as conditioning. spconv
            # SparseInverseConv3d preserves indices for the matching skip,
            # so x.features and skip.features are aligned per-voxel.
            gated_skip_feats = self.csas_gates[gate_idx](
                enc_feat=skip.features,
                dec_feat=x.features,
            )
            x = x.replace_feature(torch.cat((x.features, gated_skip_feats), dim=1))
            x = self.dec[s](x)

        x = self.final(x)

        if "grid_coord" in input_dict:
            return x
        else:
            point.sparse_conv_feat = x
            return point


class FineDetailBranch(nn.Module):
    """A shallow 0.01m sparse U-Net that produces a fine-resolution feature.

    The output is projected back to the main 0.02m point grid by nearest-voxel
    matching: each main point ``p_i`` looks up the fine voxel whose grid index
    floors to the same coarse voxel as ``p_i`` (so two main points sharing the
    same coarse voxel may receive different fine features as long as the fine
    voxels intersect the main neighbourhood).
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_dim: int = 64,
        coarse_grid_size: float = 0.02,
        fine_grid_size: float = 0.01,
        channels: Sequence[int] = (16, 32, 64, 64, 32, 16),
        layers: Sequence[int] = (1, 1, 1, 1, 1, 1),
        norm_type: Literal["bn", "gn"] = "bn",
        hash_method: Literal["fnv", "ravel"] = "fnv",
        pooling_method: Literal["mean", "random"] = "mean",
    ) -> None:
        super().__init__()
        assert fine_grid_size < coarse_grid_size, "Fine voxel must be smaller than coarse."
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.coarse_grid_size = float(coarse_grid_size)
        self.fine_grid_size = float(fine_grid_size)
        self.hash_method = hash_method
        self.pooling_method = pooling_method

        self.backbone = SpUNetBase(
            in_channels=self.in_channels,
            out_channels=self.out_dim,
            base_channels=channels[0],
            channels=list(channels),
            layers=list(layers),
            out_fpn=False,
            hash_method=hash_method,
            pooling_method=pooling_method,
            norm_type=norm_type,
        )

    def forward(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        offset: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the fine branch.

        Returns:
            fine_point_feats: ``[N, out_dim]`` tensor aligned with the input
                ``coord`` order. Each main point gets the feature of the fine
                voxel that contains it.
            fine_grid_coord: integer fine grid coordinates of each input point
                (shape ``[N, 3]``), useful for downstream debugging.
        """
        if coord.numel() == 0:
            return torch.zeros(0, self.out_dim, device=coord.device, dtype=feat.dtype), torch.zeros(
                0, 3, device=coord.device, dtype=torch.int32
            )

        point = Point(
            coord=coord.float(),
            feat=feat,
            offset=offset.long(),
            grid_size=self.fine_grid_size,
        )
        fine_point = self.backbone(point)

        # Sparse-conv produces voxel-level features and a v2p_map mapping each
        # original input point to its voxel index.
        voxel_feats = fine_point.sparse_conv_feat.features
        v2p_map = fine_point.v2p_map.long()
        fine_per_point = voxel_feats[v2p_map]

        # The fine grid coord is also stored on the Point.
        fine_grid_coord = fine_point.grid_coord
        return fine_per_point, fine_grid_coord


class MSDSpUNet34C(nn.Module):
    """Compose MSDSpUNetBase (LKCB + CSAS) with the parallel Fine-Detail Branch.

    Output is fused via a 1x1 MLP to ``out_dim`` channels and returned as a
    per-point feature tensor that downstream decoders can consume directly.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_dim: int = 128,
        coarse_grid_size: float = 0.02,
        fine_grid_size: float = 0.01,
        coarse_pad: int = 128,
        coarse_base_channels: int = 32,
        coarse_channels: Sequence[int] = (32, 64, 128, 256, 256, 128, 96, 96),
        coarse_layers: Sequence[int] = (2, 3, 4, 6, 2, 2, 2, 2),
        fine_channels: Sequence[int] = (16, 32, 64, 64, 32, 16),
        fine_layers: Sequence[int] = (1, 1, 1, 1, 1, 1),
        fine_out_dim: int = 64,
        lkcb_kernel: int = 9,
        csas_hidden_dim: int = 64,
        norm_type: Literal["bn", "gn"] = "bn",
        hash_method: Literal["fnv", "ravel"] = "fnv",
        pooling_method: Literal["mean", "random"] = "mean",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.coarse_grid_size = float(coarse_grid_size)
        self.fine_grid_size = float(fine_grid_size)
        self.coarse_pad = int(coarse_pad)

        self.coarse_backbone = MSDSpUNetBase(
            in_channels=in_channels,
            out_channels=out_dim,
            base_channels=coarse_base_channels,
            channels=list(coarse_channels),
            layers=list(coarse_layers),
            norm_type=norm_type,
            hash_method=hash_method,
            pooling_method=pooling_method,
            lkcb_kernel=lkcb_kernel,
            csas_hidden_dim=csas_hidden_dim,
        )

        self.fine_branch = FineDetailBranch(
            in_channels=in_channels,
            out_dim=fine_out_dim,
            coarse_grid_size=coarse_grid_size,
            fine_grid_size=fine_grid_size,
            channels=fine_channels,
            layers=fine_layers,
            norm_type=norm_type,
            hash_method=hash_method,
            pooling_method=pooling_method,
        )

        self.fuse = nn.Sequential(
            nn.Linear(out_dim + fine_out_dim, out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        coord: torch.Tensor,
        feat: torch.Tensor,
        offset: torch.Tensor,
    ) -> torch.Tensor:
        # Coarse path (main 0.02m SpUNet with LKCB + CSAS).
        # Construct a Point but do NOT call sparsify() here:
        # MSDSpUNetBase.forward dispatches on whether ``grid_coord`` is
        # already populated, and we want it to take the Point path so it
        # returns a Point (with v2p_map / sparse_conv_feat) rather than a
        # raw SparseConvTensor.
        coarse_point = Point(
            coord=coord.float(),
            feat=feat,
            offset=offset.long(),
            grid_size=self.coarse_grid_size,
        )
        coarse_out_point = self.coarse_backbone(coarse_point)
        coarse_voxel_feats = coarse_out_point.sparse_conv_feat.features
        coarse_v2p_map = coarse_out_point.v2p_map.long()
        coarse_point_feats = coarse_voxel_feats[coarse_v2p_map]

        # Fine path (parallel 0.01m shallow SpUNet) — Paper Eq. (4)/(5).
        fine_point_feats, _ = self.fine_branch(coord=coord, feat=feat, offset=offset)

        # Paper Eq. (6): fuse coarse + fine per-point features.
        # Both branches independently sparsify and use v2p_map to scatter
        # voxel features back to the original input point order, so
        # ``coarse_point_feats[i]`` and ``fine_point_feats[i]`` always refer
        # to the same physical input point — alignment is automatic as long
        # as the dataloader does not reorder points between the two branches.
        fused = self.fuse(torch.cat([coarse_point_feats, fine_point_feats], dim=1))
        return fused
