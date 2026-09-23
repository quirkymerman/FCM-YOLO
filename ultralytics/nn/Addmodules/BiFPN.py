# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch/Ultralytics port of the original EfficientDet BiFPN.

Paper:
    EfficientDet: Scalable and Efficient Object Detection (CVPR 2020)
    https://arxiv.org/abs/1911.09070

Official sources:
    https://github.com/google/automl/blob/master/efficientdet/tf2/efficientdet_keras.py
    https://github.com/google/automl/blob/master/efficientdet/tf2/fpn_configs.py
    inspected at commit 6a54c8741e7c3265d4547c4f35f47a0391122dc5

This file translates the official TensorFlow implementation to PyTorch while
preserving its dynamic BiFPN graph, resampling rules, fusion methods, operation
order, D0 defaults, and batch-normalization hyperparameters.  ``BiFPN`` accepts
YOLO feature lists such as ``[P3, P4, P5]`` and returns the fused pyramid as a
tuple.  Set ``num_outs=5`` to reproduce EfficientDet's P3-P7 pyramid; leave it
unset for YOLO11's native three-output P3-P5 neck.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = (
    "BiFPN",
    "FNode",
    "FPNCell",
    "FPNCells",
    "OpAfterCombine",
    "ResampleFeatureMap",
)


def _activation_fn(x: torch.Tensor, act_type: str) -> torch.Tensor:
    """Match EfficientDet's ``utils.activation_fn`` for supported activations."""
    if act_type in {"silu", "swish", "swish_native"}:
        return F.silu(x)
    if act_type == "relu":
        return F.relu(x)
    if act_type == "relu6":
        return F.relu6(x)
    if act_type == "hswish":
        return x * F.relu6(x + 3.0) / 6.0
    if act_type == "mish":
        return F.mish(x)
    raise ValueError(f"Unsupported act_type {act_type}")


def _bifpn_config(num_levels: int) -> list[dict[str, object]]:
    """Translate the official dynamic ``bifpn_config`` node construction."""
    if num_levels < 2:
        raise ValueError(f"BiFPN requires at least two pyramid levels, got {num_levels}")

    node_ids = {level: [level] for level in range(num_levels)}
    level_last_id = lambda level: node_ids[level][-1]
    level_all_ids = lambda level: list(node_ids[level])
    id_cnt = itertools.count(num_levels)
    nodes = []

    for level in range(num_levels - 2, -1, -1):
        # Top-down path.
        nodes.append({
            "feat_level": level,
            "inputs_offsets": [level_last_id(level), level_last_id(level + 1)],
        })
        node_ids[level].append(next(id_cnt))

    for level in range(1, num_levels):
        # Bottom-up path.
        nodes.append({
            "feat_level": level,
            "inputs_offsets": level_all_ids(level) + [level_last_id(level - 1)],
        })
        node_ids[level].append(next(id_cnt))

    return nodes


def _max_pool2d_same(x: torch.Tensor, kernel_size: tuple[int, int], stride: tuple[int, int]) -> torch.Tensor:
    """TensorFlow ``MaxPooling2D(..., padding='SAME')`` equivalent."""
    height, width = x.shape[-2:]
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    out_h = math.ceil(height / stride_h)
    out_w = math.ceil(width / stride_w)
    pad_h = max((out_h - 1) * stride_h + kernel_h - height, 0)
    pad_w = max((out_w - 1) * stride_w + kernel_w - width, 0)
    pad_top, pad_bottom = pad_h // 2, pad_h - pad_h // 2
    pad_left, pad_right = pad_w // 2, pad_w - pad_w // 2
    if pad_h or pad_w:
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), value=float("-inf"))
    return F.max_pool2d(x, kernel_size=kernel_size, stride=stride)


def _reset_conv(conv: nn.Conv2d) -> None:
    """Match Keras Conv2D/SeparableConv2D Glorot-uniform initialization."""
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        nn.init.zeros_(conv.bias)


class _Projection(nn.Module):
    """Official conditional 1x1 projection used during feature resampling."""

    def __init__(self, in_channels: int, out_channels: int, apply_bn: bool):
        super().__init__()
        if in_channels == out_channels:
            self.conv = None
            self.bn = None
        else:
            # Keras Conv2D keeps its bias enabled in the official resampler.
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
            _reset_conv(self.conv)
            # TF momentum=0.99 corresponds to PyTorch momentum=0.01.
            self.bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01) if apply_bn else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.conv is not None:
            x = self.conv(x)
            if self.bn is not None:
                x = self.bn(x)
        return x


class ResampleFeatureMap(nn.Module):
    """Official BiFPN feature-map channel and resolution resampling."""

    def __init__(
        self,
        in_channels: int,
        target_num_channels: int,
        apply_bn: bool = True,
        conv_after_downsample: bool = False,
    ):
        super().__init__()
        self.conv_after_downsample = conv_after_downsample
        self.projection = _Projection(in_channels, target_num_channels, apply_bn)

    @staticmethod
    def _pool2d(x: torch.Tensor, target_height: int, target_width: int) -> torch.Tensor:
        height, width = x.shape[-2:]
        height_stride = int((height - 1) // target_height + 1)
        width_stride = int((width - 1) // target_width + 1)
        return _max_pool2d_same(
            x,
            kernel_size=(height_stride + 1, width_stride + 1),
            stride=(height_stride, width_stride),
        )

    def forward(self, x: torch.Tensor, target_size: Sequence[int]) -> torch.Tensor:
        height, width = x.shape[-2:]
        target_height, target_width = int(target_size[0]), int(target_size[1])

        if height > target_height and width > target_width:
            if not self.conv_after_downsample:
                x = self.projection(x)
            x = self._pool2d(x, target_height, target_width)
            if self.conv_after_downsample:
                x = self.projection(x)
        elif height <= target_height and width <= target_width:
            x = self.projection(x)
            if height < target_height or width < target_width:
                x = F.interpolate(x.float(), size=(target_height, target_width), mode="nearest").to(x.dtype)
        else:
            raise ValueError(
                f"Incompatible resampling: feature shape {height}x{width}, "
                f"target shape {target_height}x{target_width}"
            )

        if x.shape[-2:] != (target_height, target_width):
            raise ValueError(
                f"Resampling produced {tuple(x.shape[-2:])}, expected {(target_height, target_width)}"
            )
        return x


class _SeparableConv2d(nn.Module):
    """Keras SeparableConv2D equivalent used by the original BiFPN."""

    def __init__(self, channels: int, use_bias: bool):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=use_bias)
        _reset_conv(self.depthwise)
        _reset_conv(self.pointwise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class OpAfterCombine(nn.Module):
    """Operation applied after each official BiFPN feature-fusion node."""

    def __init__(
        self,
        fpn_num_filters: int,
        conv_bn_act_pattern: bool = False,
        separable_conv: bool = True,
        act_type: str = "swish",
    ):
        super().__init__()
        self.conv_bn_act_pattern = conv_bn_act_pattern
        self.act_type = act_type
        use_bias = not conv_bn_act_pattern

        if separable_conv:
            self.conv = _SeparableConv2d(fpn_num_filters, use_bias=use_bias)
        else:
            self.conv = nn.Conv2d(fpn_num_filters, fpn_num_filters, kernel_size=3, padding=1, bias=use_bias)
            _reset_conv(self.conv)
        # TF momentum=0.99 corresponds to PyTorch momentum=0.01.
        self.bn = nn.BatchNorm2d(fpn_num_filters, eps=1e-3, momentum=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.conv_bn_act_pattern:
            x = _activation_fn(x, self.act_type)
        x = self.conv(x)
        x = self.bn(x)
        if self.conv_bn_act_pattern:
            x = _activation_fn(x, self.act_type)
        return x


class _FeatureFusion(nn.Module):
    """All fusion modes implemented by Google EfficientDet's ``FNode``."""

    _SUPPORTED = {"attn", "fastattn", "channel_attn", "channel_fastattn", "sum"}

    def __init__(self, num_inputs: int, num_channels: int, weight_method: str):
        super().__init__()
        if weight_method not in self._SUPPORTED:
            raise ValueError(f"Unknown weight_method {weight_method}")
        self.weight_method = weight_method

        if weight_method in {"channel_attn", "channel_fastattn"}:
            self.edge_weights = nn.Parameter(torch.ones(num_inputs, num_channels))
        elif weight_method in {"attn", "fastattn"}:
            self.edge_weights = nn.Parameter(torch.ones(num_inputs))
        else:
            self.register_parameter("edge_weights", None)

    @staticmethod
    def _add_n(nodes: Sequence[torch.Tensor]) -> torch.Tensor:
        new_node = nodes[0]
        for node in nodes[1:]:
            new_node = new_node + node
        return new_node

    def forward(self, nodes: Sequence[torch.Tensor]) -> torch.Tensor:
        if self.weight_method == "sum":
            return self._add_n(nodes)

        weights = self.edge_weights.to(dtype=nodes[0].dtype)
        if self.weight_method == "attn":
            weights = torch.softmax(weights, dim=0)
            stacked = torch.stack(tuple(nodes), dim=0)
            return torch.sum(stacked * weights[:, None, None, None, None], dim=0)

        if self.weight_method == "fastattn":
            weights = F.relu(weights)
            weights_sum = self._add_n(tuple(weights.unbind(0)))
            weighted = [node * weights[i] / (weights_sum + 0.0001) for i, node in enumerate(nodes)]
            return self._add_n(weighted)

        if self.weight_method == "channel_attn":
            weights = torch.softmax(weights, dim=0)
            stacked = torch.stack(tuple(nodes), dim=0)
            return torch.sum(stacked * weights[:, None, :, None, None], dim=0)

        weights = F.relu(weights)
        weights_sum = self._add_n(tuple(weights.unbind(0)))
        weighted = [
            node * weights[i][None, :, None, None] / (weights_sum[None, :, None, None] + 0.0001)
            for i, node in enumerate(nodes)
        ]
        return self._add_n(weighted)


class FNode(nn.Module):
    """A PyTorch layer implementing one node of the original BiFPN graph."""

    def __init__(
        self,
        feat_level: int,
        inputs_offsets: Sequence[int],
        input_channels: Sequence[int],
        fpn_num_filters: int,
        apply_bn_for_resampling: bool = True,
        conv_after_downsample: bool = False,
        conv_bn_act_pattern: bool = False,
        separable_conv: bool = True,
        act_type: str = "swish",
        weight_method: str = "fastattn",
    ):
        super().__init__()
        self.feat_level = feat_level
        self.inputs_offsets = tuple(int(offset) for offset in inputs_offsets)
        self.resample_layers = nn.ModuleList([
            ResampleFeatureMap(
                input_channels[offset],
                fpn_num_filters,
                apply_bn=apply_bn_for_resampling,
                conv_after_downsample=conv_after_downsample,
            )
            for offset in self.inputs_offsets
        ])
        self.fuse_features = _FeatureFusion(len(self.inputs_offsets), fpn_num_filters, weight_method)
        self.op_after_combine = OpAfterCombine(
            fpn_num_filters,
            conv_bn_act_pattern=conv_bn_act_pattern,
            separable_conv=separable_conv,
            act_type=act_type,
        )

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        target_size = feats[self.feat_level].shape[-2:]
        nodes = [
            resample(feats[input_offset], target_size)
            for resample, input_offset in zip(self.resample_layers, self.inputs_offsets)
        ]
        return self.op_after_combine(self.fuse_features(nodes))


class FPNCell(nn.Module):
    """A single dynamic BiFPN cell, matching Google's ``FPNCell``."""

    def __init__(
        self,
        in_channels: Sequence[int],
        fpn_num_filters: int,
        apply_bn_for_resampling: bool = True,
        conv_after_downsample: bool = False,
        conv_bn_act_pattern: bool = False,
        separable_conv: bool = True,
        act_type: str = "swish",
        weight_method: str = "fastattn",
    ):
        super().__init__()
        self.num_levels = len(in_channels)
        self.fpn_config = _bifpn_config(self.num_levels)
        node_channels = list(in_channels)
        fnodes = []

        for node_config in self.fpn_config:
            fnodes.append(FNode(
                feat_level=int(node_config["feat_level"]),
                inputs_offsets=node_config["inputs_offsets"],
                input_channels=node_channels,
                fpn_num_filters=fpn_num_filters,
                apply_bn_for_resampling=apply_bn_for_resampling,
                conv_after_downsample=conv_after_downsample,
                conv_bn_act_pattern=conv_bn_act_pattern,
                separable_conv=separable_conv,
                act_type=act_type,
                weight_method=weight_method,
            ))
            node_channels.append(fpn_num_filters)

        self.fnodes = nn.ModuleList(fnodes)
        self.output_offsets = []
        for level in range(self.num_levels):
            for reverse_index, node_config in enumerate(reversed(self.fpn_config)):
                if node_config["feat_level"] == level:
                    self.output_offsets.append(self.num_levels + len(self.fpn_config) - 1 - reverse_index)
                    break

    def forward(self, feats: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        cell_feats = list(feats)
        for fnode in self.fnodes:
            cell_feats.append(fnode(cell_feats))
        return tuple(cell_feats[offset] for offset in self.output_offsets)


class FPNCells(nn.Module):
    """Repeated BiFPN cells, matching EfficientDet's ``FPNCells``."""

    def __init__(
        self,
        in_channels: Sequence[int],
        fpn_num_filters: int = 64,
        fpn_cell_repeats: int = 3,
        apply_bn_for_resampling: bool = True,
        conv_after_downsample: bool = False,
        conv_bn_act_pattern: bool = False,
        separable_conv: bool = True,
        act_type: str = "swish",
        weight_method: str = "fastattn",
    ):
        super().__init__()
        if fpn_cell_repeats < 1:
            raise ValueError(f"fpn_cell_repeats must be positive, got {fpn_cell_repeats}")

        common_args = dict(
            fpn_num_filters=fpn_num_filters,
            apply_bn_for_resampling=apply_bn_for_resampling,
            conv_after_downsample=conv_after_downsample,
            conv_bn_act_pattern=conv_bn_act_pattern,
            separable_conv=separable_conv,
            act_type=act_type,
            weight_method=weight_method,
        )
        cells = [FPNCell(in_channels, **common_args)]
        cells.extend(
            FPNCell([fpn_num_filters] * len(in_channels), **common_args)
            for _ in range(1, fpn_cell_repeats)
        )
        self.cells = nn.ModuleList(cells)

    def forward(self, feats: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        feats = tuple(feats)
        for cell in self.cells:
            feats = cell(feats)
        return feats


class BiFPN(nn.Module):
    """Complete EfficientDet BiFPN adapted to a YOLO11 multi-scale input list.

    Args:
        in_channels: Channels of the input pyramid ordered from high to low
            spatial resolution, e.g. YOLO11 ``[P3, P4, P5]``.
        out_channels: Shared BiFPN width. EfficientDet-D0 uses 64.
        num_repeats: Number of repeated BiFPN cells. EfficientDet-D0 uses 3.
        num_outs: Number of output levels. ``None`` keeps the input count for
            YOLO11; use 5 with P3-P5 inputs to build EfficientDet-style P3-P7.
        weight_method: ``fastattn`` is the original D0-D5 default. Other
            official modes are ``attn``, ``channel_attn``,
            ``channel_fastattn``, and ``sum``.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (256, 512, 1024),
        out_channels: int = 64,
        num_repeats: int = 3,
        num_outs: int | None = None,
        weight_method: str = "fastattn",
        apply_bn_for_resampling: bool = True,
        conv_after_downsample: bool = False,
        conv_bn_act_pattern: bool = False,
        separable_conv: bool = True,
        act_type: str = "swish",
    ):
        super().__init__()
        self.in_channels = tuple(int(channel) for channel in in_channels)
        if len(self.in_channels) < 2:
            raise ValueError(f"BiFPN requires at least two input levels, got {len(self.in_channels)}")

        self.num_outs = len(self.in_channels) if num_outs is None else int(num_outs)
        if self.num_outs < len(self.in_channels):
            raise ValueError(
                f"num_outs ({self.num_outs}) cannot be smaller than the input count ({len(self.in_channels)})"
            )

        extra_levels = self.num_outs - len(self.in_channels)
        extra_resamplers = []
        source_channels = self.in_channels[-1]
        for _ in range(extra_levels):
            extra_resamplers.append(ResampleFeatureMap(
                source_channels,
                out_channels,
                apply_bn=apply_bn_for_resampling,
                conv_after_downsample=conv_after_downsample,
            ))
            source_channels = out_channels
        self.extra_resamplers = nn.ModuleList(extra_resamplers)

        pyramid_channels = list(self.in_channels) + [out_channels] * extra_levels
        self.fpn_cells = FPNCells(
            pyramid_channels,
            fpn_num_filters=out_channels,
            fpn_cell_repeats=num_repeats,
            apply_bn_for_resampling=apply_bn_for_resampling,
            conv_after_downsample=conv_after_downsample,
            conv_bn_act_pattern=conv_bn_act_pattern,
            separable_conv=separable_conv,
            act_type=act_type,
            weight_method=weight_method,
        )
        self.out_channels = tuple([out_channels] * self.num_outs)

    def forward(self, inputs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
        if not isinstance(inputs, (list, tuple)):
            raise TypeError(f"BiFPN expects a list or tuple of feature maps, got {type(inputs).__name__}")
        if len(inputs) != len(self.in_channels):
            raise ValueError(f"BiFPN expects {len(self.in_channels)} inputs, got {len(inputs)}")

        feats = list(inputs)
        for resample in self.extra_resamplers:
            height, width = feats[-1].shape[-2:]
            target_size = ((height - 1) // 2 + 1, (width - 1) // 2 + 1)
            feats.append(resample(feats[-1], target_size))

        return self.fpn_cells(feats)
