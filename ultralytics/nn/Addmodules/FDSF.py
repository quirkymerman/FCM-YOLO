import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DCTFreqGate", "SpectralBandGateBranch", "B_FDSF", "N_FDSF", "OSF_FDSF", "OSF_QKV"]

# Final paper terminology (legacy class names are retained for YAML/checkpoint compatibility):
#   B_FDSF   -> SDFE: Spectral-Directional Feature Extraction
#   N_FDSF   -> WDSF: Wavelet-Directional Selective Fusion
#   OSF_FDSF -> SCCF: Semantic-Consistent Cross-scale Fusion



def _make_divisible_channels(channels, groups):
    return math.ceil(channels / groups) * groups


def _dct_basis(pos, freq, size):
    value = math.cos(math.pi * freq * (pos + 0.5) / size) / math.sqrt(size)
    return value if freq == 0 else value * math.sqrt(2.0)


class DCTFreqGate(nn.Module):
    """DCT multi-frequency channel gate adapted from FSFFM, without FFT branches."""

    def __init__(self, channels, dct_size=7, num_freq=16, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.dct_size = dct_size
        self.num_freq = min(num_freq, dct_size * dct_size)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.register_buffer("weight", self._build_dct_weight(channels, dct_size, self.num_freq), persistent=False)

    @staticmethod
    def _build_dct_weight(channels, dct_size, num_freq):
        freq_pairs = [
            (0, 0), (0, 1), (1, 0), (1, 1),
            (0, 2), (2, 0), (1, 2), (2, 1),
            (2, 2), (0, 3), (3, 0), (1, 3),
            (3, 1), (2, 3), (3, 2), (3, 3),
        ]
        if num_freq > len(freq_pairs):
            freq_pairs.extend(
                (u, v)
                for u in range(dct_size)
                for v in range(dct_size)
                if (u, v) not in freq_pairs
            )
        freq_pairs = freq_pairs[:num_freq]

        weight = torch.zeros(channels, dct_size, dct_size)
        channels_per_freq = _make_divisible_channels(channels, len(freq_pairs)) // len(freq_pairs)
        for c in range(channels):
            u, v = freq_pairs[min(c // channels_per_freq, len(freq_pairs) - 1)]
            for i in range(dct_size):
                for j in range(dct_size):
                    weight[c, i, j] = _dct_basis(i, u, dct_size) * _dct_basis(j, v, dct_size)
        return weight.unsqueeze(0)

    def forward(self, x):
        x_pool = F.adaptive_avg_pool2d(x, (self.dct_size, self.dct_size))
        y = (x_pool * self.weight.to(dtype=x.dtype, device=x.device)).sum(dim=(2, 3))
        y = self.conv(y.unsqueeze(1)).squeeze(1)
        return torch.sigmoid(y).view(x.shape[0], self.channels, 1, 1)


# SBG-v1 kept for reference: fixed hard radial masks.
class SpectralBandGateBranch(nn.Module):
    """FFT frequency branch with learnable radial spectral-band gating."""

    def __init__(self, channels, band_split_low=0.33, band_split_high=0.66, reduction=4, fft_norm="ortho"):
        super().__init__()
        edges = [0.0, float(band_split_low), float(band_split_high), 1.01]
        self.band_edges = []
        for edge in sorted(max(0.0, edge) for edge in edges):
            edge = min(edge, 1.01)
            if not self.band_edges or edge - self.band_edges[-1] > 1e-4:
                self.band_edges.append(edge)
        if self.band_edges[0] > 0.0:
            self.band_edges.insert(0, 0.0)
        if self.band_edges[-1] < 1.0:
            self.band_edges[-1] = 1.01

        self.num_bands = len(self.band_edges) - 1
        hidden = max(self.num_bands // reduction, 1)
        self.band_gate = nn.Sequential(
            nn.Linear(self.num_bands, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.num_bands, bias=True),
        )
        nn.init.zeros_(self.band_gate[-1].weight)
        nn.init.zeros_(self.band_gate[-1].bias)

        self.freq_conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.freq_bn = nn.BatchNorm2d(channels)
        self.freq_act = nn.SiLU(inplace=True)
        self.fft_norm = fft_norm
        self._band_mask_cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_band_mask_cache"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_band_mask_cache" not in self.__dict__:
            self._band_mask_cache = {}

    def _get_band_masks(self, h, w, device, dtype):
        key = (h, w, device, dtype)
        cached = self._band_mask_cache.get(key)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached

        y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(xx * xx + yy * yy).clamp(0.0, 1.0)

        masks = []
        for idx in range(self.num_bands):
            low = self.band_edges[idx]
            high = self.band_edges[idx + 1]
            if idx == self.num_bands - 1:
                mask = ((radius >= low) & (radius <= high)).to(dtype)
            else:
                mask = ((radius >= low) & (radius < high)).to(dtype)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        self._band_mask_cache[key] = masks
        return masks

    def _band_gate_fp32(self, x):
        fc1 = self.band_gate[0]
        fc2 = self.band_gate[2]
        x = F.linear(x, fc1.weight.float(), None if fc1.bias is None else fc1.bias.float())
        x = F.relu(x, inplace=False)
        return F.linear(x, fc2.weight.float(), None if fc2.bias is None else fc2.bias.float())

    def _conv_bn_act_fp32(self, x):
        x = F.conv2d(x, self.freq_conv.weight.float(), None, stride=1, padding=0)
        x = F.batch_norm(
            x,
            self.freq_bn.running_mean.float(),
            self.freq_bn.running_var.float(),
            self.freq_bn.weight.float() if self.freq_bn.weight is not None else None,
            self.freq_bn.bias.float() if self.freq_bn.bias is not None else None,
            self.freq_bn.training,
            self.freq_bn.momentum,
            self.freq_bn.eps,
        )
        return F.silu(x, inplace=False)

    def _complex_conv(self, x):
        real = self._conv_bn_act_fp32(x.real)
        imag = self._conv_bn_act_fp32(x.imag)
        return torch.complex(real, imag)

    def forward(self, x):
        orig_dtype = x.dtype
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x_fp32 = x.float()
            ffted = torch.fft.rfftn(x_fp32, dim=(-2, -1), norm=self.fft_norm)
            masks = self._get_band_masks(ffted.shape[-2], ffted.shape[-1], ffted.device, ffted.real.dtype)

            amp = torch.abs(ffted).mean(dim=1)
            mask_sums = masks.flatten(1).sum(dim=1).clamp_min(1e-6).to(amp.dtype)
            band_stats = torch.einsum("bhw,khw->bk", amp, masks) / mask_sums.unsqueeze(0)
            gate = 1.0 + torch.tanh(self._band_gate_fp32(band_stats))
            gating_map = torch.einsum("bk,khw->bhw", gate, masks).unsqueeze(1)

            ffted = self._complex_conv(ffted * gating_map)
            out = torch.fft.irfftn(ffted, s=x_fp32.shape[-2:], dim=(-2, -1), norm=self.fft_norm)
        return out.to(orig_dtype)


# class SpectralBandGateBranch(nn.Module):
#     """SBG-v3: FFT branch with fixed band centers and learnable soft bandwidths."""
#
#     def __init__(
#         self,
#         channels,
#         band_split_low=0.33,
#         band_split_high=0.66,
#         reduction=4,
#         fft_norm="ortho",
#         min_bandwidth=0.04,
#         max_bandwidth=0.25,
#     ):
#         super().__init__()
#         edges = self._sanitize_edges(band_split_low, band_split_high)
#         self.num_bands = len(edges) - 1
#         self.min_bandwidth = float(min_bandwidth)
#         self.max_bandwidth = float(max(max_bandwidth, min_bandwidth + 1e-4))
#
#         centers = [(edges[i] + edges[i + 1]) * 0.5 for i in range(self.num_bands)]
#         widths = [max((edges[i + 1] - edges[i]) * 0.5, self.min_bandwidth + 1e-4) for i in range(self.num_bands)]
#         self.register_buffer("band_centers", torch.tensor(centers, dtype=torch.float32), persistent=False)
#         self.band_width_raw = nn.Parameter(torch.tensor([self._width_to_raw(v) for v in widths], dtype=torch.float32))
#
#         hidden = max(self.num_bands // reduction, 1)
#         self.band_gate = nn.Sequential(
#             nn.Linear(self.num_bands, hidden, bias=True),
#             nn.ReLU(inplace=True),
#             nn.Linear(hidden, self.num_bands, bias=True),
#         )
#         nn.init.zeros_(self.band_gate[-1].weight)
#         nn.init.zeros_(self.band_gate[-1].bias)
#
#         self.freq_conv = nn.Conv2d(channels, channels, 1, bias=False)
#         self.freq_bn = nn.BatchNorm2d(channels)
#         self.freq_act = nn.SiLU(inplace=True)
#         self.fft_norm = fft_norm
#         self._radius_cache = {}
#
#     @staticmethod
#     def _sanitize_edges(band_split_low, band_split_high):
#         edges = [0.0, float(band_split_low), float(band_split_high), 1.0]
#         edges = sorted(min(max(edge, 0.0), 1.0) for edge in edges)
#         clean = []
#         for edge in edges:
#             if not clean or edge - clean[-1] > 1e-4:
#                 clean.append(edge)
#         if clean[0] > 0.0:
#             clean.insert(0, 0.0)
#         if clean[-1] < 1.0:
#             clean.append(1.0)
#         if len(clean) < 2:
#             clean = [0.0, 1.0]
#         return clean
#
#     @staticmethod
#     def _logit(value, eps=1e-4):
#         value = min(max(float(value), eps), 1.0 - eps)
#         return math.log(value / (1.0 - value))
#
#     def _width_to_raw(self, width):
#         value = (float(width) - self.min_bandwidth) / (self.max_bandwidth - self.min_bandwidth)
#         return self._logit(value)
#
#     def __getstate__(self):
#         state = self.__dict__.copy()
#         state["_radius_cache"] = {}
#         return state
#
#     def __setstate__(self, state):
#         self.__dict__.update(state)
#         if "_radius_cache" not in self.__dict__:
#             self._radius_cache = {}
#
#     def _get_radius_grid(self, h, w, device, dtype):
#         key = (h, w, device, dtype)
#         cached = self._radius_cache.get(key)
#         if cached is not None and cached.device == device and cached.dtype == dtype:
#             return cached
#
#         y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
#         x = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
#         yy, xx = torch.meshgrid(y, x, indexing="ij")
#         radius = torch.sqrt(xx * xx + yy * yy).clamp(0.0, 1.0)
#         self._radius_cache[key] = radius
#         return radius
#
#     def _get_band_masks(self, radius):
#         dtype = radius.dtype
#         centers = self.band_centers.to(dtype=dtype, device=radius.device)
#         widths = self.min_bandwidth + (self.max_bandwidth - self.min_bandwidth) * torch.sigmoid(self.band_width_raw)
#         widths = widths.to(dtype=dtype, device=radius.device).clamp_min(1e-4)
#
#         distance = (radius.unsqueeze(0) - centers.view(-1, 1, 1)) / widths.view(-1, 1, 1)
#         masks = torch.exp(-0.5 * distance.square())
#         return masks / masks.sum(dim=0, keepdim=True).clamp_min(1e-6)
#
#     def _band_gate_fp32(self, x):
#         fc1 = self.band_gate[0]
#         fc2 = self.band_gate[2]
#         x = F.linear(x, fc1.weight.float(), None if fc1.bias is None else fc1.bias.float())
#         x = F.relu(x, inplace=False)
#         return F.linear(x, fc2.weight.float(), None if fc2.bias is None else fc2.bias.float())
#
#     def _conv_bn_act_fp32(self, x):
#         x = F.conv2d(x, self.freq_conv.weight.float(), None, stride=1, padding=0)
#         x = F.batch_norm(
#             x,
#             self.freq_bn.running_mean.float(),
#             self.freq_bn.running_var.float(),
#             self.freq_bn.weight.float() if self.freq_bn.weight is not None else None,
#             self.freq_bn.bias.float() if self.freq_bn.bias is not None else None,
#             self.freq_bn.training,
#             self.freq_bn.momentum,
#             self.freq_bn.eps,
#         )
#         return F.silu(x, inplace=False)
#
#     def _complex_conv(self, x):
#         real = self._conv_bn_act_fp32(x.real)
#         imag = self._conv_bn_act_fp32(x.imag)
#         return torch.complex(real, imag)
#
#     def forward(self, x):
#         orig_dtype = x.dtype
#         with torch.amp.autocast(device_type=x.device.type, enabled=False):
#             x_fp32 = x.float()
#             ffted = torch.fft.rfftn(x_fp32, dim=(-2, -1), norm=self.fft_norm)
#             radius = self._get_radius_grid(ffted.shape[-2], ffted.shape[-1], ffted.device, ffted.real.dtype)
#             masks = self._get_band_masks(radius)
#
#             amp = torch.abs(ffted).mean(dim=1)
#             mask_sums = masks.flatten(1).sum(dim=1).clamp_min(1e-6).to(amp.dtype)
#             band_stats = torch.einsum("bhw,khw->bk", amp, masks) / mask_sums.unsqueeze(0)
#             gate = 1.0 + torch.tanh(self._band_gate_fp32(band_stats))
#             gating_map = torch.einsum("bk,khw->bhw", gate, masks).unsqueeze(1)
#
#             ffted = self._complex_conv(ffted * gating_map)
#             out = torch.fft.irfftn(ffted, s=x_fp32.shape[-2:], dim=(-2, -1), norm=self.fft_norm)
#         return out.to(orig_dtype)

# class BranchSoftmaxGate(nn.Module):
#     """Channel-level branch selection with softmax across branches."""
#
#     def __init__(self, channels, branches, reduction=4):
#         super().__init__()
#         hidden = max(channels // reduction, 16)
#         self.branches = branches
#         self.channels = channels
#         self.mlp = nn.Sequential(
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(channels, hidden, 1, bias=True),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(hidden, branches * channels, 1, bias=True),
#         )
#
#     def forward(self, x):
#         alpha = self.mlp(x).view(x.shape[0], self.branches, self.channels, 1, 1)
#         return torch.softmax(alpha, dim=1)


class SpatialGate(nn.Module):
    """Foreground-aware spatial gate for fused neck features."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        maxv = torch.amax(x, dim=1, keepdim=True)
        return torch.sigmoid(self.conv(torch.cat((avg, maxv), dim=1)))

class GaussianBlurDWConv(nn.Module):
    """Fixed depthwise Gaussian smoothing for weak high-frequency responses."""

    def __init__(self, channels):
        super().__init__()
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=torch.float32,
        )
        kernel = kernel / kernel.sum()
        self.register_buffer("weight", kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1), persistent=False)
        self.channels = channels

    def forward(self, x):
        return F.conv2d(x, self.weight.to(dtype=x.dtype, device=x.device), padding=1, groups=self.channels)

class HaarDWT2D(nn.Module):
    """Fixed one-level Haar DWT using grouped convolution."""

    def __init__(self, channels):
        super().__init__()
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32) * 0.5
        weight = torch.stack((ll, lh, hl, hh), dim=0).view(4, 1, 2, 2).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight, persistent=False)
        self.channels = channels

    def forward(self, x):
        b, c, h, w = x.shape
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        y = F.conv2d(x, self.weight.to(dtype=x.dtype, device=x.device), stride=2, groups=self.channels)
        y = y.view(b, c, 4, y.shape[-2], y.shape[-1])
        return y[:, :, 0], y[:, :, 1], y[:, :, 2], y[:, :, 3]

class HaarIDWT2D(nn.Module):
    """Fixed one-level inverse Haar DWT using grouped transposed convolution."""

    def __init__(self, channels):
        super().__init__()
        ll = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[-1.0, 1.0], [-1.0, 1.0]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32) * 0.5
        weight = torch.stack((ll, lh, hl, hh), dim=0).view(4, 1, 2, 2).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight, persistent=False)
        self.channels = channels

    def forward(self, ll, lh, hl, hh, output_size=None):
        b, c, h, w = ll.shape
        y = torch.stack((ll, lh, hl, hh), dim=2).view(b, 4 * c, h, w)
        out = F.conv_transpose2d(y, self.weight.to(dtype=y.dtype, device=y.device), stride=2, groups=self.channels)
        if output_size is not None:
            out = out[..., : output_size[0], : output_size[1]]
        return out


class LearnableGaussianFilterBank(nn.Module):
    """Depthwise Gaussian filter bank with learnable sigma, adapted from NS-FPN LFP."""

    def __init__(self, channels, kernel_size=3, num_filters=1, sigma_init=1.0, sigma_min=0.05):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Gaussian kernel_size must be odd")
        self.channels = channels
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.padding = kernel_size // 2
        self.sigma_min = float(sigma_min)
        raw_init = math.log(math.exp(max(float(sigma_init) - self.sigma_min, 1e-4)) - 1.0)
        self.sigma_raw = nn.Parameter(torch.full((num_filters,), raw_init, dtype=torch.float32))

    def _gaussian_kernels(self, dtype, device):
        coords = torch.arange(self.kernel_size, dtype=dtype, device=device) - self.kernel_size // 2
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        dist2 = xx.square() + yy.square()
        sigmas = F.softplus(self.sigma_raw).to(dtype=dtype, device=device) + self.sigma_min
        kernels = []
        for sigma in sigmas:
            kernel = torch.exp(-dist2 / (2.0 * sigma.square().clamp_min(1e-6)))
            kernels.append(kernel / kernel.sum().clamp_min(1e-6))
        return torch.stack(kernels, dim=0).view(self.num_filters, 1, self.kernel_size, self.kernel_size)

    def forward(self, x):
        kernels = self._gaussian_kernels(x.dtype, x.device)
        weight = kernels.repeat(self.channels, 1, 1, 1)
        x = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode="replicate")
        return F.conv2d(x, weight, groups=self.channels)

# Previous B_FDSF-v1 kept for reference.
# class B_FDSF(nn.Module):
#     """Backbone frequency-directional selective enhancement."""
#
#     def __init__(self, c1, c2, k=7, dct_size=7):
#         super().__init__()
#         self.proj = nn.Identity() if c1 == c2 else nn.Sequential(
#             nn.Conv2d(c1, c2, 1, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(inplace=True),
#         )
#         self.local = nn.Sequential(
#             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(inplace=True),
#         )
#         self.high_freq_pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
#         self.dir_h = nn.Conv2d(c2, c2, (1, k), padding=(0, k // 2), groups=c2, bias=False)
#         self.dir_v = nn.Conv2d(c2, c2, (k, 1), padding=(k // 2, 0), groups=c2, bias=False)
#         self.dir_bn = nn.BatchNorm2d(c2)
#         self.dir_act = nn.SiLU(inplace=True)
#         self.branch_gate = BranchSoftmaxGate(c2, branches=3)
#         self.freq_gate = DCTFreqGate(c2, dct_size=dct_size)
#
#     def forward(self, x):
#         x = self.proj(x)
#         f_l = self.local(x)
#         f_f = x - self.high_freq_pool(x)
#         f_d = self.dir_act(self.dir_bn(self.dir_h(x) + self.dir_v(x)))
#         alpha = self.branch_gate(x)
#         fused = alpha[:, 0] * f_l + alpha[:, 1] * f_f + alpha[:, 2] * f_d
#         return x + self.freq_gate(x) * fused


# class B_FDSF(nn.Module):
#     """B_FDSF-v2: split-enhance-fuse backbone frequency-directional block."""
#
#     def __init__(self, c1, c2, k=7, dct_size=7, gamma_init=0.1):
#         super().__init__()
#         self.expand = nn.Sequential(
#             nn.Conv2d(c1, 2 * c2, 1, bias=False),
#             nn.BatchNorm2d(2 * c2),
#             nn.SiLU(inplace=True),
#         )
#         self.local = nn.Sequential(
#             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(inplace=True),
#         )
#         self.high_freq_pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
#         self.dir_h = nn.Sequential(
#             nn.Conv2d(c2, c2, (1, k), padding=(0, k // 2), groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(inplace=True),
#         )
#         self.dir_v = nn.Sequential(
#             nn.Conv2d(c2, c2, (k, 1), padding=(k // 2, 0), groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(inplace=True),
#         )
#         self.branch_gate = BranchSoftmaxGate(c2, branches=4)
#         self.freq_gate = DCTFreqGate(c2, dct_size=dct_size)
#         self.fuse = nn.Sequential(
#             nn.Conv2d(2 * c2, c2, 1, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(inplace=True),
#         )
#         self.shortcut = nn.Identity() if c1 == c2 else nn.Sequential(
#             nn.Conv2d(c1, c2, 1, bias=False),
#             nn.BatchNorm2d(c2),
#         )
#         self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
#
#     def forward(self, x):
#         z_keep, z_enhance = self.expand(x).chunk(2, dim=1)
#         f_l = self.local(z_enhance)
#         f_h = self.dir_h(z_enhance)
#         f_v = self.dir_v(z_enhance)
#         f_f = z_enhance - self.high_freq_pool(z_enhance)
#         alpha = self.branch_gate(z_enhance)
#         enhanced = z_enhance + self.freq_gate(z_enhance) * (
#             alpha[:, 0] * f_l + alpha[:, 1] * f_h + alpha[:, 2] * f_v + alpha[:, 3] * f_f
#         )
#         fused = self.fuse(torch.cat((z_keep, enhanced), dim=1))
#         return self.shortcut(x) + self.gamma * fused

class B_FDSF(nn.Module):
    """FSFE（Frequency–Spatial Feature Extraction） (legacy symbol: B_FDSF)."""

    def __init__(self, c1, c2, k=5, dct_size=7):
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(c1, 2 * c2, 1, bias=False),
            nn.BatchNorm2d(2 * c2),
            nn.SiLU(inplace=True),
        )
        self.local = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.dir_h = nn.Sequential(
            nn.Conv2d(c2, c2, (1, k), padding=(0, k // 2), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.dir_v = nn.Sequential(
            nn.Conv2d(c2, c2, (k, 1), padding=(k // 2, 0), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.freq_branch = SpectralBandGateBranch(c2)
        self.freq_gate = DCTFreqGate(4 * c2, dct_size=dct_size)
        self.branch_fuse = nn.Sequential(
            nn.Conv2d(4 * c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.shortcut = nn.Identity() if c1 == c2 else nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
        )
        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x):
        z_keep, z_enhance = self.expand(x).chunk(2, dim=1)
        f_l = self.local(z_enhance)
        f_h = self.dir_h(z_enhance)
        f_v = self.dir_v(z_enhance)
        f_f = self.freq_branch(z_enhance)
        z_c = torch.cat((f_l, f_h, f_v, f_f), dim=1)

        delta = self.branch_fuse(self.freq_gate(z_c) * z_c)
        enhanced = z_enhance + delta
        fused = self.fuse(torch.cat((z_keep, enhanced), dim=1))
        return self.shortcut(x) + self.gamma * fused



# Previous N_FDSF-v1/v2 kept for reference.
# # Previous N_FDSF-v4 weakened LFP kept for reference before NS-FPN correction.
# # Previous N_FDSF-v5 kept for reference.
# class N_FDSF(nn.Module):
# #     """Neck frequency-directional selective calibration."""
# #
# #     def __init__(self, c1, c2, k=7, dct_size=7):
# #         super().__init__()
# #         self.proj = nn.Identity() if c1 == c2 else nn.Sequential(
# #             nn.Conv2d(c1, c2, 1, bias=False),
# #             nn.BatchNorm2d(c2),
# #             nn.SiLU(inplace=True),
# #         )
# #         self.local = nn.Sequential(
# #             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
# #             nn.BatchNorm2d(c2),
# #             nn.SiLU(inplace=True),
# #         )
# #         self.high_freq_pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
# #         self.dir_h = nn.Sequential(
# #             nn.Conv2d(c2, c2, (1, k), padding=(0, k // 2), groups=c2, bias=False),
# #             nn.BatchNorm2d(c2),
# #             nn.SiLU(inplace=True),
# #         )
# #         self.dir_v = nn.Sequential(
# #             nn.Conv2d(c2, c2, (k, 1), padding=(k // 2, 0), groups=c2, bias=False),
# #             nn.BatchNorm2d(c2),
# #             nn.SiLU(inplace=True),
# #         )
# #         self.branch_gate = BranchSoftmaxGate(c2, branches=4)
# #         self.freq_gate = DCTFreqGate(c2, dct_size=dct_size)
# #         self.spatial_gate = SpatialGate(kernel_size=7)
# #
# #     def forward(self, x):
# #         x = self.proj(x)
# #         f_l = self.local(x)
# #         f_h = self.dir_h(x)
# #         f_v = self.dir_v(x)
# #         f_f = x - self.high_freq_pool(x)
# #         alpha = self.branch_gate(x)
# #         fused = alpha[:, 0] * f_l + alpha[:, 1] * f_h + alpha[:, 2] * f_v + alpha[:, 3] * f_f
# #         return x + self.spatial_gate(x) * self.freq_gate(x) * fused
# #
# #
# # class N_FDSF(nn.Module):
# #     """N_FDSF-v2: low-frequency guided frequency-directional selective calibration."""
# #
# #     def __init__(self, c1, c2, k=7, dct_size=7, gamma_init=0.1, freq_threshold=0.05):
# #         super().__init__()
# #         self.proj = nn.Identity() if c1 == c2 else nn.Sequential(
# #             nn.Conv2d(c1, c2, 1, bias=False),
# #             nn.BatchNorm2d(c2),
# #             nn.SiLU(inplace=True),
# #         )
# #         self.local = nn.Sequential(
# #             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
# #             nn.BatchNorm2d(c2),
# #             nn.SiLU(inplace=True),
# #         )
# #         self.high_freq_pool = nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False)
# #         self.dir_h = nn.Sequential(
# #             nn.Conv2d(c2, c2, (1, k), padding=(0, k // 2), groups=c2, bias=False),
# #             nn.BatchNorm2d(c2),
# #             nn.SiLU(inplace=True),
# #         )
# #         self.dir_v = nn.Sequential(
# #             nn.Conv2d(c2, c2, (k, 1), padding=(k // 2, 0), groups=c2, bias=False),
# #             nn.BatchNorm2d(c2),
# #             nn.SiLU(inplace=True),
# #         )
# #         self.branch_gate = BranchSoftmaxGate(c2, branches=4)
# #         self.freq_gate = DCTFreqGate(c2, dct_size=dct_size)
# #         self.low_spatial_gate = SpatialGate(kernel_size=7)
# #         self.spatial_gate = SpatialGate(kernel_size=7)
# #         self.gaussian_blur = GaussianBlurDWConv(c2)
# #         self.freq_threshold = freq_threshold
# #         self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))
# #
# #     def forward(self, x):
# #         x = self.proj(x)
# #         f_l = self.local(x)
# #         f_h = self.dir_h(x)
# #         f_v = self.dir_v(x)
# #
# #         x_low = self.high_freq_pool(x)
# #         x_high = x - x_low
# #         low_gate = self.low_spatial_gate(x_low)
# #         high_blur = self.gaussian_blur(x_high)
# #         high_conf = torch.sigmoid((x_high.abs() - self.freq_threshold) * 10.0)
# #         f_f = low_gate * (high_conf * x_high + (1.0 - high_conf) * high_blur)
# #
# #         alpha = self.branch_gate(x)
# #         fused = alpha[:, 0] * f_l + alpha[:, 1] * f_h + alpha[:, 2] * f_v + alpha[:, 3] * f_f
# #         spatial = 0.5 + self.spatial_gate(x)
# #         return x + self.gamma * spatial * self.freq_gate(x) * fused
#
# class N_FDSF(nn.Module):
#     """WDSF: Wavelet-Directional Selective Fusion (legacy symbol: N_FDSF)."""
#
#     def __init__(self, c1, c2, k=7, dct_size=7, gamma_init=0.1, freq_threshold=0.05):
#         super().__init__()
#         self.proj = nn.Identity() if c1 == c2 else nn.Sequential(
#             nn.Conv2d(c1, c2, 1, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(inplace=True),
#         )
#         self.local = nn.Sequential(
#             nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(inplace=True),
#         )
#         self.dir_h = nn.Sequential(
#             nn.Conv2d(c2, c2, (1, k), padding=(0, k // 2), groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(inplace=True),
#         )
#         self.dir_v = nn.Sequential(
#             nn.Conv2d(c2, c2, (k, 1), padding=(k // 2, 0), groups=c2, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(inplace=True),
#         )
#         self.haar = HaarDWT2D(c2)
#         self.low_spatial_gate = SpatialGate(kernel_size=7)
#         self.high_fuse = nn.Sequential(
#             nn.Conv2d(3 * c2, c2, 1, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(inplace=True),
#         )
#         self.gaussian_blur = GaussianBlurDWConv(c2)
#         self.freq_gate = DCTFreqGate(4 * c2, dct_size=dct_size)
#         self.branch_fuse = nn.Sequential(
#             nn.Conv2d(4 * c2, c2, 1, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(inplace=True),
#         )
#         self.freq_threshold = freq_threshold
#         self.gamma = nn.Parameter(torch.ones(1))
#
#     def forward(self, x):
#         x = self.proj(x)
#         f_l = self.local(x)
#         f_h = self.dir_h(x)
#         f_v = self.dir_v(x)
#
#         ll, lh, hl, hh = self.haar(x)
#         low_gate = F.interpolate(self.low_spatial_gate(ll), size=x.shape[-2:], mode="bilinear", align_corners=False)
#         high = self.high_fuse(torch.cat((lh, hl, hh), dim=1))
#         high = F.interpolate(high, size=x.shape[-2:], mode="bilinear", align_corners=False)
#         high_blur = self.gaussian_blur(high)
#         high_conf = torch.sigmoid((high.abs() - self.freq_threshold) * 10.0)
#         f_f = low_gate * (high_conf * high + (1.0 - high_conf) * high_blur)
#
#         z_c = torch.cat((f_l, f_h, f_v, f_f), dim=1)
#         delta = self.branch_fuse(self.freq_gate(z_c) * z_c)
#         return x + self.gamma * delta


class N_FDSF(nn.Module):
    """WDSF-LFP: wavelet neck fusion with NS-FPN-style low-frequency guided purification."""

    def __init__(self, c1, c2, k=7, dct_size=7, gamma_init=0.1, freq_threshold=0.5):
        super().__init__()
        self.proj = nn.Identity() if c1 == c2 else nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.local = nn.Sequential(
            nn.Conv2d(c2, c2, 3, padding=1, groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.dir_h = nn.Sequential(
            nn.Conv2d(c2, c2, (1, k), padding=(0, k // 2), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.dir_v = nn.Sequential(
            nn.Conv2d(c2, c2, (k, 1), padding=(k // 2, 0), groups=c2, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.haar = HaarDWT2D(c2)
        self.ihaar = HaarIDWT2D(c2)
        self.low_spatial_gate = SpatialGate(kernel_size=7)
        self.gaussian_filter = LearnableGaussianFilterBank(3 * c2, kernel_size=3, num_filters=1)
        self.freq_gate = DCTFreqGate(4 * c2, dct_size=dct_size)
        self.branch_fuse = nn.Sequential(
            nn.Conv2d(4 * c2, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.freq_threshold = float(freq_threshold)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def _lfp_reconstruct(self, x):
        ll, lh, hl, hh = self.haar(x)
        yh = torch.cat((lh, hl, hh), dim=1)

        low_gate = self.low_spatial_gate(ll)
        yh = yh * low_gate

        yh_blurred = self.gaussian_filter(yh)
        weak_mask = (yh.abs() < self.freq_threshold).to(dtype=yh.dtype)
        yh = yh * (1.0 - weak_mask) + yh_blurred * weak_mask

        lh, hl, hh = yh.chunk(3, dim=1)
        return self.ihaar(ll, lh, hl, hh, output_size=x.shape[-2:])

    def forward(self, x):
        x = self.proj(x)
        f_l = self.local(x)
        f_h = self.dir_h(x)
        f_v = self.dir_v(x)
        f_f = self._lfp_reconstruct(x)

        z_c = torch.cat((f_l, f_h, f_v, f_f), dim=1)
        delta = self.branch_fuse(self.freq_gate(z_c) * z_c)
        return x + self.gamma * delta



# Previous OSF_FDSF-v1 kept for reference.
# class OSF_FDSF(nn.Module):
#     # Semantic-gated shallow bridge for combining the OSF path with FDSF neck blocks.
#     # It keeps OSF as a high-resolution structural cue without adding another
#     # frequency/directional enhancement stack on top of B_FDSF and N_FDSF.
#
#     def __init__(self, ch, c2, e=0.5, eps=1e-6):
#         super().__init__()
#         if not isinstance(ch, (list, tuple)) or len(ch) != 2:
#             raise ValueError(f"OSF_FDSF expects [P2_channels, P3_channels], got {ch}")
#         hidden = max(int(c2 * e), 16)
#         self.p2_proj = nn.Sequential(
#             nn.Conv2d(ch[0], hidden, 1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#         )
#         self.p2_down = nn.Sequential(
#             nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, groups=hidden, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(hidden, hidden, 1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#         )
#         self.p3_proj = nn.Sequential(
#             nn.Conv2d(ch[1], hidden, 1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#         )
#         self.struct = nn.Sequential(
#             nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(hidden, hidden, 1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#         )
#         self.support = nn.Sequential(
#             nn.Conv2d(2 * hidden + 2, hidden, 1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(hidden, 1, 1),
#             nn.Sigmoid(),
#         )
#         self.out = nn.Sequential(
#             nn.Conv2d(2 * hidden, c2, 1, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(inplace=True),
#         )
#         self.bridge_scale_raw = nn.Parameter(torch.tensor(-2.25))
#         self.eps = float(eps)
#         self.last_stats = None
#
#     def _detail_energy(self, x):
#         detail = x - F.avg_pool2d(x, 3, 1, 1)
#         energy = detail.square().mean(dim=1, keepdim=True).add(self.eps).sqrt()
#         return detail, energy
#
#     def forward(self, x):
#         if not isinstance(x, (list, tuple)) or len(x) != 2:
#             raise ValueError("OSF_FDSF forward expects [P2, P3]")
#         p2, p3 = x
#         shallow = self.p2_down(self.p2_proj(p2))
#         semantic = self.p3_proj(p3)
#         if shallow.shape[-2:] != semantic.shape[-2:]:
#             shallow = F.interpolate(shallow, size=semantic.shape[-2:], mode="bilinear", align_corners=False)
#
#         shallow_struct = self.struct(shallow)
#         shallow_detail, shallow_energy = self._detail_energy(shallow_struct)
#         semantic_detail, semantic_energy = self._detail_energy(semantic)
#         coherence = (
#             F.normalize(shallow_detail, dim=1, eps=self.eps)
#             * F.normalize(semantic_detail, dim=1, eps=self.eps)
#         ).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
#         energy_gap = torch.log((shallow_energy + self.eps) / (semantic_energy + self.eps)).abs()
#         reliability = torch.sigmoid(coherence - energy_gap)
#         support = self.support(torch.cat((shallow_struct, semantic, reliability, shallow_energy), dim=1))
#
#         injection = self.out(torch.cat((shallow_struct, support * reliability * shallow_detail), dim=1))
#         bridge_scale = F.softplus(self.bridge_scale_raw)
#         output = bridge_scale * injection
#         if not self.training:
#             self.last_stats = {
#                 "coherence": coherence.detach(),
#                 "energy_gap": energy_gap.detach(),
#                 "reliability": reliability.detach(),
#                 "support": support.detach(),
#                 "bridge_scale": bridge_scale.detach(),
#             }
#         return output
# Previous OSF_FDSF-v2 kept for reference.
# class OSF_FDSF(nn.Module):
#     """SCCF: Semantic-Consistent Cross-scale Fusion (legacy symbol: OSF_FDSF).
#
#     P2 supplies shallow wavelet/structural cues, while P3 supplies semantic support
#     for reliability-aware cross-scale fusion before the subsequent concatenation.
#     """
#
#     def __init__(self, ch, c2, e=0.5, eps=1e-6):
#         super().__init__()
#         if not isinstance(ch, (list, tuple)) or len(ch) != 2:
#             raise ValueError(f"OSF_FDSF expects [P2_channels, P3_channels], got {ch}")
#         hidden = max(int(c2 * e), 16)
#         self.p2_proj = nn.Sequential(
#             nn.Conv2d(ch[0], hidden, 1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#         )
#         self.p2_down = nn.Sequential(
#             nn.Conv2d(hidden, hidden, 3, stride=2, padding=1, groups=hidden, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(hidden, hidden, 1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#         )
#         self.p3_proj = nn.Sequential(
#             nn.Conv2d(ch[1], hidden, 1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#         )
#         self.haar = HaarDWT2D(hidden)
#         self.low_context = nn.Sequential(
#             nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(hidden, hidden, 1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#         )
#         self.detail_fuse = nn.Sequential(
#             nn.Conv2d(3 * hidden, hidden, 1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#         )
#         self.struct = nn.Sequential(
#             nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(hidden, hidden, 1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#         )
#         self.semantic_lowpass = nn.Sequential(
#             nn.Conv2d(hidden, hidden, 5, padding=2, groups=hidden, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(hidden, hidden, 1, bias=False),
#             nn.BatchNorm2d(hidden),
#         )
#         self.support_base = nn.Sequential(
#             nn.Conv2d(2 * hidden + 3, hidden, 1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(hidden, 1, 1),
#             nn.Sigmoid(),
#         )
#         self.aca_q = nn.Sequential(
#             nn.Conv2d(hidden, hidden, (1, 3), padding=(0, 1), groups=hidden, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#         )
#         self.aca_k = nn.Sequential(
#             nn.Conv2d(hidden, hidden, (3, 1), padding=(1, 0), groups=hidden, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#         )
#         self.aca_v = nn.Sequential(
#             nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#         )
#         self.aca_out = nn.Sequential(
#             nn.Conv2d(3 * hidden, hidden, 1, bias=False),
#             nn.BatchNorm2d(hidden),
#             nn.SiLU(inplace=True),
#             nn.Conv2d(hidden, 1, 1),
#             nn.Sigmoid(),
#         )
#         self.out = nn.Sequential(
#             nn.Conv2d(3 * hidden, c2, 1, bias=False),
#             nn.BatchNorm2d(c2),
#             nn.SiLU(inplace=True),
#         )
#         self.bridge_scale_raw = nn.Parameter(torch.tensor(-2.25))
#         self.eps = float(eps)
#         self.last_stats = None
#
#     def _haar_detail_energy(self, x, target_size):
#         ll, lh, hl, hh = self.haar(x)
#         detail_bands = torch.cat((lh, hl, hh), dim=1)
#         detail = self.detail_fuse(detail_bands)
#         low = self.low_context(ll)
#         energy = detail_bands.square().mean(dim=1, keepdim=True).add(self.eps).sqrt()
#         if detail.shape[-2:] != target_size:
#             detail = F.interpolate(detail, size=target_size, mode="bilinear", align_corners=False)
#             low = F.interpolate(low, size=target_size, mode="bilinear", align_corners=False)
#             energy = F.interpolate(energy, size=target_size, mode="bilinear", align_corners=False)
#         return low, detail, energy
#
#     def _semantic_detail_energy(self, semantic):
#         semantic_low = self.semantic_lowpass(semantic)
#         detail = semantic - semantic_low
#         energy = detail.square().mean(dim=1, keepdim=True).add(self.eps).sqrt()
#         return semantic_low, detail, energy
#
#     def _local_center_score(self, energy):
#         local_mean = F.avg_pool2d(energy, 3, 1, 1)
#         local_peak = energy / (local_mean + self.eps)
#         return torch.sigmoid(local_peak - 1.0)
#
#     def forward(self, x):
#         if not isinstance(x, (list, tuple)) or len(x) != 2:
#             raise ValueError("OSF_FDSF forward expects [P2, P3]")
#         p2, p3 = x
#         p2_feat = self.p2_proj(p2)
#         semantic = self.p3_proj(p3)
#         target_size = semantic.shape[-2:]
#
#         shallow_struct = self.struct(self.p2_down(p2_feat))
#         if shallow_struct.shape[-2:] != target_size:
#             shallow_struct = F.interpolate(shallow_struct, size=target_size, mode="bilinear", align_corners=False)
#
#         shallow_low, shallow_detail, shallow_energy = self._haar_detail_energy(p2_feat, target_size)
#         semantic_low, semantic_detail, semantic_energy = self._semantic_detail_energy(semantic)
#
#         coherence = (
#             F.normalize(shallow_detail, dim=1, eps=self.eps)
#             * F.normalize(semantic_detail, dim=1, eps=self.eps)
#         ).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
#         energy_gap = torch.log((shallow_energy + self.eps) / (semantic_energy + self.eps)).abs()
#         center_score = self._local_center_score(shallow_energy)
#         reliability = torch.sigmoid(coherence - energy_gap) * center_score
#
#         support_base = self.support_base(
#             torch.cat((shallow_low, semantic_low, reliability, shallow_energy, semantic_energy), dim=1)
#         )
#         aca_support = self.aca_out(
#             torch.cat((self.aca_q(shallow_detail), self.aca_k(semantic_detail), self.aca_v(shallow_struct)), dim=1)
#         )
#         support = support_base * aca_support
#
#         gated_detail = support * reliability * shallow_detail
#         injection = self.out(torch.cat((shallow_struct, shallow_low, gated_detail), dim=1))
#         bridge_scale = F.softplus(self.bridge_scale_raw)
#         output = bridge_scale * injection
#         if not self.training:
#             self.last_stats = {
#                 "coherence": coherence.detach(),
#                 "energy_gap": energy_gap.detach(),
#                 "center_score": center_score.detach(),
#                 "reliability": reliability.detach(),
#                 "support_base": support_base.detach(),
#                 "aca_support": aca_support.detach(),
#                 "support": support.detach(),
#                 "bridge_scale": bridge_scale.detach(),
#             }
#         return output


class CrossScaleACA(nn.Module):
    # Cross-scale asymmetric correlation attention for semantic-guided detail calibration.

    def __init__(self, channels, ratio=0.5, eps=1e-6):
        super().__init__()
        inner = max(int(channels * ratio), 16)
        self.q_proj = nn.Sequential(
            nn.Conv2d(channels, inner, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(inner),
            nn.SiLU(inplace=True),
        )
        self.k_proj = nn.Sequential(
            nn.Conv2d(channels, inner, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(inner),
            nn.SiLU(inplace=True),
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.SiLU(inplace=True),
        )
        self.out = nn.Sequential(
            nn.Conv2d(3 * inner, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )
        self.eps = float(eps)

    def forward(self, shallow_detail, semantic_low):
        q = self.q_proj(shallow_detail)
        k = self.k_proj(semantic_low)
        v = self.v_proj(shallow_detail)

        b, c, h, w = q.shape
        n = h * w
        q_flat = q.flatten(2)
        k_flat = k.flatten(2)
        v_flat = v.flatten(2)

        attention = torch.softmax(torch.bmm(q_flat, k_flat.transpose(1, 2)) / math.sqrt(max(n, 1)), dim=-1)
        v_attn = torch.bmm(attention, v_flat).view(b, c, h, w)
        return self.out(torch.cat((q, k, v_attn), dim=1))


class TripleInputACA(nn.Module):
    """Three-source asymmetric channel attention.

    Q, K and V are projected independently, while the attention matrix is
    computed across channels. This avoids the quadratic spatial cost of a
    full P2 attention map.
    """

    def __init__(self, channels, ratio=0.5):
        super().__init__()
        inner = max(int(channels * ratio), 16)
        self.q_proj = nn.Sequential(
            nn.Conv2d(channels, inner, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(inner),
            nn.SiLU(inplace=True),
        )
        self.k_proj = nn.Sequential(
            nn.Conv2d(channels, inner, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(inner),
            nn.SiLU(inplace=True),
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(channels, inner, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner),
            nn.SiLU(inplace=True),
        )
        self.to_gate = nn.Sequential(
            nn.Conv2d(3 * inner, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid(),
        )

    def forward(self, q_source, k_source, v_source):
        target_size = k_source.shape[-2:]
        if q_source.shape[-2:] != target_size:
            q_source = F.adaptive_avg_pool2d(q_source, target_size)
        if v_source.shape[-2:] != target_size:
            v_source = F.interpolate(v_source, size=target_size, mode="bilinear", align_corners=False)

        q = self.q_proj(q_source)
        k = self.k_proj(k_source)
        v = self.v_proj(v_source)

        b, c, h, w = q.shape
        n = h * w
        q_flat = q.flatten(2)
        k_flat = k.flatten(2)
        v_flat = v.flatten(2)

        # [B, C, N] @ [B, N, C] -> [B, C, C]
        attention = torch.softmax(
            torch.bmm(q_flat, k_flat.transpose(1, 2)) / math.sqrt(max(n, 1)), dim=-1
        )
        attended_v = torch.bmm(attention, v_flat).view(b, c, h, w)
        return self.to_gate(torch.cat((q, k, attended_v), dim=1)), attention


class OSF_QKV(nn.Module):
    """Three-input OSF that returns semantic-guided, weighted P2 features.

    Inputs are ordered as [P2, neck_P3, backbone_P3]:
      - P2 provides Q and is also the feature being weighted.
      - neck_P3 provides K (deep backbone + neck context).
      - backbone_P3 provides V (the original P3 information).
    """

    def __init__(self, ch, c2, e=0.5, aca_ratio=0.5):
        super().__init__()
        if not isinstance(ch, (list, tuple)) or len(ch) != 3:
            raise ValueError(f"OSF_QKV expects [P2_channels, neck_P3_channels, backbone_P3_channels], got {ch}")

        hidden = max(int(c2 * e), 16)
        self.p2_proj = nn.Sequential(
            nn.Conv2d(ch[0], hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.k_proj = nn.Sequential(
            nn.Conv2d(ch[1], hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.v_proj = nn.Sequential(
            nn.Conv2d(ch[2], hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.aca = TripleInputACA(hidden, ratio=aca_ratio)
        self.out = nn.Sequential(
            nn.Conv2d(hidden, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        # Learnable positive residual-gate strength. Initialized near zero for stable training.
        self.gate_scale_raw = nn.Parameter(torch.tensor(-3.0))
        self.last_stats = None

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("OSF_QKV forward expects [P2, neck_P3, backbone_P3]")
        p2, neck_p3, backbone_p3 = x

        p2_feat = self.p2_proj(p2)
        k_source = self.k_proj(neck_p3)
        v_source = self.v_proj(backbone_p3)
        gate, attention = self.aca(p2_feat, k_source, v_source)
        gate = F.interpolate(gate, size=p2_feat.shape[-2:], mode="bilinear", align_corners=False)

        gate_scale = F.softplus(self.gate_scale_raw)
        weighted_p2 = p2_feat * (1.0 + gate_scale * gate)
        # Direct centered gating (disabled while the current training run is active):
        # gate_weight = 2.0 * gate
        # weighted_p2 = p2_feat * gate_weight
        output = self.out(weighted_p2)

        if not self.training:
            self.last_stats = {
                "gate": gate.detach(),
                "attention": attention.detach(),
                "gate_scale": gate_scale.detach(),
            }
        return output


class OSF_FDSF(nn.Module):
    # SCCF-v3: compact semantic-guided cross-scale fusion.
    # P2 supplies Haar low-frequency structure and high-frequency details.
    # P3 supplies strictly low-pass semantic support. CrossScaleACA models QK^T
    # correlation between shallow details and deep semantics, then calibrates details.

    def __init__(self, ch, c2, e=0.5, aca_ratio=0.5, eps=1e-6):
        super().__init__()
        if not isinstance(ch, (list, tuple)) or len(ch) != 2:
            raise ValueError(f"OSF_FDSF expects [P2_channels, P3_channels], got {ch}")
        hidden = max(int(c2 * e), 16)
        self.p2_proj = nn.Sequential(
            nn.Conv2d(ch[0], hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.p3_proj = nn.Sequential(
            nn.Conv2d(ch[1], hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.haar = HaarDWT2D(hidden)
        self.low_context = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.detail_fuse = nn.Sequential(
            nn.Conv2d(3 * hidden, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.semantic_lowpass = nn.AvgPool2d(5, stride=1, padding=2, count_include_pad=False)
        self.cross_aca = CrossScaleACA(hidden, ratio=aca_ratio, eps=eps)
        self.out = nn.Sequential(
            nn.Conv2d(2 * hidden, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )
        self.detail_scale_raw = nn.Parameter(torch.tensor(-2.25))
        self.bridge_scale_raw = nn.Parameter(torch.tensor(-2.25))
        self.eps = float(eps)
        self.last_stats = None

    def _haar_low_detail(self, x, target_size):
        ll, lh, hl, hh = self.haar(x)
        detail = self.detail_fuse(torch.cat((lh, hl, hh), dim=1))
        low = self.low_context(ll)
        if detail.shape[-2:] != target_size:
            detail = F.interpolate(detail, size=target_size, mode="bilinear", align_corners=False)
            low = F.interpolate(low, size=target_size, mode="bilinear", align_corners=False)
        return low, detail

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("OSF_FDSF forward expects [P2, P3]")
        p2, p3 = x
        p2_feat = self.p2_proj(p2)
        semantic = self.p3_proj(p3)
        target_size = semantic.shape[-2:]

        shallow_low, shallow_detail = self._haar_low_detail(p2_feat, target_size)
        semantic_low = self.semantic_lowpass(semantic)
        gate = self.cross_aca(shallow_detail, semantic_low)

        detail_scale = F.softplus(self.detail_scale_raw)
        calibrated_detail = shallow_detail + detail_scale * gate * shallow_detail
        injection = self.out(torch.cat((shallow_low, calibrated_detail), dim=1))
        bridge_scale = F.softplus(self.bridge_scale_raw)
        output = bridge_scale * injection

        if not self.training:
            self.last_stats = {
                "aca_gate": gate.detach(),
                "detail_scale": detail_scale.detach(),
                "bridge_scale": bridge_scale.detach(),
            }
        return output
