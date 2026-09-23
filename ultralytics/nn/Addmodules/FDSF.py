import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DCTFreqGate", "SpectralBandGateBranch", "B_FDSF",  "OSF_QKV"]

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



class SpectralBandGateBranch(nn.Module):
    """SBG-v3: FFT branch with fixed band centers and learnable soft bandwidths."""

    def __init__(
        self,
        channels,
        band_split_low=0.33,
        band_split_high=0.66,
        reduction=4,
        fft_norm="ortho",
        min_bandwidth=0.04,
        max_bandwidth=0.25,
    ):
        super().__init__()
        edges = self._sanitize_edges(band_split_low, band_split_high)
        self.num_bands = len(edges) - 1
        self.min_bandwidth = float(min_bandwidth)
        self.max_bandwidth = float(max(max_bandwidth, min_bandwidth + 1e-4))

        centers = [(edges[i] + edges[i + 1]) * 0.5 for i in range(self.num_bands)]
        widths = [max((edges[i + 1] - edges[i]) * 0.5, self.min_bandwidth + 1e-4) for i in range(self.num_bands)]
        self.register_buffer("band_centers", torch.tensor(centers, dtype=torch.float32), persistent=False)
        self.band_width_raw = nn.Parameter(torch.tensor([self._width_to_raw(v) for v in widths], dtype=torch.float32))

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
        self._radius_cache = {}

    @staticmethod
    def _sanitize_edges(band_split_low, band_split_high):
        edges = [0.0, float(band_split_low), float(band_split_high), 1.0]
        edges = sorted(min(max(edge, 0.0), 1.0) for edge in edges)
        clean = []
        for edge in edges:
            if not clean or edge - clean[-1] > 1e-4:
                clean.append(edge)
        if clean[0] > 0.0:
            clean.insert(0, 0.0)
        if clean[-1] < 1.0:
            clean.append(1.0)
        if len(clean) < 2:
            clean = [0.0, 1.0]
        return clean

    @staticmethod
    def _logit(value, eps=1e-4):
        value = min(max(float(value), eps), 1.0 - eps)
        return math.log(value / (1.0 - value))

    def _width_to_raw(self, width):
        value = (float(width) - self.min_bandwidth) / (self.max_bandwidth - self.min_bandwidth)
        return self._logit(value)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_radius_cache"] = {}
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "_radius_cache" not in self.__dict__:
            self._radius_cache = {}

    def _get_radius_grid(self, h, w, device, dtype):
        key = (h, w, device, dtype)
        cached = self._radius_cache.get(key)
        if cached is not None and cached.device == device and cached.dtype == dtype:
            return cached

        y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, w, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        radius = torch.sqrt(xx * xx + yy * yy).clamp(0.0, 1.0)
        self._radius_cache[key] = radius
        return radius

    def _get_band_masks(self, radius):
        dtype = radius.dtype
        centers = self.band_centers.to(dtype=dtype, device=radius.device)
        widths = self.min_bandwidth + (self.max_bandwidth - self.min_bandwidth) * torch.sigmoid(self.band_width_raw)
        widths = widths.to(dtype=dtype, device=radius.device).clamp_min(1e-4)

        distance = (radius.unsqueeze(0) - centers.view(-1, 1, 1)) / widths.view(-1, 1, 1)
        masks = torch.exp(-0.5 * distance.square())
        return masks / masks.sum(dim=0, keepdim=True).clamp_min(1e-6)

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
            radius = self._get_radius_grid(ffted.shape[-2], ffted.shape[-1], ffted.device, ffted.real.dtype)
            masks = self._get_band_masks(radius)

            amp = torch.abs(ffted).mean(dim=1)
            mask_sums = masks.flatten(1).sum(dim=1).clamp_min(1e-6).to(amp.dtype)
            band_stats = torch.einsum("bhw,khw->bk", amp, masks) / mask_sums.unsqueeze(0)
            gate = 1.0 + torch.tanh(self._band_gate_fp32(band_stats))
            gating_map = torch.einsum("bk,khw->bhw", gate, masks).unsqueeze(1)

            ffted = self._complex_conv(ffted * gating_map)
            out = torch.fft.irfftn(ffted, s=x_fp32.shape[-2:], dim=(-2, -1), norm=self.fft_norm)
        return out.to(orig_dtype)

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


