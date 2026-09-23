import torch
import torch.nn as nn

from ultralytics.nn.modules.block import Bottleneck, C3k
from ultralytics.nn.modules.conv import Conv
from .FDSF import HaarDWT2D, HaarIDWT2D, LearnableGaussianFilterBank, SpatialGate

__all__ = ["LowFrequencyGuidedPurification", "C3k2_LFP"]


class LowFrequencyGuidedPurification(nn.Module):
    """LFP module extracted from N_FDSF without changing the original hard-threshold logic."""

    def __init__(self, channels, freq_threshold=0.5):
        super().__init__()
        self.haar = HaarDWT2D(channels)
        self.ihaar = HaarIDWT2D(channels)
        self.low_spatial_gate = SpatialGate(kernel_size=7)
        self.gaussian_filter = LearnableGaussianFilterBank(3 * channels, kernel_size=3, num_filters=1)
        self.freq_threshold = float(freq_threshold)

    def forward(self, x):
        ll, lh, hl, hh = self.haar(x)
        yh = torch.cat((lh, hl, hh), dim=1)

        low_gate = self.low_spatial_gate(ll)
        yh = yh * low_gate

        yh_blurred = self.gaussian_filter(yh)
        weak_mask = (yh.abs() < self.freq_threshold).to(dtype=yh.dtype)
        yh = yh * (1.0 - weak_mask) + yh_blurred * weak_mask

        lh, hl, hh = yh.chunk(3, dim=1)
        return self.ihaar(ll, lh, hl, hh, output_size=x.shape[-2:])


class C3k2_LFP(nn.Module):
    """C3k2 variant that additively fuses Bottleneck and LFP features before concatenation."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True, freq_threshold=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.m = nn.Sequential(
            *(C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))
        )
        self.lfp = LowFrequencyGuidedPurification(self.c, freq_threshold=freq_threshold)
        self.lfp_weight = nn.Parameter(torch.ones(1))
        self.cv2 = Conv(2 * self.c, c2, 1)

    def forward(self, x):
        y_keep, y_work = self.cv1(x).chunk(2, 1)
        y_bottleneck = self.m(y_work)
        y_lfp = self.lfp_weight.clamp(-1.0, 1.0) * self.lfp(y_work)
        y_fused = y_bottleneck + y_lfp
        return self.cv2(torch.cat((y_keep, y_fused), dim=1))

    def forward_split(self, x):
        y_keep, y_work = self.cv1(x).split((self.c, self.c), 1)
        y_bottleneck = self.m(y_work)
        y_lfp = self.lfp_weight.clamp(-1.0, 1.0) * self.lfp(y_work)
        y_fused = y_bottleneck + y_lfp
        return self.cv2(torch.cat((y_keep, y_fused), dim=1))
