"""Semantic-Guided Feature Calibration (SGFC)."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SGFC"]


class _TripleSourceACA(nn.Module):
    """SGFC-v2 channel correlation attention for three feature sources."""

    def __init__(self, channels, ratio=0.5):
        super().__init__()
        inner = max(int(channels * ratio), 16)

        # SGFC-v1 (kept for reference): Q and K used different directional projections.
        # self.q_proj: 1x3 Conv -> BN -> SiLU
        # self.k_proj: 3x1 Conv -> BN -> SiLU

        # SGFC-v2: Q and K use the same 3x1 -> 1x3 factorized processing.
        # Their architectures are identical, while their parameters remain independent.
        self.q_proj = self._make_qk_projection(channels, inner)
        self.k_proj = self._make_qk_projection(channels, inner)
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

    @staticmethod
    def _make_qk_projection(channels, inner):
        return nn.Sequential(
            nn.Conv2d(channels, inner, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(inner),
            nn.SiLU(inplace=True),
            nn.Conv2d(inner, inner, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(inner),
            nn.SiLU(inplace=True),
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

        # Channel correlation: [B, C, N] @ [B, N, C] -> [B, C, C].
        attention = torch.softmax(
            torch.bmm(q_flat, k_flat.transpose(1, 2)) / math.sqrt(max(n, 1)), dim=-1
        )
        attended_v = torch.bmm(attention, v_flat).view(b, c, h, w)
        gate = self.to_gate(torch.cat((q, k, attended_v), dim=1))
        return gate, attention


class SGFC(nn.Module):
    """Semantic-Guided Feature Calibration.

    Input order is [P2, backbone_P3, neck_P3]:
      - P2 is the high-resolution query source and calibrated feature.
      - backbone_P3 is the key source.
      - neck_P3 is the value source carrying neck-enhanced semantics.
    """

    def __init__(self, ch, c2, e=0.5, aca_ratio=0.5):
        super().__init__()
        if not isinstance(ch, (list, tuple)) or len(ch) != 3:
            raise ValueError(
                f"SGFC expects [P2_channels, backbone_P3_channels, neck_P3_channels], got {ch}"
            )

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
        self.aca = _TripleSourceACA(hidden, ratio=aca_ratio)
        self.out = nn.Sequential(
            nn.Conv2d(hidden, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        # SGFC-v1 (disabled): self.gate_scale_raw = nn.Parameter(torch.tensor(-3.0))
        self.last_stats = None

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 3:
            raise ValueError("SGFC forward expects [P2, backbone_P3, neck_P3]")
        p2, backbone_p3, neck_p3 = x

        p2_feat = self.p2_proj(p2)
        k_source = self.k_proj(backbone_p3)
        v_source = self.v_proj(neck_p3)
        gate, attention = self.aca(p2_feat, k_source, v_source)
        gate = F.interpolate(gate, size=p2_feat.shape[-2:], mode="bilinear", align_corners=False)

        # SGFC-v1 (disabled):
        # gate_scale = F.softplus(self.gate_scale_raw)
        # calibrated_p2 = p2_feat * (1.0 + gate_scale * gate)

        # SGFC-v2: direct centered calibration without an extra Softplus scale.
        gate_weight = 2.0 * gate
        calibrated_p2 = p2_feat * gate_weight
        output = self.out(calibrated_p2)

        if not self.training:
            self.last_stats = {
                "gate": gate.detach(),
                "gate_weight": gate_weight.detach(),
                "attention": attention.detach(),
            }
        return output
