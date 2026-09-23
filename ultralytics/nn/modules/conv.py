# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import pywt.data
from functools import partial


_all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "Index",
    "OSF",
    "SCSF",
    "SCSFv2",
    "WTConv",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """
    Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """
        Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))



class Conv2(Conv):
    """
    Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """
        Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """
        Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """
        Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """
        Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """
        Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """
        Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class DWConvNoBN(nn.Module):
    """Depth-wise convolution without BN and activation."""
    def __init__(self, c1, c2, k=1, s=1, d=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, None, d),
                             groups=math.gcd(c1, c2), dilation=d, bias=False)

    def forward(self, x):
        return self.conv(x)




class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """
        Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """
    Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """
        Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """
        Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """
        Apply activation and convolution transpose operation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """
    Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """
        Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Apply Focus operation and convolution to input tensor.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """
    Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/ghostnet
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """
        Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """
        Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """
        Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """
        Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """
        Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """
        Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (tuple): Tuple containing:
                - Equivalent kernel (torch.Tensor)
                - Equivalent bias (torch.Tensor)
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """
        Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """
        Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            (tuple): Tuple containing:
                - Fused kernel (torch.Tensor)
                - Fused bias (torch.Tensor)
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """
    Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """
        Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """
    Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """
        Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """
        Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """
        Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """
        Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """
    Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """
        Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """
        Concatenate input tensors along specified dimension.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """
    Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """
        Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Select and return a particular index from input.

        Args:
            x (List[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]

#--------------------------------------------OSF2.0---------------------------------------------#
class OSF(nn.Module):
    """
    Omni-Scale Fusion module for small object detection.
    Combines local, context and global information in a lightweight manner.

    Attributes:
        local_conv (nn.Module): Local branch convolution
        context_convs (nn.ModuleList): Context branch convolutions
        light_att (nn.Module): Global attention branch
    """

    def __init__(self, c1, c2, e=0.5):
        """
        Initialize OSF module.

        Args:
            c1 (int): Input channels
            c2 (int, optional): Output channels.
            k (int): Base kernel size for local branch
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 1, 1)

        # Local branch (small kernel)
        self.local = Conv(c_, c_, 3, act=False)

        # Context branch (multiple medium kernels)
        self.context_convs = nn.ModuleList([
            DWConv(c_, c_, (1, 7), act=False),  # 1x7
            DWConv(c_, c_, (7, 1), act=False)  # 7x1
        ])

        # Global branch (lightweight attention)
        self.light_att = EfficientLocalizationAttention(c_)
        # self.light_att = CBAM(c_)
        # self.light_att = ECA(c_)
        # self.light_att = EMA(c_)
        # self.light_att = SE_block(c_)
        # self.light_att = CoordAtt(c_,c_)
    def forward(self, x):
        """Forward pass through OSF module."""
        y = self.cv1(x)

        # Context branch
        context = [conv(y) for conv in self.context_convs]

        return self.cv2(y + self.local(y) + sum(context) + self.light_att(y))


class CSFAv2(nn.Module):
    """CSFA with P3-conditioned spatial positive/negative frequency residuals."""

    def __init__(self, ch, c2, e=0.5, temperature=0.25, eps=1e-6):
        super().__init__()
        if not isinstance(ch, (list, tuple)) or len(ch) != 2:
            raise ValueError(f"CSFAv2 expects [P2_channels, P3_channels], got {ch}")
        hidden = max(int(c2 * e), 8)
        self.p2_in = Conv(ch[0], hidden, 1, 1)
        self.local = Conv(hidden, hidden, 3, act=False)
        self.context_h = DWConv(hidden, hidden, (1, 7), act=False)
        self.context_v = DWConv(hidden, hidden, (7, 1), act=False)
        self.location = EfficientLocalizationAttention(hidden)
        self.p2_out = Conv(hidden, c2, 1, 1)
        self.p2_down = Conv(c2, c2, 3, 2, act=False)
        self.p3_proj = nn.Identity() if ch[1] == c2 else Conv(ch[1], c2, 1, 1, act=False)
        self.detail_proj = nn.Conv2d(c2, c2, 1, bias=False)
        self.semantic_support = nn.Sequential(nn.Conv2d(c2 * 2, 1, 1), nn.Sigmoid())
        self.fuse = Conv(c2 * 2, c2, 1, 1, act=False)
        # Softplus keeps both gains non-negative. -3 initializes them near zero (0.0498),
        # so training starts close to the original CSFA fusion rather than an identity map.
        self.positive_gain_raw = nn.Parameter(torch.tensor(-3.0))
        self.negative_gain_raw = nn.Parameter(torch.tensor(-3.0))
        self.temperature = float(temperature)
        self.eps = float(eps)
        self.last_stats = None

    def _local_reliability(self, shallow_energy, semantic_energy):
        shallow_mean = F.avg_pool2d(shallow_energy, 3, 1, 1)
        semantic_mean = F.avg_pool2d(semantic_energy, 3, 1, 1)
        shallow_centered = shallow_energy - shallow_mean
        semantic_centered = semantic_energy - semantic_mean
        numerator = F.avg_pool2d(shallow_centered * semantic_centered, 3, 1, 1)
        denominator = (
            F.avg_pool2d(shallow_centered.square(), 3, 1, 1)
            * F.avg_pool2d(semantic_centered.square(), 3, 1, 1)
            + self.eps
        ).sqrt()
        coherence = (numerator / denominator).clamp(-1.0, 1.0)
        log_mismatch = (
            torch.log(shallow_energy / (shallow_mean + self.eps) + self.eps)
            - torch.log(semantic_energy / (semantic_mean + self.eps) + self.eps)
        ).abs()
        reliability = torch.sigmoid((coherence - log_mismatch) / max(self.temperature, self.eps))
        return reliability, coherence, log_mismatch

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("CSFAv2 forward expects [P2, P3]")
        p2, p3 = x
        shallow = self.p2_in(p2)
        enhanced = self.p2_out(
            shallow + self.local(shallow) + self.context_h(shallow) + self.context_v(shallow) + self.location(shallow)
        )
        semantic = self.p3_proj(p3)
        aligned = self.p2_down(enhanced)
        if aligned.shape[-2:] != semantic.shape[-2:]:
            aligned = F.interpolate(aligned, size=semantic.shape[-2:], mode="bilinear", align_corners=False)

        # High-pass residuals provide explicit local-frequency evidence while the original
        # CSFA multi-branch feature remains the main information path.
        shallow_high = enhanced - F.avg_pool2d(enhanced, 3, 1, 1)
        shallow_high = F.avg_pool2d(shallow_high, 2, 2, ceil_mode=True)
        if shallow_high.shape[-2:] != semantic.shape[-2:]:
            shallow_high = F.interpolate(shallow_high, size=semantic.shape[-2:], mode="bilinear", align_corners=False)
        semantic_high = semantic - F.avg_pool2d(semantic, 3, 1, 1)
        shallow_energy = shallow_high.square().mean(1, keepdim=True).add(self.eps).sqrt()
        semantic_energy = semantic_high.square().mean(1, keepdim=True).add(self.eps).sqrt()
        reliability, coherence, mismatch = self._local_reliability(shallow_energy, semantic_energy)
        support = self.semantic_support(torch.cat((semantic, aligned), dim=1))
        positive_gate = reliability * support
        negative_gate = (1.0 - reliability) * (1.0 - support)
        detail = self.detail_proj(shallow_high)
        positive_gain = F.softplus(self.positive_gain_raw)
        negative_gain = F.softplus(self.negative_gain_raw)
        calibrated = aligned + positive_gain * positive_gate * detail - negative_gain * negative_gate * detail
        output = self.fuse(torch.cat((semantic, calibrated), dim=1))

        if not self.training:
            self.last_stats = {
                "reliability": reliability.detach(),
                "coherence": coherence.detach(),
                "mismatch": mismatch.detach(),
                "semantic_support": support.detach(),
                "positive_gate": positive_gate.detach(),
                "negative_gate": negative_gate.detach(),
                "positive_gain": positive_gain.detach(),
                "negative_gain": negative_gain.detach(),
            }
        return F.silu(output)


class CSFAv3(nn.Module):
    """Topology-controlled CSFA: calibrate P2 detail but leave P2/P3 concatenation outside."""

    def __init__(self, ch, c2, e=0.5, temperature=0.75, mismatch_weight=0.5, eps=1e-6):
        super().__init__()
        if not isinstance(ch, (list, tuple)) or len(ch) != 2:
            raise ValueError(f"CSFAv3 expects [P2_channels, P3_channels], got {ch}")
        hidden = max(int(c2 * e), 8)
        self.p2_in = Conv(ch[0], hidden, 1, 1)
        self.local = Conv(hidden, hidden, 3, act=False)
        self.context_h = DWConv(hidden, hidden, (1, 7), act=False)
        self.context_v = DWConv(hidden, hidden, (7, 1), act=False)
        self.location = EfficientLocalizationAttention(hidden)
        self.p2_out = Conv(hidden, c2, 1, 1)
        self.p2_down = Conv(c2, c2, 3, 2, act=False)
        self.p3_proj = nn.Identity() if ch[1] == c2 else Conv(ch[1], c2, 1, 1, act=False)
        self.gain_raw = nn.Parameter(torch.tensor(-2.25))
        self.temperature = float(temperature)
        self.mismatch_weight = float(mismatch_weight)
        self.eps = float(eps)
        self.last_stats = None

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("CSFAv3 forward expects [P2, P3]")
        p2, p3 = x
        shallow = self.p2_in(p2)
        enhanced = self.p2_out(
            shallow + self.local(shallow) + self.context_h(shallow) + self.context_v(shallow) + self.location(shallow)
        )
        aligned = self.p2_down(enhanced)
        semantic = self.p3_proj(p3)
        if aligned.shape[-2:] != semantic.shape[-2:]:
            aligned = F.interpolate(aligned, size=semantic.shape[-2:], mode="bilinear", align_corners=False)

        shallow_high = aligned - F.avg_pool2d(aligned, 3, 1, 1)
        semantic_high = semantic - F.avg_pool2d(semantic, 3, 1, 1)
        shallow_energy = shallow_high.square().mean(1, keepdim=True).add(self.eps).sqrt()
        semantic_energy = semantic_high.square().mean(1, keepdim=True).add(self.eps).sqrt()
        shallow_mean = F.avg_pool2d(shallow_energy, 3, 1, 1)
        semantic_mean = F.avg_pool2d(semantic_energy, 3, 1, 1)
        shallow_centered = shallow_energy - shallow_mean
        semantic_centered = semantic_energy - semantic_mean
        coherence = F.avg_pool2d(shallow_centered * semantic_centered, 3, 1, 1) / (
            F.avg_pool2d(shallow_centered.square(), 3, 1, 1)
            * F.avg_pool2d(semantic_centered.square(), 3, 1, 1)
            + self.eps
        ).sqrt()
        coherence = coherence.clamp(-1.0, 1.0)
        mismatch = (
            torch.log(shallow_energy / (shallow_mean + self.eps) + self.eps)
            - torch.log(semantic_energy / (semantic_mean + self.eps) + self.eps)
        ).abs()
        score = coherence - self.mismatch_weight * mismatch
        # Per-image spatial standardization prevents the reliability map collapsing to a
        # constant and expresses whether each location is more or less reliable than its image context.
        score_mean = score.mean(dim=(-2, -1), keepdim=True)
        score_std = score.var(dim=(-2, -1), keepdim=True, unbiased=False).add(self.eps).sqrt()
        signed_gate = torch.tanh((score - score_mean) / (score_std * self.temperature + self.eps))
        gain = F.softplus(self.gain_raw)
        output = aligned + gain * signed_gate * shallow_high

        if not self.training:
            self.last_stats = {
                "coherence": coherence.detach(),
                "mismatch": mismatch.detach(),
                "score": score.detach(),
                "signed_gate": signed_gate.detach(),
                "gain": gain.detach(),
            }
        return output


class SpectralTap(nn.Module):
    """Training-only identity tap exposing aligned P2 and P3 features to an auxiliary loss."""

    def __init__(self):
        super().__init__()
        self.enabled = False
        self.p2 = None
        self.p3 = None

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("SpectralTap expects [aligned_P2, P3]")
        if self.training and self.enabled:
            self.p2, self.p3 = x
        else:
            self.p2 = self.p3 = None
        return x[0]


class SCSF(nn.Module):
    """Semantic-consistent Haar subband fusion for a high-resolution P2 and semantic P3 feature pair."""

    def __init__(self, ch, c2, e=0.5, temperature=1.0, variance_weight=0.25, eps=1e-6):
        super().__init__()
        if not isinstance(ch, (list, tuple)) or len(ch) != 2:
            raise ValueError(f"SCSF expects [P2_channels, P3_channels], got {ch}")
        hidden = max(int(c2 * e), 8)
        self.p2_proj = Conv(ch[0], hidden, 1, 1)
        self.p3_proj = Conv(ch[1], hidden, 1, 1)
        self.detail_predictors = nn.ModuleList(
            nn.Sequential(DWConv(hidden, hidden, 3), nn.Conv2d(hidden, hidden, 1, bias=False)) for _ in range(3)
        )
        self.detail_projections = nn.ModuleList(nn.Conv2d(hidden, hidden, 1, bias=False) for _ in range(3))
        self.gate = nn.Sequential(nn.Conv2d(hidden * 2 + 1, hidden, 1), nn.Sigmoid())
        self.semantic_out = Conv(hidden, c2, 1, 1, act=False)
        self.injection_out = Conv(hidden * 2, c2, 1, 1, act=False)
        self.temperature = float(temperature)
        self.variance_weight = float(variance_weight)
        self.eps = float(eps)
        self.last_stats = None

        # Orthonormal 2-D Haar analysis filters: LL, horizontal, vertical, diagonal.
        scale = 0.5
        filters = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ]
        ).unsqueeze(1) * scale
        self.register_buffer("haar_filters", filters, persistent=True)

    def _haar_analysis(self, x):
        b, c, h, w = x.shape
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
        weight = self.haar_filters.to(dtype=x.dtype).repeat(c, 1, 1, 1)
        subbands = F.conv2d(x, weight, stride=2, groups=c)
        subbands = subbands.view(b, c, 4, subbands.shape[-2], subbands.shape[-1])
        return subbands.unbind(dim=2)

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("SCSF forward expects [P2, P3]")
        p2, p3 = x
        low, *details = self._haar_analysis(self.p2_proj(p2))
        semantic = self.p3_proj(p3)
        target_size = semantic.shape[-2:]
        if low.shape[-2:] != target_size:
            low = F.interpolate(low, size=target_size, mode="bilinear", align_corners=False)
            details = [F.interpolate(z, size=target_size, mode="bilinear", align_corners=False) for z in details]

        scores, residual_energies, correlations, projected_details = [], [], [], []
        for detail, predictor, projector in zip(details, self.detail_predictors, self.detail_projections):
            expected = predictor(semantic)
            correlation = (F.normalize(detail, dim=1, eps=self.eps) * F.normalize(expected, dim=1, eps=self.eps)).sum(
                dim=1, keepdim=True
            )
            residual_energy = F.avg_pool2d((detail - expected).square().mean(dim=1, keepdim=True), 3, 1, 1)
            score = correlation / max(self.temperature, self.eps) - self.variance_weight * torch.log(
                residual_energy + self.eps
            )
            correlations.append(correlation)
            residual_energies.append(residual_energy)
            scores.append(score)
            projected_details.append(projector(detail))

        reliability = torch.softmax(torch.cat(scores, dim=1), dim=1)
        reliable_detail = sum(
            reliability[:, i : i + 1] * projected_details[i] for i in range(len(projected_details))
        )
        entropy = -(reliability * torch.log(reliability + self.eps)).sum(dim=1, keepdim=True) / math.log(3.0)
        confidence = 1.0 - entropy
        injection_gate = self.gate(torch.cat((semantic, low, confidence), dim=1))
        output = self.semantic_out(semantic) + self.injection_out(torch.cat((low, injection_gate * reliable_detail), dim=1))

        if not self.training:
            self.last_stats = {
                "reliability": reliability.detach(),
                "residual_energy": torch.cat(residual_energies, dim=1).detach(),
                "correlation": torch.cat(correlations, dim=1).detach(),
                "injection_gate": injection_gate.detach(),
            }
        return F.silu(output)


class SCSFv2(nn.Module):
    """SCSF with fixed directional coherence and zero-initialized residual injection."""

    def __init__(self, ch, c2, e=0.5, temperature=0.25, mismatch_weight=0.25, eps=1e-6):
        super().__init__()
        if not isinstance(ch, (list, tuple)) or len(ch) != 2:
            raise ValueError(f"SCSFv2 expects [P2_channels, P3_channels], got {ch}")
        hidden = max(int(c2 * e), 8)
        self.p2_proj = Conv(ch[0], hidden, 1, 1)
        self.p3_analysis = Conv(ch[1], hidden, 1, 1)
        self.semantic_skip = nn.Identity() if ch[1] == c2 else Conv(ch[1], c2, 1, 1, act=False)
        self.detail_projections = nn.ModuleList(nn.Conv2d(hidden, hidden, 1, bias=False) for _ in range(3))
        self.semantic_support = nn.Sequential(nn.Conv2d(hidden, 1, 1), nn.Sigmoid())
        self.injection_out = Conv(hidden * 2, c2, 1, 1, act=False)
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self.temperature = float(temperature)
        self.mismatch_weight = float(mismatch_weight)
        self.eps = float(eps)
        self.last_stats = None

        haar = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[-1.0, -1.0], [1.0, 1.0]],
                [[-1.0, 1.0], [-1.0, 1.0]],
                [[1.0, -1.0], [-1.0, 1.0]],
            ]
        ).unsqueeze(1) * 0.5
        self.register_buffer("haar_filters", haar, persistent=True)

        # Vertical derivative, horizontal derivative, and mixed derivative. They correspond
        # to the LH, HL, and HH Haar detail bands without introducing learned orientation priors.
        directional = torch.tensor(
            [
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
            ]
        ).unsqueeze(1)
        directional[0:2] /= 8.0
        directional[2:3] /= 2.0
        self.register_buffer("directional_filters", directional, persistent=True)

    def _haar_analysis(self, x):
        b, c, h, w = x.shape
        if h % 2 or w % 2:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")
        weight = self.haar_filters.to(dtype=x.dtype).repeat(c, 1, 1, 1)
        bands = F.conv2d(x, weight, stride=2, groups=c)
        return bands.view(b, c, 4, bands.shape[-2], bands.shape[-1]).unbind(dim=2)

    def _semantic_directions(self, semantic):
        b, c, h, w = semantic.shape
        weight = self.directional_filters.to(dtype=semantic.dtype).repeat(c, 1, 1, 1)
        response = F.conv2d(semantic, weight, padding=1, groups=c)
        return response.view(b, c, 3, h, w).unbind(dim=2)

    def _local_coherence(self, shallow_energy, semantic_energy):
        shallow_mean = F.avg_pool2d(shallow_energy, 3, 1, 1)
        semantic_mean = F.avg_pool2d(semantic_energy, 3, 1, 1)
        shallow_centered = shallow_energy - shallow_mean
        semantic_centered = semantic_energy - semantic_mean
        numerator = F.avg_pool2d(shallow_centered * semantic_centered, 3, 1, 1)
        denominator = (
            F.avg_pool2d(shallow_centered.square(), 3, 1, 1)
            * F.avg_pool2d(semantic_centered.square(), 3, 1, 1)
            + self.eps
        ).sqrt()
        coherence = (numerator / denominator).clamp(-1.0, 1.0)

        shallow_relative = shallow_energy / (shallow_mean + self.eps)
        semantic_relative = semantic_energy / (semantic_mean + self.eps)
        mismatch = (torch.log(shallow_relative + self.eps) - torch.log(semantic_relative + self.eps)).abs()
        return coherence, mismatch

    def forward(self, x):
        if not isinstance(x, (list, tuple)) or len(x) != 2:
            raise ValueError("SCSFv2 forward expects [P2, P3]")
        p2, p3 = x
        low, *details = self._haar_analysis(self.p2_proj(p2))
        semantic = self.p3_analysis(p3)
        target_size = semantic.shape[-2:]
        if low.shape[-2:] != target_size:
            low = F.interpolate(low, size=target_size, mode="bilinear", align_corners=False)
            details = [F.interpolate(z, size=target_size, mode="bilinear", align_corners=False) for z in details]

        semantic_directions = self._semantic_directions(semantic)
        coherences, mismatches, scores, projected_details = [], [], [], []
        for detail, semantic_direction, projection in zip(details, semantic_directions, self.detail_projections):
            shallow_energy = detail.square().mean(dim=1, keepdim=True).add(self.eps).sqrt()
            semantic_energy = semantic_direction.square().mean(dim=1, keepdim=True).add(self.eps).sqrt()
            coherence, mismatch = self._local_coherence(shallow_energy, semantic_energy)
            score = (coherence - self.mismatch_weight * mismatch) / max(self.temperature, self.eps)
            coherences.append(coherence)
            mismatches.append(mismatch)
            scores.append(score)
            projected_details.append(projection(detail))

        reliability = torch.softmax(torch.cat(scores, dim=1), dim=1)
        reliable_detail = sum(
            reliability[:, i : i + 1] * projected_details[i] for i in range(len(projected_details))
        )
        entropy = -(reliability * torch.log(reliability + self.eps)).sum(dim=1, keepdim=True) / math.log(3.0)
        confidence = 1.0 - entropy
        support = self.semantic_support(semantic) * (0.5 + 0.5 * confidence)
        injection = self.injection_out(torch.cat((low, support * reliable_detail), dim=1))
        scale = torch.tanh(self.residual_scale)
        output = self.semantic_skip(p3) + scale * injection

        if not self.training:
            self.last_stats = {
                "reliability": reliability.detach(),
                "coherence": torch.cat(coherences, dim=1).detach(),
                "mismatch": torch.cat(mismatches, dim=1).detach(),
                "entropy": entropy.detach(),
                "semantic_support": support.detach(),
                "residual_scale": scale.detach(),
            }
        return output

#------------------------------------------------------ELA-2024--------------------------------------------------------#
class EfficientLocalizationAttention(nn.Module):
    def __init__(self, channel, kernel_size=7):
        super(EfficientLocalizationAttention, self).__init__()
        self.pad = kernel_size // 2
        self.conv = nn.Conv1d(channel, channel, kernel_size=kernel_size, padding=self.pad, groups=channel, bias=False)
        self.gn = nn.GroupNorm(16, channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()

        # 处理高度维度
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)

        # 处理宽度维度
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)
        x_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        # 在两个维度上应用注意力
        return x * x_h * x_w


#----------------------------------------------------------SE----------------------------------------------------------#
class SE_block(nn.Module):
    def __init__(self, channel, scaling=16, use_inplace=True):
        """
        SE注意力模块
        :param channel: 输入特征图的通道数
        :param scaling: 中间层的缩放比例，默认为16
        :param use_inplace: 是否使用inplace操作，默认为True
        """
        super(SE_block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // scaling, bias=False),  # 压缩通道
            nn.ReLU(inplace=use_inplace),  # 激活函数
            nn.Linear(channel // scaling, channel, bias=False),  # 恢复通道
            nn.Sigmoid()  # 生成注意力权重
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: 全局平均池化并展平
        y = self.avg_pool(x).flatten(1)
        # Excitation: 通过全连接层生成注意力权重
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: 对输入特征图进行重标定
        return x * y

#----------------------------------------------------CA--------------------------------------------------#
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, groups=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // groups)

        self.conv1 = Conv(inp, mip, 1, 1)
        self.conv2 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        x_h = self.conv2(x_h).sigmoid()
        x_w = self.conv3(x_w).sigmoid()
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        y = identity * x_w * x_h

        return y

#----------------------------------------------------EMA-2023-ICCASSP--------------------------------------------------#
class EMA(nn.Module):
    def __init__(self, channels, c2=None, factor=32):
        super(EMA, self).__init__()
        self.groups = factor
        assert channels // self.groups > 0
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(channels // self.groups, channels // self.groups)
        self.conv1x1 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(channels // self.groups, channels // self.groups, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)  # b*g,c//g,h,w
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)  # b*g, c//g, hw
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(b * self.groups, 1, h, w)
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)

#-------------------------------------------------------ECA------------------------------------------------------------#

class ECA(nn.Module):
    def __init__(self, channel, b=1, gamma=2):
        super(ECA, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x * y.expand_as(x)
        return out

#-----------------------------------------------WaveConv-2024-ECCV-----------------------------------------------------#

def create_wavelet_filter(wave, in_size, out_size, type=torch.float):
    w = pywt.Wavelet(wave)
    dec_hi = torch.tensor(w.dec_hi[::-1], dtype=type)
    dec_lo = torch.tensor(w.dec_lo[::-1], dtype=type)
    dec_filters = torch.stack([dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1),
                               dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)], dim=0)

    dec_filters = dec_filters[:, None].repeat(in_size, 1, 1, 1)

    rec_hi = torch.tensor(w.rec_hi[::-1], dtype=type).flip(dims=[0])
    rec_lo = torch.tensor(w.rec_lo[::-1], dtype=type).flip(dims=[0])
    rec_filters = torch.stack([rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                               rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)], dim=0)

    rec_filters = rec_filters[:, None].repeat(out_size, 1, 1, 1)

    return dec_filters, rec_filters

def wavelet_transform(x, filters):
    b, c, h, w = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = F.conv2d(x, filters, stride=2, groups=c, padding=pad)
    x = x.reshape(b, c, 4, h // 2, w // 2)
    return x


def inverse_wavelet_transform(x, filters):
    b, c, _, h_half, w_half = x.shape
    pad = (filters.shape[2] // 2 - 1, filters.shape[3] // 2 - 1)
    x = x.reshape(b, c * 4, h_half, w_half)
    x = F.conv_transpose2d(x, filters, stride=2, groups=c, padding=pad)
    return x


class WTConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True, wt_levels=1, wt_type='db1'):
        super(WTConv2d, self).__init__()

        assert in_channels == out_channels

        self.in_channels = in_channels
        self.wt_levels = wt_levels
        self.stride = stride
        self.dilation = 1

        self.wt_filter, self.iwt_filter = create_wavelet_filter(wt_type, in_channels, in_channels, torch.float)
        self.wt_filter = nn.Parameter(self.wt_filter, requires_grad=False)
        self.iwt_filter = nn.Parameter(self.iwt_filter, requires_grad=False)

        self.base_conv = nn.Conv2d(in_channels, in_channels, kernel_size, padding='same', stride=1, dilation=1,
                                   groups=in_channels, bias=bias)
        self.base_scale = _ScaleModule([1, in_channels, 1, 1])

        self.wavelet_convs = nn.ModuleList(
            [nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size, padding='same', stride=1, dilation=1,
                       groups=in_channels * 4, bias=False) for _ in range(self.wt_levels)]
        )
        self.wavelet_scale = nn.ModuleList(
            [_ScaleModule([1, in_channels * 4, 1, 1], init_scale=0.1) for _ in range(self.wt_levels)]
        )

        if self.stride > 1:
            self.do_stride = nn.AvgPool2d(kernel_size=1, stride=stride)
        else:
            self.do_stride = None

    def forward(self, x):

        x_ll_in_levels = []
        x_h_in_levels = []
        shapes_in_levels = []

        curr_x_ll = x

        for i in range(self.wt_levels):
            curr_shape = curr_x_ll.shape
            shapes_in_levels.append(curr_shape)
            if (curr_shape[2] % 2 > 0) or (curr_shape[3] % 2 > 0):
                curr_pads = (0, curr_shape[3] % 2, 0, curr_shape[2] % 2)
                curr_x_ll = F.pad(curr_x_ll, curr_pads)

            curr_x = wavelet_transform(curr_x_ll, self.wt_filter)
            curr_x_ll = curr_x[:, :, 0, :, :]

            shape_x = curr_x.shape
            curr_x_tag = curr_x.reshape(shape_x[0], shape_x[1] * 4, shape_x[3], shape_x[4])
            curr_x_tag = self.wavelet_scale[i](self.wavelet_convs[i](curr_x_tag))
            curr_x_tag = curr_x_tag.reshape(shape_x)

            x_ll_in_levels.append(curr_x_tag[:, :, 0, :, :])
            x_h_in_levels.append(curr_x_tag[:, :, 1:4, :, :])

        next_x_ll = 0

        for i in range(self.wt_levels - 1, -1, -1):
            curr_x_ll = x_ll_in_levels.pop()
            curr_x_h = x_h_in_levels.pop()
            curr_shape = shapes_in_levels.pop()

            curr_x_ll = curr_x_ll + next_x_ll

            curr_x = torch.cat([curr_x_ll.unsqueeze(2), curr_x_h], dim=2)
            next_x_ll = inverse_wavelet_transform(curr_x, self.iwt_filter)

            next_x_ll = next_x_ll[:, :, :curr_shape[2], :curr_shape[3]]

        x_tag = next_x_ll
        assert len(x_ll_in_levels) == 0

        x = self.base_scale(self.base_conv(x))
        x = x + x_tag

        if self.do_stride is not None:
            x = self.do_stride(x)

        return x


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class WTConv(nn.Module):
    def __init__(self, c1, c2, k=3, s=2):
        """
        WTConv
        Args:
            c1 (int): Input channels
            c2 (int, optional): Output channels.
            k (int): Kernel size for WTConv2d and base_conv (usually 3 or 5)
            s (int): Stride for downsampling (usually 2)
        """
        super().__init__()
        self.wtcv0 = WTConv2d(c1, c1, k, s)
        self.cv1 = Conv(c1, c2, 1,1)

    def forward(self, x):
        return self.cv1(self.wtcv0(x))


#-----------------------------------------------WaveletPooling-2024-ECCV-----------------------------------------------------#
class WaveletDownsampleV2(nn.Module):
    """
    使用DWT进行下采样，支持自定义输出通道数。

    处理流程:
    1. 对输入张量 (B, C_in, H, W) 进行2D-DWT。
    2. 选择LL和LH两个子带进行拼接，得到 (B, 2*C_in, H/2, W/2) 的张量。
    3. 使用一个 1x1 卷积将通道数从 2*C_in 调整到指定的 C_out。
    """

    def __init__(self, c1, c2, wavelet='haar'):
        """
        初始化模块。
        :param c1: 输入通道数 (in_channels)。
        :param c2: 输出通道数 (out_channels)。
        :param wavelet: 使用的小波基。
        """
        super().__init__()
        self.c1 = c1
        self.c2 = c2

        # 内部通道数为输入通道数的2倍
        intermediate_channels = c1 * 2

        # 使用1x1卷积来调整通道数，并添加BatchNorm和激活函数
        self.conv = nn.Conv2d(intermediate_channels, c2, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()  # SiLU (或Swish) 是YOLOv8中常用的激活函数

        # 检查小波基是否有效
        if wavelet not in pywt.wavelist(kind='discrete'):
            raise ValueError(f"Wavelet '{wavelet}' is not a valid discrete wavelet.")
        self.wavelet = wavelet

    def forward(self, x):
        """
        前向传播。
        输入 x 的形状: (B, C_in, H, W)
        输出形状: (B, C_out, H/2, W/2)
        """
        # 获取输入张量的维度
        B, C, H, W = x.shape

        # 确认输入通道数与初始化时一致
        assert C == self.c1, f"Input channel {C} does not match initialized channel {self.c1}"

        # 为了提升效率，我们将 (B, C, H, W) reshape 为 (B*C, H, W)
        # 这样可以避免在Python中对通道进行循环
        x_reshaped = x.view(B * C, H, W)

        # 准备存储DWT结果的列表
        coeffs_ll = []
        coeffs_lh = []

        # 在CPU上执行DWT
        x_np = x_reshaped.cpu().numpy()
        for i in range(x_np.shape[0]):
            LL, (LH, _, _) = pywt.dwt2(x_np[i], self.wavelet)
            coeffs_ll.append(torch.from_numpy(LL))
            coeffs_lh.append(torch.from_numpy(LH))

        # 将列表堆叠成张量
        ll_tensor = torch.stack(coeffs_ll, dim=0).to(x.device)
        lh_tensor = torch.stack(coeffs_lh, dim=0).to(x.device)

        # 恢复批次和通道维度
        _, h_half, w_half = ll_tensor.shape
        ll_tensor = ll_tensor.view(B, C, h_half, w_half)
        lh_tensor = lh_tensor.view(B, C, h_half, w_half)

        # 在通道维度上拼接LL和LH分量
        # (B, C, H/2, W/2) + (B, C, H/2, W/2) -> (B, 2*C, H/2, W/2)
        wavelet_features = torch.cat([ll_tensor, lh_tensor], dim=1)

        # 使用1x1卷积调整通道数，并应用BN和激活函数
        output = self.act(self.bn(self.conv(wavelet_features)))

        return output


# --- 使用示例 ---
if __name__ == '__main__':
    # 定义输入和输出通道数
    in_channels = 32
    out_channels = 64

    # 创建一个虚拟的输入张量
    # (批次大小, 输入通道, 高, 宽)
    dummy_input = torch.randn(4, in_channels, 128, 128)

    # 初始化我们的小波下采样模块
    # 它现在像一个标准的卷积层一样被初始化
    wavelet_downsampler_v2 = WaveletDownsampleV2(c1=in_channels, c2=out_channels, wavelet='haar')

    # 执行前向传播
    output = wavelet_downsampler_v2(dummy_input)

    # 打印输入和输出的形状以验证
    print(f"模块: WaveletDownsampleV2(in_channels={in_channels}, out_channels={out_channels})")
    print(f"输入张量形状: {dummy_input.shape}")
    print(f"输出张量形状: {output.shape}")

    # 期望的输出形状: (4, out_channels, 128/2, 128/2) -> (4, 64, 64, 64)
    expected_shape = (4, out_channels, 64, 64)
    assert output.shape == expected_shape, f"形状不匹配! 期望: {expected_shape}, 得到: {output.shape}"
    print("\n代码运行成功，模块功能符合预期！")
