"""
The EDSR and MDSR models for image super-resolution.

Paper: Enhanced Deep Residual Networks 
       for Single Image Super-Resolution
       (arxiv: https://arxiv.org/abs/1707.02921)

"""


from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


__all__ = [
    'EDSR', 'MDSR',
    'edsr_x2', 'edsr_x3', 'edsr_x4',
    'edsr_baseline_x2', 'edsr_baseline_x3', 'edsr_baseline_x4',
    'mdsr', 'mdsr_baseline',
]


def edsr_x2(in_channels, out_channels):
    return EDSR(in_channels, out_channels,
                scale_factor=2,
                num_channels=256, num_res_blocks=32, resblock_scaling=0.1)


def edsr_x3(in_channels, out_channels):
    return EDSR(in_channels, out_channels,
                scale_factor=3,
                num_channels=256, num_res_blocks=32, resblock_scaling=0.1)


def edsr_x4(in_channels, out_channels):
    return EDSR(in_channels, out_channels,
                scale_factor=4,
                num_channels=256, num_res_blocks=32, resblock_scaling=0.1)


def edsr_baseline_x2(in_channels, out_channels):
    return EDSR(in_channels, out_channels,
                scale_factor=2,
                num_channels=64, num_res_blocks=16)


def edsr_baseline_x3(in_channels, out_channels):
    return EDSR(in_channels, out_channels,
                scale_factor=3,
                num_channels=64, num_res_blocks=16)


def edsr_baseline_x4(in_channels, out_channels):
    return EDSR(in_channels, out_channels,
                scale_factor=4,
                num_channels=64, num_res_blocks=16)


def mdsr(in_channels, out_channels):
    return MDSR(in_channels, out_channels,
                scale_factors={2, 3, 4},
                num_channels=256, num_res_blocks=32, resblock_scaling=0.1)


def mdsr_baseline(in_channels, out_channels):
    return MDSR(in_channels, out_channels,
                scale_factors={2, 3, 4},
                num_channels=64, num_res_blocks=16)


class EDSR(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,
                 num_channels=64,
                 num_res_blocks=16,
                 resblock_scaling=None):
        super().__init__()

        self.head = nn.Conv2d(
            in_channels, num_channels,
            kernel_size=3, padding=1)
        self.features = nn.Sequential(*[
            ResidualBlock(num_channels, num_channels, scaling=resblock_scaling)
            for _ in range(num_res_blocks)
        ])
        self.upsampling = UpsamplingBlock(num_channels, num_channels, scale_factor)
        self.tail = nn.Conv2d(
            num_channels, out_channels,
            kernel_size=3, padding=1)

    def forward(self, input):
        x = self.head(input)
        features = self.features(x)
        x = x + features
        x = self.upsampling(x)
        return self.tail(x)


class MDSR(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factors=[2, 3, 4],
                 num_channels=64,
                 num_res_blocks=16,
                 resblock_scaling=None):
        super().__init__()

        self.head = nn.ModuleDict({
            'scale{}'.format(scale): nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2)
            for scale in scale_factors
        })
        self.features = nn.Sequential(*[
            ResidualBlock(num_channels, num_channels, scaling=resblock_scaling)
            for _ in range(num_res_blocks)
        ])
        self.upsampling = nn.ModuleDict({
            'scale{}'.format(scale): UpsamplingBlock(num_channels, num_channels, scale)
            for scale in scale_factors
        })
        self.tail = nn.Conv2d(num_channels, out_channels,
                              kernel_size=3, padding=1)

    def forward(self, input, scale_factor):
        x = self.head['scale{}'.format(scale_factor)](input)
        features = self.features(x)
        x = x + features
        x = self.upsampling['scale{}'.format(scale_factor)](x)
        return self.tail(x)


def UpsamplingBlock(in_channels, out_channels, scale_factor):
    if in_channels != out_channels:
        raise ValueError("input and output channels must match: {} != {}."
                         .format(in_channels, out_channels))

    if scale_factor in {2, 3}:
        return nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(
                in_channels, in_channels * scale_factor**2,
                kernel_size=3, padding=1)),
            ('shuffle', nn.PixelShuffle(scale_factor)),
        ]))
    elif scale_factor == 4:
        return nn.Sequential(
            UpsamplingBlock(in_channels, in_channels, scale_factor=2),
            UpsamplingBlock(in_channels, in_channels, scale_factor=2),
        )
    else:
        raise ValueError(
            "scale_factor should be either 2, 3 or 4, "
            "got {}".format(scale_factor))


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, scaling=1.0):
        super().__init__()

        self.scaling = scaling

        self.conv1 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=3, padding=1)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = self.conv2(x)
        if self.scaling is not None:
            x = self.scaling * x
        return input + x
