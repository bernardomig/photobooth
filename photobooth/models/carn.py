from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from math import log2, ceil

__all__ = [
    'CARN_2x', 'CARN_3x', 'CARN_4x',
    'CARN_mobile_2x', 'CARN_mobile_3x', 'CARN_mobile_4x',
]


def CARN_2x(in_channels, out_channels):
    inner_block = partial(CascadingBlock, block=ResidualBlock, num_blocks=3)
    return CARN(in_channels, out_channels, scale_factor=2,
                num_channels=64, num_blocks=3, inner_block=inner_block)


def CARN_3x(in_channels, out_channels):
    inner_block = partial(CascadingBlock, block=ResidualBlock, num_blocks=3)
    return CARN(in_channels, out_channels, scale_factor=3,
                num_channels=64, num_blocks=3, inner_block=inner_block)


def CARN_4x(in_channels, out_channels):
    inner_block = partial(CascadingBlock, block=ResidualBlock, num_blocks=3)
    return CARN(in_channels, out_channels, scale_factor=4,
                num_channels=64, num_blocks=3, inner_block=inner_block)


def CARN_mobile_2x(in_channels, out_channels, groups=1):
    inner_block = partial(RecursiveBlock,
                          block=partial(EResidualBlock, groups=groups),
                          num_blocks=3)
    return CARN(in_channels, out_channels, scale_factor=2,
                num_channels=64, num_blocks=3, inner_block=inner_block)


def CARN_mobile_3x(in_channels, out_channels, groups=1):
    inner_block = partial(RecursiveBlock,
                          block=partial(EResidualBlock, groups=groups),
                          num_blocks=3)
    return CARN(in_channels, out_channels, scale_factor=3,
                num_channels=64, num_blocks=3, inner_block=inner_block)


def CARN_mobile_4x(in_channels, out_channels, groups=1):
    inner_block = partial(RecursiveBlock,
                          block=partial(EResidualBlock, groups=groups),
                          num_blocks=3)
    return CARN(in_channels, out_channels, scale_factor=3,
                num_channels=64, num_blocks=3, inner_block=inner_block)


class CARN(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor, num_channels, num_blocks, inner_block):
        super().__init__()

        self.head = nn.Conv2d(in_channels, num_channels,
                              kernel_size=3, padding=1)

        self.features = CascadingBlock(num_channels, num_channels,
                                       block=inner_block,
                                       num_blocks=num_blocks)
        self.upsample = UpscaleBlock(num_channels, num_channels, scale_factor)
        self.tail = nn.Conv2d(num_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, input):
        x = self.head(input)
        x = self.features(x)
        x = self.upsample(x)
        x = self.tail(x)
        return x


def UpscaleBlock(in_channels, out_channels, scale_factor):
    if in_channels != out_channels:
        raise ValueError("input and output channels must match. "
                         "Got {} != {}".format(in_channels, out_channels))

    if scale_factor < 2:
        raise ValueError("scale factor has to be at least 2. Got {}"
                         .format(scale_factor))

    layers = []

    if scale_factor == 3:
        layers += [nn.Conv2d(in_channels, 9 * in_channels,
                             kernel_size=3, padding=1),
                   nn.ReLU(inplace=True)]
        layers += [nn.PixelShuffle(3)]
    elif scale_factor in {2, 4, 8}:
        for _ in range(int(log2(scale_factor))):
            layers += [nn.Conv2d(in_channels, 4 * in_channels,
                                 kernel_size=3, padding=1),
                       nn.ReLU(inplace=True)]
            layers += [nn.PixelShuffle(2)]
    return nn.Sequential(*layers)


class RecursiveBlock(nn.Module):

    def __init__(self, in_channels, out_channels, block, num_blocks):
        super().__init__()

        if in_channels != out_channels:
            raise ValueError("input and output channels must match. "
                             "Got {} != {}".format(in_channels, out_channels))

        self.residual = block(in_channels, in_channels)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels * (i + 2), in_channels, kernel_size=1)
            for i in range(num_blocks)
        ])

    def forward(self, input):
        x = r = input

        for layer in self.convs.children():
            x_ = self.residual(x)
            r = torch.cat([r, x_], dim=1)
            x = layer(r)

        return x


class CascadingBlock(nn.ModuleList):

    def __init__(self, in_channels, out_channels, block, num_blocks):

        if in_channels != out_channels:
            raise ValueError("input and output channels must match. "
                             "Got {} != {}".format(in_channels, out_channels))

        layers = [
            nn.ModuleDict({
                'residual': block(in_channels, in_channels),
                'conv': nn.Conv2d(in_channels * (i + 2), in_channels, kernel_size=1)
            })
            for i in range(num_blocks)
        ]

        super().__init__(layers)

    def forward(self, input):
        x = r = input

        for layer in self.children():
            x_ = layer['residual'](x)
            r = torch.cat([r, x_], dim=1)
            x = layer['conv'](r)

        return x


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, input):
        x = F.relu(self.conv1(input), inplace=True)
        x = self.conv2(x)
        return F.relu(x + input, inplace=True)


class EResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, groups=groups)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, input):
        x = F.relu(self.conv1(input), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = self.conv3(x)
        return F.relu(x + input, inplace=True)
