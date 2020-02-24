import torch
from torch import nn
from torch.nn import functional as F


__all__ = [
    'EDSR',
    'EDSR_2x', 'EDSR_3x', 'EDSR_4x',
    'EDSR_baseline_2x', 'EDSR_baseline_3x', 'EDSR_baseline_4x',
]


def EDSR_2x(in_channels, out_channels):
    return EDSR(in_channels, out_channels,
                scale_factor=2,
                num_channels=256, num_res_blocks=32, resblock_scaling=0.1)


def EDSR_3x(in_channels, out_channels):
    return EDSR(in_channels, out_channels,
                scale_factor=3,
                num_channels=256, num_res_blocks=32, resblock_scaling=0.1)


def EDSR_4x(in_channels, out_channels):
    return EDSR(in_channels, out_channels,
                scale_factor=4,
                num_channels=256, num_res_blocks=32, resblock_scaling=0.1)


def EDSR_baseline_2x(in_channels, out_channels):
    return EDSR(in_channels, out_channels,
                scale_factor=2,
                num_channels=64, num_res_blocks=16, resblock_scaling=None)


def EDSR_baseline_3x(in_channels, out_channels):
    return EDSR(in_channels, out_channels,
                scale_factor=3,
                num_channels=64, num_res_blocks=16, resblock_scaling=None)


def EDSR_baseline_4x(in_channels, out_channels):
    return EDSR(in_channels, out_channels,
                scale_factor=4,
                num_channels=64, num_res_blocks=16, resblock_scaling=None)


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
        if scale_factor in {2, 3}:
            self.upsampling = Upsampling(num_channels, scale_factor)
        elif scale_factor == 4:
            self.upsampling = nn.Sequential(
                Upsampling(num_channels, 2),
                Upsampling(num_channels, 2),
            )
        else:
            raise ValueError(
                "scale_factor should be either 2, 3 or 4, "
                "got {}".format(scale_factor))
        self.tail = nn.Conv2d(
            num_channels, out_channels,
            kernel_size=3, padding=1)

    def forward(self, input):
        x = self.head(input)
        features = self.features(x)
        x = x + features
        x = self.upsampling(x)
        return self.tail(x)


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


class Upsampling(nn.Module):

    def __init__(self, channels, scale_factor):
        super().__init__()

        self.conv = nn.Conv2d(
            channels, channels * scale_factor**2,
            kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(
            upscale_factor=scale_factor)

    def forward(self, input):
        return self.shuffle(self.conv(input))
