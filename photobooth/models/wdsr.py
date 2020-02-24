from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

__all__ = [
    'WDSR',
    'WDSR_A', 'WDSR_B',
]


def WDSR_A(in_channels, out_channels, scale_factor):
    return WDSR(in_channels, out_channels, scale_factor,
                num_channels=32, num_res_blocks=8,
                resblock=partial(ResidualBlockA, expansion_ratio=4))


def WDSR_B(in_channels, out_channels, scale_factor):
    return WDSR(in_channels, out_channels, scale_factor,
                num_channels=32, num_res_blocks=8,
                resblock=partial(ResidualBlockB, expansion_ratio=6))


class WDSR(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor,
                 num_channels=32, num_res_blocks=8,
                 resblock=None):
        super().__init__()

        self.head = weight_norm(nn.Conv2d(
            in_channels, num_channels,
            kernel_size=3, padding=1))
        self.features = nn.Sequential(*[
            resblock(num_channels, num_channels)
            for _ in range(num_res_blocks)
        ])
        self.upscaling = nn.Sequential(
            nn.Conv2d(num_channels, out_channels * scale_factor**2,
                      kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
        )

        self.skip = nn.Sequential(
            weight_norm(nn.Conv2d(in_channels, out_channels * scale_factor**2,
                                  kernel_size=5, padding=2)),
            nn.PixelShuffle(scale_factor),
        )

    def forward(self, input):
        x = self.head(input)
        x = self.features(x)
        x = self.upscaling(x)

        s = self.skip(input)

        return s + x


class ResidualBlockA(nn.Module):
    def __init__(self, in_channels, out_channels,
                 expansion_ratio, scaling=None):
        super().__init__()

        self.scaling = scaling

        expansion_channels = in_channels * expansion_ratio

        self.conv1 = weight_norm(nn.Conv2d(
            in_channels, expansion_channels,
            kernel_size=3, padding=1))
        self.conv2 = weight_norm(nn.Conv2d(
            expansion_channels, out_channels,
            kernel_size=3, padding=1))

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.conv2(x)
        if self.scaling is not None:
            x = self.scaling * x
        return input + x


class ResidualBlockB(nn.Module):
    def __init__(self, in_channels, out_channels,
                 expansion_ratio, lowrank_ratio=0.8, scaling=None):
        super().__init__()

        self.scaling = scaling

        self.conv1 = weight_norm(nn.Conv2d(
            in_channels, in_channels * expansion_ratio,
            kernel_size=1
        ))
        self.conv2 = weight_norm(nn.Conv2d(
            in_channels * expansion_ratio, int(out_channels * lowrank_ratio),
            kernel_size=3, padding=1
        ))
        self.conv3 = weight_norm(nn.Conv2d(
            int(out_channels, lowrank_ratio), out_channels,
            kernel_size=1
        ))

    def forward(self, input):
        x = self.conv1(input)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        if self.scaling is not None:
            x = self.scaling * x
        return x + input
