import torch
from torch import nn
from torch.nn import functional as F


class StackedCrossModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, num_blocks):
        super().__init__()

        if in_channels == out_channels:
            raise ValueError("input and output channels must match.")

        self.head = nn.Conv2d(in_channels, in_channels,
                              kernel_size=3, padding=1)
        self.features = nn.Sequential(*[
            CrossModule(in_channels, out_channels, kernel_size=kernel_size)
            for _ in range(num_blocks)
        ])

    def forward(self, input):
        x = self.head(input)
        left, right = self.features((x, x))
        return (left + right) + x


class CrossModule(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        if isinstance(kernel_size, [list, tuple]):
            if len(kernel_size) != 2:
                raise ValueError(
                    "kernel size must be a single number for Merge-and-run (MR) "
                    "modules and a tuple of two values for a Multi-scale module.")
            left_kernel, right_kernel = kernel_size
        else:
            left_kernel = right_kernel = kernel_size

        self.left = nn.Sequential(
            ConvBlock(in_channels, in_channels, left_kernel,
                      padding=left_kernel // 2),
            ConvBlock(in_channels, in_channels, left_kernel,
                      padding=left_kernel // 2),
        )

        self.right = nn.Sequential(
            ConvBlock(in_channels, in_channels, right_kernel,
                      padding=right_kernel // 2),
            ConvBlock(in_channels, in_channels, right_kernel,
                      padding=right_kernel // 2),
        )

    def forward(self, input):
        left, right = input
        average = 0.5 * (left + right)
        left = average * self.left(left)
        right = average * self.right(right)
        return left, right


def ConvBlock(in_channels, out_channels, kernel_size, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(),
    )
