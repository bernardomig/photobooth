import torch
from torch import nn
from torch.nn import functional as F


class VDSR(nn.Module):

    def __init__(self, in_channels, out_channels,
                 num_channels=64, num_res_blocks=20):
        super().__init__()

        if num_res_blocks < 2:
            raise ValueError("num_res_blocks has to be at least 2. Got {}"
                             .format(num_channels))

        self.head = ConvBlock(in_channels, num_channels,
                              kernel_size=3, padding=1)
        self.features = nn.Sequential(*[
            ConvBlock(num_channels, num_channels, kernel_size=3, padding=1)
            for _ in range(num_res_blocks - 2)
        ])
        self.tail = nn.Conv2d(num_channels, out_channels,
                              kernel_size=3, padding=1)

    def forward(self, input):
        x = self.head(input)
        x = self.features(x)
        x = self.tail(x)
        return x + input


def ConvBlock(in_channels, out_channels, kernel_size,
              padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
        nn.PReLU(),
    )
