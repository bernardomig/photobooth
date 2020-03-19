import torch
from torch import nn
from torch.nn import functional as F


class SRDescriminator(nn.Sequential):

    def __init__(self, in_channels, out_channels):
        layers = [
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            ConvBlock(64, 128, 2),
            ConvBlock(128, 128),
            ConvBlock(128, 256, 2),
            ConvBlock(256, 256),
            ConvBlock(256, 512, 2),
            ConvBlock(512, 512),
            ConvBlock(512, 1024, 2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, out_channels)
        ]
        super().__init__(*layers)


def ConvBlock(in_channels, out_channels, stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels,
                  kernel_size=3, padding=1, stride=stride, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )
