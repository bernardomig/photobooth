import torch
from torch import nn
from torch.nn import functional as F

__all__ = ["SRCNN", "SRCNN_9_1_5", "SRCNN_9_5_5"]


def SRCNN_9_1_5(in_channels, out_channels):
    return SRCNN(in_channels, out_channels,
                 filters=[64, 32], kernel_sizes=[9, 1, 5])


def SRCNN_9_5_5(in_channels, out_channels):
    return SRCNN(in_channels, out_channels,
                 filters=[64, 32], kernel_sizes=[9, 5, 5])


class SRCNN(nn.Module):

    def __init__(self, in_channels, out_channels,
                 filters=[64, 32], kernel_sizes=[9, 1, 5]):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, filters[0],
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2)
        self.conv2 = nn.Conv2d(
            filters[0], filters[1],
            kernel_size=kernel_sizes[1],
            padding=kernel_sizes[1] // 2)
        self.conv3 = nn.Conv2d(
            filters[1], out_channels,
            kernel_size=kernel_sizes[2],
            padding=kernel_sizes[2] // 2)

        nn.init.kaiming_normal_(
            self.conv1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(
            self.conv2.weight, mode='fan_out', nonlinearity='relu')
        nn.init.xavier_normal_(self.conv3.weight)

        nn.init.zeros_(self.conv1.bias)
        nn.init.zeros_(self.conv2.bias)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.relu(self.conv2(x))
        return self.conv3(x)
