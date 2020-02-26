import torch
from torch import nn


class Normalize(nn.Module):
    """A 
    """

    def __init__(self, module, mean, clamp=True):
        super().__init__()
        self.module = module
        self.weight = nn.Parameter(mean, False)
        self.clamp = clamp

    def forward(self, input):
        if self.training:
            raise AssertionError(
                "the Normalize wrapper is only available for evaluation. "
                "Do not use during training.")

        input = input - self.weight
        output = self.module(self.module(input))
        if self.clamp:
            output = output.clamp(0.0, 1.0)
        return output
