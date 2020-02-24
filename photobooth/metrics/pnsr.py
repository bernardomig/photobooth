from math import log10
import torch

from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
from ignite.exceptions import NotComputableError

__all__ = ["PNSR"]


class PNSR(Metric):

    def __init__(self, max_value, output_transform=lambda x: x, device=None):
        self.max_value = max_value
        self._pnsr = None
        self._num_examples = None
        super(PNSR, self).__init__(
            output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._pnsr = 0.
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        mse = torch.mean((y_pred - y)**2, dim=(1, 2, 3))
        psnr = 10 * torch.log10(self.max_value / mse)
        self._pnsr += torch.sum(psnr)
        self._num_examples += y_pred.shape[0]

    @sync_all_reduce("_pnsr", "_num_examples")
    def compute(self):
        if self._num_examples == 0:
            raise NotComputableError(
                "PNSR must have at least one "
                "example before it can be computed.")

        return self._pnsr / self._num_examples
