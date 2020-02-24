import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

from apex import amp

import numpy as np
from numpy import array

from ignite.contrib.handlers import ProgressBar

from photobooth.engines.sr_supervised import create_sr_evaluator, create_sr_trainer
from photobooth.data.datasets import DIV2K
from photobooth.models import edsr


DEVICE = torch.device('cuda', 0)
EPOCHS = 500
MEAN = torch.tensor([0.4488, 0.4371, 0.4040])
DS_ROOT = '/srv/datasets/DIV2K'
BATCH_SIZE = 16


def train_tfms(scale_factor, crop_size, mean):
    def _transform(lowres, highres):
        h, w, _ = lowres.shape

        x = np.random.randint(h // crop_size)
        y = np.random.randint(w // crop_size)

        x = x * crop_size, (x + 1) * crop_size
        y = y * crop_size, (y + 1) * crop_size
        lr_crop = lowres[x[0]:x[1], y[0]:y[1]]
        x = x[0] * scale_factor, x[1] * scale_factor
        y = y[0] * scale_factor, y[1] * scale_factor
        hr_crop = highres[x[0]:x[1], y[0]:y[1]]

        if np.random.rand() > 0.5:
            # Rotate by 90
            lr_crop = np.swapaxes(lr_crop, 0, 1)
            hr_crop = np.swapaxes(hr_crop, 0, 1)

        if np.random.rand() > 0.5:
            # Vertical Flip
            lr_crop = lr_crop[::-1, :, :]
            hr_crop = hr_crop[::-1, :, :]

        if np.random.rand() > 0.5:
            # Horizontal Flip
            lr_crop = lr_crop[:, ::-1, :]
            hr_crop = hr_crop[:, ::-1, :]

        # To torch tensor
        lr_crop = torch.from_numpy(lr_crop.astype('f4') / 255.)
        hr_crop = torch.from_numpy(hr_crop.astype('f4') / 255.)
        # Normalize
        lr_crop.sub_(MEAN)
        hr_crop.sub_(MEAN)
        # Swap axis from HWC to CHW
        lr_crop = lr_crop.permute(2, 0, 1)
        hr_crop = hr_crop.permute(2, 0, 1)

        # make it bw
        # lr_crop = lr_crop.mean(0)
        # hr_crop = hr_crop.mean(0)
        return {'lowres': lr_crop, 'highres': hr_crop}
    return _transform


def valid_tfms(scale_factor, crop_size, mean):
    def _transform(lowres, highres):
        h, w, _ = lowres.shape

        x, y = h // 2, w // 2

        x = x - crop_size // 2, x + crop_size // 2
        y = y - crop_size // 2, y + crop_size // 2
        lr_crop = lowres[x[0]:x[1], y[0]:y[1]]
        x = x[0] * scale_factor, x[1] * scale_factor
        y = y[0] * scale_factor, y[1] * scale_factor
        hr_crop = highres[x[0]:x[1], y[0]:y[1]]

        # To torch tensor
        lr_crop = torch.from_numpy(lr_crop.astype('f4') / 255.)
        hr_crop = torch.from_numpy(hr_crop.astype('f4') / 255.)
        # Normalize
        lr_crop.sub_(MEAN)
        hr_crop.sub_(MEAN)
        # Swap axis from HWC to CHW
        lr_crop = lr_crop.permute(2, 0, 1)
        hr_crop = hr_crop.permute(2, 0, 1)

        # make it bw
        # lr_crop = lr_crop.mean(0)
        # hr_crop = hr_crop.mean(0)
        return {'lowres': lr_crop, 'highres': hr_crop}
    return _transform


train_ds = DIV2K(DS_ROOT, split='train', config='bicubic/x2',
                 transforms=train_tfms(2, 96, MEAN))
val_ds = DIV2K(DS_ROOT, split='valid', config='bicubic/x2',
               transforms=valid_tfms(2, 400, MEAN))

train_loader = DataLoader(train_ds, batch_size=16,
                          shuffle=True, num_workers=16)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                        num_workers=16, drop_last=False)

model = edsr.EDSR_baseline_2x(3, 3)
model = model.to(DEVICE)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3, weight_decay=1e-4,
)
loss_fn = torch.nn.L1Loss()
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=100, gamma=0.5)

(model, loss_fn), optimizer = amp.initialize([model, loss_fn], optimizer)

trainer = create_sr_trainer(
    model,
    loss_fn,
    optimizer,
    mixed_precision=True,
)
ProgressBar(persist=False).attach(trainer, ['loss', 'pnsr'])
