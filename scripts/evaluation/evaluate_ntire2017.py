import torch
from torch.utils.data import DataLoader

import os
import numpy as np
import cv2

from photobooth.engines.sr_supervised import create_sr_evaluator
from photobooth.data.datasets import DIV2K

from photobooth.models import edsr

from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
from tqdm import tqdm

DS_ROOT = '/srv/datasets/DIV2K'
MEAN = torch.tensor([0.4488, 0.4371, 0.4040])


def create_transforms(upscale, mean):
    def transform(lowres, highres):
        lr_crop = lowres
        hr_crop = highres

        lr_crop = torch.from_numpy(lr_crop.astype('f4') / 255.)
        hr_crop = torch.from_numpy(hr_crop.astype('f4') / 255.)
        # Normalize
        lr_crop.sub_(MEAN)
        hr_crop.sub_(MEAN)
        # Swap axis from HWC to CHW
        lr_crop = lr_crop.permute(2, 0, 1)
        hr_crop = hr_crop.permute(2, 0, 1)

        return {'lowres': lr_crop, 'highres': hr_crop}

    return transform


val_ds = DIV2K(DS_ROOT, split='valid', config='bicubic/x4',
               transforms=create_transforms(4, MEAN))
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                        num_workers=16, drop_last=False)

device = torch.device('cuda')

model = edsr.edsr_baseline_x4(3, 3)
checkpoint = torch.load(
    'checkpoints/model_model_1690_pnsr=27.356374740600586.pth', map_location='cpu'
)

model.load_state_dict(checkpoint)
model.to(torch.device('cuda'))

# evaluator = create_sr_evaluator(
#     model,
#     device=torch.device('cuda'),
#     denormalize=True,
#     mean=MEAN)

# state = evaluator.run(val_loader)

# print(state.metrics)


def denormalize(input):
    mean = MEAN.to(input.device)
    mean = mean.reshape(1, 3, 1, 1)
    img = torch.clamp((input + mean) * 1., 0, 1.)
    return img[:, :, 9:-9, 9:-9]


mse_total = 0.
pnsr_total = 0.
ssim_total = 0.
num_examples = 0

for X, y in tqdm(val_loader):
    X = X.to(device, non_blocking=True)
    y = y.to(device, non_blocking=True)
    with torch.no_grad():
        y_pred = model(X)

    y_pred = denormalize(y_pred).permute(0, 2, 3, 1).cpu().numpy()
    y = denormalize(y).permute(0, 2, 3, 1).cpu().numpy()

    num_examples += y_pred.shape[0]

    for id in range(y_pred.shape[0]):
        im1 = y_pred[id]
        im2 = y[id]
        mse_total += mean_squared_error(im1, im2)
        pnsr_total += peak_signal_noise_ratio(im1, im2)
        ssim_total += structural_similarity(im1,
                                            im2, multichannel=True, win_size=7)


print("mse_total = ", mse_total / num_examples)
print("pnsr = ", pnsr_total / num_examples)
print("ssim = ", ssim_total / num_examples)
