import torch
import numpy as np


def crop_bounding_box(img, x, y, h, w):
    x0, y0, x1, y1 = x, y, x + h, y + w
    return img[x0:x1, y0:y1]


def rot_90(img):
    return np.swapaxes(img, 0, 1)


def flip_vertical(img):
    return img[:, ::-1]


def flip_horizontal(img):
    return img[::-1, :]


def to_tensor(img):
    if isinstance(img, np.ndarray):
        img = torch.from_numpy(img)
    return img.permute(2, 0, 1)
