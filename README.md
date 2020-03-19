# 📷 Photo Booth
Photo Booth is a repository for Image Restauration (SR, Colorization)

## Get started

### Installation

This package can be installed with pip with:

```
python -m pip install git+https://github.com/bernardomig/photobooth.git
```

### Download weights

All weights are downloadable in [this folder](https://uapt33090-my.sharepoint.com/:f:/g/personal/bernardo_lourenco_ua_pt/EjDXHnex3RBCnaMhObpx6v0BNW7foyr6tpVim8eOKUUK6g?e=6BrAQe).


### Example application

This is an example application using the edsr_baseline_x2 model in the cpu. Remember that the model has to be first compiled to torch_jit (see [this](https://pytorch.org/tutorials/advanced/cpp_export.html)).

```python
import argparse
import numpy as np
import torch
import cv2

torch.set_num_threads(8)

mean = torch.tensor([0.4488, 0.4371, 0.4040])

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--model', required=True)
args = parser.parse_args()

print("loading model...")
model = torch.jit.load(args.model)
model = model.cpu()


def read_img(f):
    im = cv2.imread(f)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def save_img(im, f):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    return cv2.imwrite(f, im)


def normalize(im):
    return torch.from_numpy(im) \
        .float() \
        .div(255.) \
        .sub(mean) \
        .permute(2, 0, 1) \
        .unsqueeze(0)


def denormalize(im):
    im = im[0].permute(1, 2, 0)
    im = (im + mean) * 255.
    im = torch.clamp(im, 0, 255)
    im = im.byte()
    return im.numpy()


im = read_img(args.input)
im = normalize(im)
with torch.no_grad():
    im = model(im)
im = denormalize(im)
save_img(im, args.output)
```

To run this for an image, execute this script as:

```
python main.py --input input_image.png --output upscaled_x2.png --model edsr_baseline_x2.pth
```