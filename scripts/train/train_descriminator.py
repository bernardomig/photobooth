import os
import sys
import argparse

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from torchvision.transforms import functional as tfms

# import wandb

from apex import amp

import numpy as np
from numpy import array

from ignite.engine import Events, Engine, _prepare_batch
from ignite.metrics import RunningAverage
from ignite.handlers import TerminateOnNan, ModelCheckpoint, DiskSaver, global_step_from_engine
from ignite.contrib.handlers import ProgressBar


from photobooth.engines.sr_supervised import create_sr_evaluator, create_sr_trainer
from photobooth.data.datasets import DIV2K
from photobooth.models import edsr
from photobooth.models.srgan import SRDescriminator
from photobooth.transforms import flip_horizontal, flip_vertical, to_tensor, crop_bounding_box, rot_90


MEAN = torch.tensor([0.4488, 0.4371, 0.4040])
DS_ROOT = '/srv/datasets/DIV2K'


def train_tfms(crop_size, mean):
    def _transform(lowres, highres):
        h, w, _ = lowres.shape
        scaling_factor = highres.shape[0] // lowres.shape[0]

        x = np.random.randint(h // crop_size)
        y = np.random.randint(w // crop_size)

        lr = crop_bounding_box(
            lowres,
            x * crop_size, y * crop_size,
            crop_size, crop_size)
        hr = crop_bounding_box(
            highres,
            x * crop_size * scaling_factor, y * crop_size * scaling_factor,
            crop_size * scaling_factor, crop_size * scaling_factor)

        if np.random.rand() > 0.5:
            # Rotate by 90
            lr = rot_90(lr)
            hr = rot_90(hr)

        if np.random.rand() > 0.5:
            # Vertical Flip
            lr = flip_vertical(lr)
            hr = flip_vertical(hr)
        if np.random.rand() > 0.5:
            # Horizontal Flip
            lr = flip_horizontal(lr)
            hr = flip_horizontal(hr)

        # Normalize
        lr = lr / 255. - MEAN.numpy()
        hr = hr / 255. - MEAN.numpy()

        # To torch tensor
        lr = torch.from_numpy(lr.astype('f4'))
        hr = torch.from_numpy(hr.astype('f4'))

        lr = lr.permute(2, 0, 1)
        hr = hr.permute(2, 0, 1)

        return {'lowres': lr, 'highres': hr}
    return _transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-4)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--crop_size', type=int, default=48)
    parser.add_argument('--state_dict', type=str, required=False)
    parser.add_argument('--bootstrap', action='store_true')
    parser.add_argument('--distributed', action='store_true')
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--local_rank', type=int)

    args = parser.parse_args()

    if args.distributed:
        dist.init_process_group('nccl', init_method='env://')
        world_size = dist.get_world_size()
        world_rank = dist.get_rank()
        local_rank = args.local_rank
    else:
        local_rank = 0

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda')

    train_ds = DIV2K(DS_ROOT, split='train', config='bicubic/x4',
                     transforms=train_tfms(args.crop_size, MEAN))

    if args.distributed:
        sampler_args = dict(num_replicas=world_size, rank=local_rank)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=8,
        sampler=(DistributedSampler(train_ds, **sampler_args) if args.distributed else None)
    )

    model = edsr.edsr_baseline_x4(3, 3)
    checkpoint = torch.load('weights/edsr_baseline_x4_pnsr=27.36.pth', map_location='cpu')
    model.load_state_dict(checkpoint)

    model = model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    descriminator = SRDescriminator(3, 1)
    descriminator = descriminator.to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(descriminator.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)

    if args.mixed_precision:
        (model, descriminator), optimizer = amp.initialize([model, descriminator], optimizer)

    if args.distributed:
        descriminator = DistributedDataParallel(descriminator, device_ids=[local_rank])

    def _update_model(engine, batch):
        x, y = _prepare_batch(batch, device=device, non_blocking=True)

        optimizer.zero_grad()
        with torch.no_grad():
            fake = model(x)
        real = y
        x_gan = torch.cat([fake, real], dim=0)
        y_gan = torch.cat([
            torch.zeros(fake.size(0), 1),
            torch.ones(real.size(0), 1)
        ]).to(device)

        y_pred = descriminator(x_gan)

        loss = loss_fn(y_pred, y_gan)

        if args.mixed_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()
        return loss

    trainer = Engine(_update_model)
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    ProgressBar(persist=False).attach(trainer, ['loss'])

    if local_rank == 0:
        checkpointer = ModelCheckpoint(
            dirname='checkpoints',
            filename_prefix='model',
            score_name='loss',
            score_function=lambda engine: engine.state.metrics['loss'],
            n_saved=5,
            global_step_transform=global_step_from_engine(trainer),
        )
        trainer.add_event_handler(
            Events.COMPLETED, checkpointer,
            to_save={'descriminator': descriminator if not args.distributed else descriminator.module})

    trainer.run(train_loader, max_epochs=args.epochs)
