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

from ignite.engine import Events
from ignite.handlers import TerminateOnNan, ModelCheckpoint, DiskSaver, global_step_from_engine
from ignite.contrib.handlers import ProgressBar


from photobooth.engines.sr_supervised import create_sr_evaluator, create_sr_trainer
from photobooth.data.datasets import DIV2K
from photobooth.models import edsr
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', type=int, required=True)
    parser.add_argument('--learning_rate', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=False, default=1e-4)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--crop_size', type=int, default=48)
    parser.add_argument('--state_dict', type=str, required=False)
    parser.add_argument('--bootstrap_state_dict')
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

    # if local_rank == 0:
    #     wandb.init(
    #         project='photobooth',
    #         config={
    #             'model': args.model,
    #             'batch_size': args.batch_size,
    #             'learning_rate': args.learning_rate,
    #             'weight_decay': args.weight_decay,
    #             'epochs': args.epochs,
    #             'decay_gamma': 0.5,
    #             'decay_every': 100,
    #             'crop_size': args.crop_size,
    #             'mixed_precision': args.mixed_precision,
    #         })

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda')

    train_ds = DIV2K(DS_ROOT, split='train', config='bicubic/x2',
                     transforms=train_tfms(args.crop_size, MEAN))
    val_ds = DIV2K(DS_ROOT, split='valid', config='bicubic/x2',
                   transforms=valid_tfms(2, 400, MEAN))

    if args.distributed:
        sampler_args = dict(num_replicas=world_size, rank=local_rank)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=8,
        sampler=(DistributedSampler(train_ds, **sampler_args) if args.distributed else None)
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=16, drop_last=False,
        sampler=(DistributedSampler(val_ds, **sampler_args, shuffle=False)
                 if args.distributed else None)
    )

    model = edsr.EDSR_baseline_2x(3, 3)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate, weight_decay=args.weight_decay,
    )
    loss_fn = torch.nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=100, gamma=0.5)

    if args.mixed_precision:
        (model, loss_fn), optimizer = amp.initialize([model, loss_fn], optimizer)

    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])

    trainer = create_sr_trainer(
        model,
        loss_fn,
        optimizer,
        device=device,
        mixed_precision=args.mixed_precision,
    )
    ProgressBar(persist=False).attach(trainer, ['loss'])
    trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        lambda _engine: scheduler.step())

    evaluator = create_sr_evaluator(
        model,
        device=device,
        mean=MEAN,
    )

    if local_rank == 0:
        checkpointer = ModelCheckpoint(
            dirname='checkpoints',
            filename_prefix='model',
            score_name='pnsr',
            score_function=lambda engine: engine.state.metrics['pnsr'],
            n_saved=5,
            global_step_transform=global_step_from_engine(trainer),
        )
        evaluator.add_event_handler(
            Events.COMPLETED, checkpointer,
            to_save={'model': model if not args.distributed else model.module})

    @trainer.on(Events.EPOCH_COMPLETED(every=10))
    def _evaluate(engine):
        state = evaluator.run(val_loader)
        if local_rank == 0:
            print("Epoch {}: {}"
                  .format(trainer.state.epoch, state.metrics))

    trainer.run(train_loader, max_epochs=args.epochs)
