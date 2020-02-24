import torch
from ignite.engine import Engine, _prepare_batch
from ignite.metrics import RunningAverage, MeanSquaredError, MeanAbsoluteError
from photobooth.metrics import PNSR

try:
    from apex import amp
except ImportError:
    from warnings import warn
    warn("The apex library is not installed. "
         "The mixed precision training is not possible.")


__all__ = ['create_sr_trainer', 'create_sr_evaluator']


def create_sr_trainer(
    model,
    loss_fn,
    optimizer,
    mixed_precision=False,
    device=None,
    non_blocking=True,
):
    def _update_model(engine, batch):
        model.train()
        optimizer.zero_grad()
        x, y = _prepare_batch(batch, device=device, non_blocking=non_blocking)

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        if mixed_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()
        return loss

    engine = Engine(_update_model)
    RunningAverage(output_transform=lambda x: x).attach(engine, 'loss')
    RunningAverage(PNSR(1.)).attach(engine, 'pnsr')

    return engine


def create_sr_evaluator(
    model,
    device=None,
    non_blocking=True,
    denormalize=True,
    mean=None,
):
    # transfer mean to the device and reshape it so
    # that is is broadcastable to the BCHW format
    mean = mean.to(device).reshape(1, -1, 1, 1)

    def denorm_fn(x):
        return torch.clamp(x + mean, min=0., max=1.)

    def _evaluate_model(engine, batch):
        model.eval()
        x, y = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        with torch.no_grad():
            y_pred, y = model(x)
        if denormalize:
            y_pred = denorm_fn(y_pred)
            y = denorm_fn(y)
        return y_pred, y

    engine = Engine(_evaluate_model)
    MeanAbsoluteError().attach(engine, 'l1')
    MeanSquaredError().attach(engine, 'l2')
    PNSR(max_value=1.0).attach(engine, 'pnsr')

    return engine
