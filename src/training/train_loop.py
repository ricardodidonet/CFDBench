# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from args import Args
import gc
import logging
import math
from typing import Iterable
from tqdm import tqdm

import torch
from flow_matching.path import CondOTProbPath
from torchmetrics.aggregation import MeanMetric
from training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)

PRINT_FREQUENCY = 50


def skewed_timestep_sample(num_samples: int, device: torch.device) -> torch.Tensor:
    """
    Sample timesteps with skewed distribution to focus on difficult regions.
    This puts more emphasis on t near 0 and 1.
    """
    P_mean = -1.2
    P_std = 1.2
    rnd_normal = torch.randn((num_samples,), device=device)
    sigma = (rnd_normal * P_std + P_mean).exp()
    time = 1 / (1 + sigma)
    time = torch.clip(time, min=0.0001, max=1.0)
    return time


def train_epoch(
        model: torch.nn.Module,
        dataloader: Iterable, 
        optimizer: torch.optim.Optimizer,
        lr_schedule: torch.optim.lr_scheduler.LRScheduler,
        device: torch.device,
        epoch: int,
        loss_scaler: NativeScalerWithGradNormCount,
        args: Args
    ):

    """Train for one epoch."""
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device)
    epoch_loss = MeanMetric().to(device)

    accum_iter = args.accum_iter
    path = CondOTProbPath()
    prog_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for data_iter_step, batch in enumerate(prog_bar):
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()

        # Move data to device
        x_target = batch['x_target'].to(device)  # (B, 2, H, W)
        x_prev_1 = batch['x_prev_1'].to(device)
        x_prev_2 = batch['x_prev_2'].to(device)
        case_params = batch['case_params'].to(device)

        batch_size = x_target.shape[0]

        # Sample random timesteps for flow matching
        if args.use_skewed_timesteps:
            t = skewed_timestep_sample(batch_size, device)
        else:
            t = torch.rand(batch_size, device=device)

        # Sample noise for x_0 (Gaussian noise)
        x_0 = torch.randn_like(x_target)

        # Sample from the probability path: x_t = (1-t)*x_0 + t*x_1
        # where x_1 is the target state
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_target)
        x_t = path_sample.x_t  # Noisy interpolation
        u_t = path_sample.dx_t  # Target velocity: x_1 - x_0

        # Classifier-free guidance: randomly drop case_params
        # Keep x_prev_1, x_prev_2 (essential for temporal prediction)
        # Drop case_params (optional geometric/physical conditioning)
        if torch.rand(1).item() < args.conditioning_drop_prob:
            extra = {
                'x_prev_1': x_prev_1,
                'x_prev_2': x_prev_2,
                'case_params': None  # Dropped for unconditional training
            }
        else:
            extra = {
                'x_prev_1': x_prev_1,
                'x_prev_2': x_prev_2,
                'case_params': case_params
            }

        # Flow matching loss with AMP
        with torch.amp.autocast():
            predicted_velocity = model(x_t, t, extra)
            loss = torch.nn.functional.mse_loss(predicted_velocity, u_t)

        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)

        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")

        loss /= accum_iter

        # Loss scaler applies the optimizer when update_grad is set to true.
        # Otherwise just updates the internal gradient scales
        apply_update = (data_iter_step + 1) % accum_iter == 0

        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=apply_update
            )

        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(dataloader)}]: loss = {batch_loss.compute()}, lr = {lr}"
            )

    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}