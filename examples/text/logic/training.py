# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import math
from contextlib import nullcontext
from typing import Optional

import torch
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import ProbPath
from omegaconf.dictconfig import DictConfig
from torch import nn, Tensor
from torch.cuda.amp import GradScaler
import torch.nn.functional as F

from utils.logging import TrainLogger

from .flow import SourceDistribution
from .state import TrainState


def _get_lr(lr: float, step: int, warmup: int, n_iters: int, eta_min_ratio: float):
    if step < warmup:
        # Linear warmup
        return lr * (step / warmup)
    else:
        # Cosine annealing
        total_steps = n_iters
        eta_min = eta_min_ratio * lr
        cosine_decay = 0.5 * (
            1 + math.cos(math.pi * (step - warmup) / (total_steps - warmup))
        )
        return eta_min + (lr - eta_min) * cosine_decay


def optimization_step(
    model,
    optimizer,
    state: TrainState,
    scaler: GradScaler,
    loss: Tensor,
    optim_params: DictConfig,
    logger: TrainLogger,
    accum: bool = False,
) -> None:
    scaler.scale(loss).backward()

    if accum:
        return

    scaler.unscale_(optimizer)

    lr = _get_lr(
        lr=optim_params.lr,
        step=state.step,
        warmup=optim_params.warmup,
        n_iters=optim_params.n_iters,
        eta_min_ratio=optim_params.eta_min_ratio,
    )

    # Update learning rate in optimizer
    for g in optimizer.param_groups:
        g["lr"] = lr

    if state.step % optim_params.log_lr_every == 0:
        logger.log_lr(value=lr, step=state.step)

    if optim_params.grad_clip >= 0:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=optim_params.grad_clip
        )

    scaler.step(optimizer)
    scaler.update()

    optimizer.zero_grad()


def step(
    state: TrainState,
    loss_fn: nn.Module,
    path: ProbPath,
    scaler: GradScaler,
    batch,
    device: torch.device,
    source_distribution: SourceDistribution,
    source_distribution_speech: SourceDistribution,
    logger: TrainLogger,
    training: bool,
    optim_params: Optional[DictConfig] = None,
    time_epsilon: float = 0.0,
    pad_id: int = 0,
    supervised: bool = False,
    uncond_warmup: int = 10000,
    codebook_prob: float = 0.0,
    loss_speech_weight: float = 1.0,
    time_conditioning: bool = True,
) -> Tensor:
    assert (training and (optim_params is not None)) or (not training)

    if training:
        state.train()
    else:
        state.eval()

    x_1_speech = batch["net_input"]["features"].to(device)  # speech tokens
    x_1_speech_padding = batch["net_input"]["padding_mask"].to(device)

    x_1 = batch['net_input']["random_label"].to(device)
    x_1_padding = (x_1 == pad_id).to(device)
    x_1_target = x_1.clone().detach()
    x_1_target[x_1 == 1] = -1

    # Forward and compute loss
    ctx = nullcontext() if training else torch.no_grad()

    t = torch.ones(x_1.shape[0], device=x_1.device) * (1.0 - time_epsilon)

    if training:
        # down sample km target by 2
        km_targets = batch["net_input"]["aux_target"][:,::2].to(device)
        km_input = km_targets.clone().detach()
        km_input[km_targets == -1] = 0 # TODO should really implement a pad token for km labels

    with ctx:
        logits, logits_speech = state.model(
                x_t_text=x_1,
                x_t_speech=km_input,
                time=t,
                codebook_prob=codebook_prob,
                padding_mask_text=x_1_padding,
                padding_mask_speech=x_1_speech_padding,
            )

        if isinstance(loss_fn, nn.CrossEntropyLoss):
            assert logits.dtype == torch.float
            assert x_1.dtype == torch.long

            loss_text =  loss_fn(logits.transpose(1,2), x_1_target)

            if training:
                max_t = min(logits_speech.shape[1], km_targets.shape[1])
                loss_speech =  loss_fn(logits_speech[:,:max_t].transpose(1,2), km_targets[:,:max_t])
                loss = loss_text + loss_speech

                if state.step > 1000:
                    breakpoint()
            else:
                loss = loss_text
                loss_speech = loss_text

        elif isinstance(loss_fn, MixturePathGeneralizedKL):
            # TODO set up KLD
            loss_full = loss_fn(
                logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t
            ) * ~x_1_padding

            loss_full_speech = loss_fn(
                logits=logits_speech,
                x_1=x_1_speech,
                x_t=path_sample_speech.x_t,
                t=path_sample.t,
            ) * ~x_1_speech_padding
        else:
            raise ValueError("Invalid loss function")




    # Optimization step (only if training=true)
    if training:
        optimization_step(
            model=state.model,
            optimizer=state.optimizer,
            state=state,
            loss=loss,
            scaler=scaler,
            optim_params=optim_params,
            logger=logger,
        )

    return (
        loss.detach(),
        loss_text.detach(),
        loss_speech.detach(),
    )
