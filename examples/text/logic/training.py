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
    text_batch,
    audio_batch,
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

    x_1 = text_batch["input_ids"].to(device)
    x_1_speech = audio_batch["input_ids"].to(device)  # speech tokens

    x_1_padding = text_batch["padding_mask"].to(device)
    x_1_speech_padding = audio_batch["padding_mask"].to(device)

    x_0 = source_distribution.sample_like(x_1)
    x_0_speech = source_distribution_speech.sample_like(x_1_speech)

    t = torch.rand(x_1.shape[0], device=x_1.device) * (1.0 - time_epsilon)

    assert x_1.shape[0] == x_1_speech.shape[0], f"{x_1.shape[0]} should equal {x_1_speech.shape[0]}"

    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)
    path_sample_speech = path.sample(t=t, x_0=x_0_speech, x_1=x_1_speech)

    path_sample.x_t *= ~x_1_padding
    path_sample_speech.x_t *= ~x_1_speech_padding


    if not supervised:
        assert audio_batch["id"] != text_batch["id"]

    # Forward and compute loss
    ctx = nullcontext() if training else torch.no_grad()

    with ctx:
        if time_conditioning:
            time=path_sample.t,
        else:
            time=torch.zeros_like(path_sample.t)

        logits, logits_speech = state.model(
            x_t_text=path_sample.x_t,
            x_t_speech=path_sample_speech.x_t,
            time=time,
            codebook_prob=codebook_prob,
            padding_mask_text=x_1_padding,
            padding_mask_speech=x_1_speech_padding,
        )

        if isinstance(loss_fn, nn.CrossEntropyLoss):
            assert logits.dtype == logits_speech.dtype == torch.float
            assert x_1.dtype == x_1_speech.dtype == torch.long

            loss_full = loss_fn(logits.flatten(0, 1), x_1.flatten(0, 1)) * ~x_1_padding.flatten(0,1)
            loss_full_speech = loss_fn(
                logits_speech.flatten(0, 1), x_1_speech.flatten(0, 1)
            ) * ~x_1_speech_padding.flatten(0,1)

        elif isinstance(loss_fn, MixturePathGeneralizedKL):
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

        loss_full = loss_full.reshape(x_1.shape)
        loss_full_speech = loss_full_speech.reshape(x_1_speech.shape)

    loss_text = loss_full.sum() / (~x_1_padding).sum()
    loss_speech = loss_full_speech.sum() / (~x_1_speech_padding).sum()
    loss = loss_text + loss_speech * loss_speech_weight


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
