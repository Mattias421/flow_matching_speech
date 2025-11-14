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

from torch.utils.data import DataLoader
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
    state: TrainState,
    scaler: GradScaler,
    loss: Tensor,
    optim_params: DictConfig,
    logger: TrainLogger,
) -> None:
    scaler.scale(loss).backward()
    scaler.unscale_(state.optimizer)

    lr = _get_lr(
        lr=optim_params.lr,
        step=state.step,
        warmup=optim_params.warmup,
        n_iters=optim_params.n_iters,
        eta_min_ratio=optim_params.eta_min_ratio,
    )

    # Update learning rate in optimizer
    for g in state.optimizer.param_groups:
        g["lr"] = lr

    if state.step % optim_params.log_lr_every == 0:
        logger.log_lr(value=lr, step=state.step)

    if optim_params.grad_clip >= 0:
        torch.nn.utils.clip_grad_norm_(
            state.model.parameters(), max_norm=optim_params.grad_clip
        )

    scaler.step(state.optimizer)
    scaler.update()

    state.optimizer.zero_grad()


def step(
    state: TrainState,
    loss_fn: nn.Module,
    path: ProbPath,
    scaler: GradScaler,
    iterator: DataLoader,
    device: torch.device,
    source_distribution: SourceDistribution,
    logger: TrainLogger,
    training: bool,
    partial_noise_prob: float,
    optim_params: Optional[DictConfig] = None,
    time_epsilon: float = 0.0,
    unsupervised_prob: bool = False,
    partial_loss_weight: float = 1,
) -> Tensor:
    assert (training and (optim_params is not None)) or (not training)

    if training:
        state.train()
    else:
        state.eval()

    x_1 = next(iterator)["input_ids"].to(device)

    # Sample from path
    if torch.rand(1) < unsupervised_prob:
        with torch.no_grad():
            block_size = x_1.shape[-1]
            if state.step % 2 == 0:
                # predict text label for speech
                x_0_speech = source_distribution.sample_like(
                    x_1, speech_noise_prob=0, text_noise_prob=1.0
                )
                logits = state.model(
                    x_t=x_0_speech, time=torch.zeros(x_1.shape[0], device=x_1.device)
                )
                x_1_text = logits.argmax(dim=-1)
                mask = torch.arange(block_size, device=x_1.device)[None, :] < (
                    block_size // 2
                )
                x_1 = x_1 * mask + x_1_text * ~mask

            else:
                # predict speech label for text
                x_0_text = source_distribution.sample_like(
                    x_1, speech_noise_prob=0, text_noise_prob=1.0
                )
                logits = state.model(
                    x_t=x_0_text, time=torch.zeros(x_1.shape[0], device=x_1.device)
                )
                x_1_speech = logits.argmax(dim=-1)
                mask = torch.arange(block_size, device=x_1.device)[None, :] > (
                    block_size // 2
                )
                x_1 = x_1 * mask + x_1_speech * ~mask

    with torch.no_grad():
        if state.step % 2 == 0:
            x_0 = source_distribution.sample_like(
                x_1, speech_noise_prob=1.0, text_noise_prob=partial_noise_prob
            )
        else:
            x_0 = source_distribution.sample_like(
                x_1, speech_noise_prob=partial_noise_prob, text_noise_prob=1.0
            )

        t = torch.rand(x_1.shape[0], device=x_1.device) * (1.0 - time_epsilon)
        path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

    # Forward and compute loss
    ctx = nullcontext() if training else torch.no_grad()

    with ctx:
        logits = state.model(x_t=path_sample.x_t, time=path_sample.t)

        if isinstance(loss_fn, nn.CrossEntropyLoss):
            loss_full = loss_fn(logits.flatten(0, 1), x_1.flatten(0, 1))

        elif isinstance(loss_fn, MixturePathGeneralizedKL):
            loss_full = loss_fn(
                logits=logits, x_1=x_1, x_t=path_sample.x_t, t=path_sample.t
            )
        else:
            raise ValueError("Invalid loss function")

        loss_full = loss_full.reshape(x_1.shape)
        block_size = loss_full.shape[-1]

        loss_weight = torch.ones(block_size, device=device)
        if state.step % 2 == 0:
            loss_weight[(block_size // 2 + 1) :] = partial_loss_weight
        else:
            loss_weight[: (block_size // 2)] = partial_loss_weight

        loss_weighted = loss_full * loss_weight[None, :]
        loss = loss_weighted.mean()

    # Optimization step (only if training=true)
    if training:
        optimization_step(
            state=state,
            loss=loss,
            scaler=scaler,
            optim_params=optim_params,
            logger=logger,
        )

    with torch.no_grad():
        loss_speech = loss_full[:, : (loss_full.shape[-1] // 2)].mean()
        loss_text = loss_full[:, (loss_full.shape[-1] // 2 + 1) :].mean()

        loss_speech_no_pad = loss_full[:, : (loss_full.shape[-1] // 2)][
            x_1[:, : (loss_full.shape[-1] // 2)] != 2050
        ].mean()  # TODO remove hard coded pad index
        loss_text_no_pad = loss_full[:, (loss_full.shape[-1] // 2 + 1) :][
            x_1[:, (loss_full.shape[-1] // 2 + 1) :] != 2050
        ].mean()  # TODO remove hard coded pad index

        loss_no_pad = loss_full[x_1 != 2050].mean()  # TODO remove hard coded pad index

    return (
        loss.detach(),
        loss_no_pad,
        loss_speech,
        loss_text,
        loss_speech_no_pad,
        loss_text_no_pad,
    )
