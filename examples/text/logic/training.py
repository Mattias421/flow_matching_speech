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
from flow_matching.utils import categorical
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
    accum: bool = False,
) -> None:
    loss = loss
    scaler.scale(loss).backward()

    if accum:
        return

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

    # source soft update
    with torch.no_grad():
        for source_param, param in zip(state.source_model.parameters(), state.model.parameters()):
            source_param.data.copy_(source_param.data * (1.0 - optim_params.source_soft_update_weight) + param.data * optim_params.source_soft_update_weight)



def step(
    state: TrainState,
    loss_fn: nn.Module,
    path: ProbPath,
    scaler: GradScaler,
    text,
    audio,
    device: torch.device,
    source_distribution: SourceDistribution,
    logger: TrainLogger,
    training: bool,
    optim_params: Optional[DictConfig] = None,
    time_epsilon: float = 0.0,
    pad_id: int = 0,
    supervised: bool = False,
) -> Tensor:
    assert (training and (optim_params is not None)) or (not training)

    if training:
        state.train()
    else:
        state.eval()

    audio = audio["input_ids"].to(device)
    text = text["input_ids"].to(device)

    def dfm_loss(x_tgt, mode):
        source_mode = 'text' if mode == 'audio' else 'audio'

        source_t = 1 if source_mode == 'audio' else 0

        with torch.no_grad():
            if supervised:
                t_array = torch.ones(x_tgt.shape[0], device=x_tgt.device) * source_t
                model_pred = state.source_model(x_t=x_tgt, time=t_array, x_source=x_tgt, mode=source_mode).float()
                p_src = torch.softmax(model_pred, -1)
                x_src = categorical(p_src.to(dtype=torch.float64))
            else:
                assert audio['id'] == text['id']
                x_src = text if source_mode == "text" else audio

            t = torch.rand(x_tgt.shape[0], device=x_tgt.device) * (1.0 - time_epsilon)


            if mode == "audio":
                path_sample = path.sample(t=t, x_0=x_tgt, x_1=x_src) # swap boundaries for speech
            else:
                path_sample = path.sample(t=t, x_0=x_src, x_1=x_tgt)


        # Forward and compute loss
        ctx = nullcontext() if training else torch.no_grad()

        with ctx:
            logits = state.model(x_t=path_sample.x_t, time=path_sample.t, x_source=x_src, mode=mode)

            if isinstance(loss_fn, nn.CrossEntropyLoss):
                loss_full = loss_fn(logits.flatten(0, 1), x_tgt.flatten(0, 1))

            elif isinstance(loss_fn, MixturePathGeneralizedKL):
                loss_full = loss_fn(
                    logits=logits, x_1=x_tgt, x_t=path_sample.x_t, t=path_sample.t
                )
            else:
                raise ValueError("Invalid loss function")

            loss_full = loss_full.reshape(x_tgt.shape)
        return loss_full

    loss_text = dfm_loss(text, 'text')
    loss_speech = dfm_loss(audio, 'audio')

    loss = loss_text.mean() + optim_params.loss_speech_weight * loss_speech.mean()

    # Optimization step (only if training=true)
    if training:
        optimization_step(
            state=state,
            loss=loss,
            scaler=scaler,
            optim_params=optim_params,
            logger=logger,
        )

    return (
        loss.detach(),
        loss_speech.mean(),
        loss_text.mean(),
        loss_speech[audio != pad_id].mean(),
        loss_text[text != pad_id].mean(),
    )
