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

    # source soft update
    # with torch.no_grad():
    #     for source_param, param in zip(state.source_model.parameters(), state.model.parameters()):
    #         source_param.data.copy_(source_param.data * (1.0 - optim_params.source_soft_update_weight) + param.data * optim_params.source_soft_update_weight)



def step(
    state: TrainState,
    loss_fn: nn.Module,
    path: ProbPath,
    scaler: GradScaler,
    text_batch,
    audio_batch,
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

    audio_embeddings = audio_batch["neural_features"].to(device)
    audio_embeddings_text = audio_embeddings if supervised else torch.zeros_like(audio_embeddings)

    x_1 = text_batch["input_ids"].to(device)
    x_1_speech = audio_batch["input_ids"].to(device) # speech tokens

    x_0 = source_distribution.sample_like(x_1, prompt_len=4)

    t = torch.rand(x_1.shape[0], device=x_1.device) * (1.0 - time_epsilon)

    path_sample = path.sample(t=t, x_0=x_0, x_1=x_1)

    if not supervised:
        assert audio_batch["id"] != text_batch["id"]
        # generate hypothesis and sample for speech regularisation
        with torch.no_grad():
            x_1_speech_pred, _ = state.model(x_t=x_0, time=torch.zeros_like(path_sample.t), audio_embeddings=audio_embeddings)
            x_1_speech_pred = torch.softmax(x_1_speech_pred.float(), -1)
            x_1_speech_pred = categorical(x_1_speech_pred.to(dtype=torch.float64))

            path_sample_speech = path.sample(t=t, x_0=x_0, x_1=x_1_speech_pred)

    # Forward and compute loss
    ctx = nullcontext() if training else torch.no_grad()

    with ctx:
        logits, logits_speech = state.model(x_t=path_sample.x_t, time=path_sample.t, audio_embeddings=audio_embeddings_text)
        # logits = logits[:, :128, :] # cut text length to 128 as we don't expect such long sequences
        # x_1 = x_1[:, :128]

        if not supervised:
            _, logits_speech = state.model(x_t=path_sample_speech.x_t, time=path_sample_speech.t, audio_embeddings=audio_embeddings)

        if isinstance(loss_fn, nn.CrossEntropyLoss):
            loss_full = loss_fn(logits.flatten(0, 1), x_1.flatten(0, 1))
            loss_full_speech = loss_fn(logits_speech.flatten(0, 1), x_1_speech.flatten(0, 1))

        elif isinstance(loss_fn, MixturePathGeneralizedKL):
            # TODO try KLD at some point
            print("KLD loss not supported for now")
            loss_full = loss_fn(
                    logits=logits, x_1=x_1, x_t=path_sample.x_t[:, :128], t=path_sample.t
            )
            loss_full_speech = loss_fn(
                logits=logits_speech, x_1=x_1_speech, x_t=path_sample.x_t, t=path_sample.t
            )
        else:
            raise ValueError("Invalid loss function")

        loss_full = loss_full.reshape(x_1.shape)
        loss_full_speech = loss_full_speech.reshape(x_1_speech.shape)


    if state.step < 5000:
        loss_speech_weight = 0
    elif state.step >= 5000:
        loss_speech_weight = (state.step / 10000) if state.step < 10000 else 1 # TODO undo hardcoding

    loss = loss_full.mean() + loss_full_speech.mean() * loss_speech_weight

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

    loss_full = loss_full.detach()
    loss_full_speech = loss_full_speech.detach()
    return (
        loss.detach(),
        loss_full.mean(),
        loss_full[x_1 != pad_id].mean(),
        loss_full_speech.mean(),
        loss_full_speech[x_1_speech != 2048].mean(), # TODO undo hard coding
    )
