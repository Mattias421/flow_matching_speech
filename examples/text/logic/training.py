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
    loss = loss
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

    if not supervised:
        assert audio_batch['id'] != text_batch['id']
        lengths = (audio_batch['input_ids'] != pad_id).sum(dim=1)
        _, sorted_indices = torch.sort(lengths, descending=True)
        audio_batch['input_ids'] = audio_batch['input_ids'][sorted_indices]

        lengths = (text_batch['input_ids'] != pad_id).sum(dim=1)
        _, sorted_indices = torch.sort(lengths, descending=True)
        text_batch['input_ids'] = text_batch['input_ids'][sorted_indices]
    else:
        assert audio_batch['id'] == text_batch['id']

    audio = audio_batch["input_ids"].to(device)
    text = text_batch["input_ids"].to(device)


    def dfm_loss(x_tgt, mode, source_dist=None):
        source_mode = 'text' if mode == 'audio' else 'audio'

        source_t = 1 if source_mode == 'audio' else 0

        with torch.no_grad():
            if source_dist == 'cycle':
                source_model = state.model_asr if source_mode == 'text' else state.model_tts
                t_array = torch.ones(x_tgt.shape[0], device=x_tgt.device) * source_t
                model_pred = source_model(x_t=x_tgt, time=t_array, x_source=x_tgt).float()
                p_src = torch.softmax(model_pred, -1)
                x_src = categorical(p_src.to(dtype=torch.float64))
            elif source_dist == 'mask':
                x_src = torch.full_like(audio, 0)
            elif source_dist == 'uniform':
                x_src = torch.randint_like(audio, pad_id)
            elif source_dist == 'data':
                x_src = text if source_mode == "text" else audio

            t = torch.rand(x_tgt.shape[0], device=x_tgt.device) * (1.0 - time_epsilon)

            if mode == "audio":
                path_sample = path.sample(t=t, x_0=x_tgt, x_1=x_src) # swap boundaries for speech
            else:
                path_sample = path.sample(t=t, x_0=x_src, x_1=x_tgt)


        # Forward and compute loss
        ctx = nullcontext() if training else torch.no_grad()

        with ctx:
            target_model = state.model_asr if mode == 'text' else state.model_tts
            logits = target_model(x_t=path_sample.x_t, time=path_sample.t, x_source=audio)

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

    loss_text = dfm_loss(text, 'text', 'mask')
    # loss_text_cycle = dfm_loss(text, 'text', cycle=True)
    # loss_speech = dfm_loss(audio, 'audio')
    # loss_speech_cycle = dfm_loss(audio, 'audio', cycle=True)
    #
    # cosine_decay = 0.5 * (
    #     1 + math.cos(math.pi * (state.step) / (2000))
    # )
    #
    # loss = (loss_text.mean() + loss_speech.mean()) * cosine_decay + (loss_text_cycle.mean() + loss_speech_cycle.mean())

    loss_speech = loss_text
    loss = loss_text.mean()

    # Optimization step (only if training=true)
    if training:
        optimization_step(
            model=state.model_asr,
            optimizer=state.optimizer_asr,
            state=state,
            loss=loss_text.mean(),
            scaler=scaler,
            optim_params=optim_params,
            logger=logger,
        )

        # optimization_step(
        #     model=state.model_tts,
        #     optimizer=state.optimizer_tts,
        #     state=state,
        #     loss=loss_speech.mean(),
        #     scaler=scaler,
        #     optim_params=optim_params,
        #     logger=logger,
        # )

    return (
        loss.detach(),
        loss_speech.mean(),
        loss_text.mean(),
        loss_speech[audio != pad_id].mean(),
        loss_text[text != pad_id].mean(),
    )
