# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path

import torch
from data import DataState

from torch import nn
from torch.optim import Optimizer


class TrainState:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        step: int,
        data_state: DataState,
    ):
        self._model = model
        self._optimizer = optimizer
        self._step = step
        self._data_state = data_state

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        self._step = value

    @property
    def optimizer(self) -> Optimizer:
        return self._optimizer

    @property
    def model(self) -> nn.Module:
        return self._model


    @property
    def data_state(self) -> DataState:
        return self._data_state

    def compile_model(self) -> None:
        self._model = torch.compile(self._model)


    def restore_checkpoint(
        self, ckpt_dir: Path, device: torch.device, rank: int
    ) -> None:
        if ckpt_dir.exists():
            loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=True)

            self.optimizer.load_state_dict(loaded_state["optimizer"])
            self.model.module.load_state_dict(loaded_state["model"])
            self.step = loaded_state["step"]

            if loaded_state["test_text_sampler"]:
                self._data_state.test_text.sampler.load_state_dict(loaded_state["test_text_sampler"])

            if loaded_state["audio_sampler"]:
                self._data_state.audio.sampler.load_state_dict(
                    loaded_state["audio_sampler"]
                )
            if loaded_state["text_sampler"]:
                self._data_state.text.sampler.load_state_dict(
                    loaded_state["text_sampler"]
                )
        else:
            ckpt_dir.parent.mkdir(exist_ok=True, parents=True)

            if rank == 0:
                logging.warning(
                    f"No checkpoint found at {ckpt_dir}. Returned the same state as input"
                )

    def save_checkpoint(self, ckpt_dir: str, rank: int) -> None:
        saved_state = {
            "optimizer": self.optimizer.state_dict(),
            "model": self.model.module.state_dict(),
            "step": self.step,
        }

        if self._data_state.audio.sampler:
            saved_state["audio_sampler"] = self._data_state.audio.sampler.state_dict()
        else:
            saved_state["audio_sampler"] = None

        if self._data_state.text.sampler:
            saved_state["text_sampler"] = self._data_state.text.sampler.state_dict()
        else:
            saved_state["text_sampler"] = None

        if self._data_state.test_text.sampler:
            saved_state["test_text_sampler"] = self._data_state.test_text.sampler.state_dict()
        else:
            saved_state["test_text_sampler"] = None

        if rank == 0:
            torch.save(saved_state, ckpt_dir)

    def eval(self) -> None:
        self.train(training=False)

    def train(self, training: bool = True) -> None:
        self._model.train(mode=training)
