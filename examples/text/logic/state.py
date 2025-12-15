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
        model_tts: nn.Module,
        model_asr: nn.Module,
        optimizer_tts: Optimizer,
        optimizer_asr: Optimizer,
        step: int,
        data_state: DataState,
    ):
        self._model_tts = model_tts
        self._optimizer_tts = optimizer_tts
        self._model_asr = model_asr
        self._optimizer_asr = optimizer_asr
        self._step = step
        self._data_state = data_state

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, value: int) -> None:
        self._step = value

    @property
    def optimizer_tts(self) -> Optimizer:
        return self._optimizer_tts

    @property
    def optimizer_asr(self) -> Optimizer:
        return self._optimizer_asr

    @property
    def model_tts(self) -> nn.Module:
        return self._model_tts

    @property
    def model_asr(self) -> nn.Module:
        return self._model_asr

    @property
    def data_state(self) -> DataState:
        return self._data_state

    def compile_model(self) -> None:
        self._model_tts = torch.compile(self._model_tts)
        self._model_asr = torch.compile(self._model_asr)


    def restore_checkpoint(
        self, ckpt_dir: Path, device: torch.device, rank: int
    ) -> None:
        if ckpt_dir.exists():
            loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=True)

            self.optimizer_tts.load_state_dict(loaded_state["optimizer_tts"])
            self.model_tts.module.load_state_dict(loaded_state["model_tts"])
            self.optimizer_asr.load_state_dict(loaded_state["optimizer_asr"])
            self.model_asr.module.load_state_dict(loaded_state["model_asr"])
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
            "optimizer_tts": self.optimizer_tts.state_dict(),
            "model_tts": self.model_tts.module.state_dict(),
            "optimizer_asr": self.optimizer_asr.state_dict(),
            "model_asr": self.model_asr.module.state_dict(),
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
        self._model_tts.train(mode=training)
        self._model_asr.train(mode=training)
