# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from abc import ABC
from typing import Optional, Tuple

import torch
from flow_matching.loss import MixturePathGeneralizedKL
from flow_matching.path import MixtureDiscreteProbPath, ProbPath
from flow_matching.path.scheduler import PolynomialConvexScheduler
from torch import Tensor
from torch.nn.modules.loss import _Loss


class SourceDistribution(ABC):
    def __init__(
        self,
    ) -> None: ...

    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor: ...

    def sample_like(self, tensor_like: Tensor) -> Tensor: ...


class MaskedSourceDistribution(SourceDistribution):
    def __init__(self, mask_token: int) -> None:
        self.mask_token = mask_token

    @property
    def masked(self) -> bool:
        return True

    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
        return torch.zeros(tensor_size, device=device).fill_(self.mask_token).long()

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        return torch.zeros_like(tensor_like).fill_(self.mask_token).long()


class UniformSourceDistribution(SourceDistribution):
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size

    @property
    def masked(self) -> bool:
        return False

    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
        return torch.randint(size=tensor_size, high=self.vocab_size, device=device)

    def sample_like(self, tensor_like: Tensor) -> Tensor:
        return torch.randint_like(tensor_like, high=self.vocab_size)

class ASRSourceDistribution(SourceDistribution):
    def __init__(self) -> None:
        # TODO hardcoded vocab
        self.vocab_size = 2048
        self.eos_token = 2048
        self.s2t_token = 2049

    @property
    def masked(self) -> bool:
        return False

    def sample(self, tensor_size: Tuple[int, ...], device: torch.device) -> Tensor:
        print("WARNING: ASRSourceDistribution isn't really supposed to be sampled without x_1")
        return torch.randint(size=tensor_size, high=self.vocab_size, device=device)

    def sample_like(self, x_1: Tensor, speech_noise_prob: float, text_noise_prob: float) -> Tensor:
        # preserve speech tokens
        toggle = torch.zeros_like(x_1)
        toggle[x_1 == self.s2t_token] = 1
        toggle[x_1 == self.eos_token] = -1

        text_token_pos = torch.cumsum(toggle, dim=1)
        text_token_pos = (text_token_pos == 1) & (x_1 != self.s2t_token) # remove S2T from prediction

        speech_token_pos = (text_token_pos != 1) & (x_1 != self.s2t_token) & (x_1 != self.eos_token)

        prob_noise = torch.rand(x_1.shape).to(x_1.device)
        speech_noise_mask = speech_token_pos & (prob_noise < speech_noise_prob)

        text_noise_mask = text_token_pos & (prob_noise < text_noise_prob)

        noise_mask = text_noise_mask | speech_noise_mask # true if should be noised

        uniform_noise = torch.randint_like(x_1, high=self.vocab_size)
        x_0 = x_1 * ~noise_mask + uniform_noise * noise_mask

        return x_0


def get_path(scheduler_type: str, exponent: Optional[float] = None) -> ProbPath:
    if scheduler_type == "polynomial":
        scheduler = PolynomialConvexScheduler(n=exponent)
    else:
        raise ValueError(f"{scheduler_type} is not supported")

    return MixtureDiscreteProbPath(scheduler=scheduler)


def get_source_distribution(
    source_distribution: str, vocab_size: int
) -> SourceDistribution:
    if source_distribution == "mask":
        return MaskedSourceDistribution(mask_token=vocab_size)
    elif source_distribution == "uniform":
        return UniformSourceDistribution(vocab_size=vocab_size)
    elif source_distribution == "asr":
        return ASRSourceDistribution()
    else:
        raise ValueError(f"{source_distribution} is not supported")


def get_loss_function(loss_function: str, path: Optional[ProbPath] = None) -> _Loss:
    if loss_function == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    elif loss_function == "generalized_kl":
        assert path is not None

        return MixturePathGeneralizedKL(path=path)
    else:
        raise ValueError(f"{loss_function} is not supported")
