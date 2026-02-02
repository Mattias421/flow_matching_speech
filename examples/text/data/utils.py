# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# This implementation is adapted from https://github.com/pytorch/data/blob/main/torchdata/stateful_dataloader/sampler.py#L132
# which is released under BSD-3 license

import itertools
from typing import Any, Dict, Optional
import random

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Sampler

def collate_fn_unpaired(batch, max_length, mode="text"):
    input_ids = [item["input_ids"] for item in batch]

    sizes = [len(inputs) for inputs in input_ids]
    length = max(sizes)
    length = min(length, max_length)

    fill_val = 2

    collated_inp_ids = torch.full((len(batch), length), fill_val, dtype=torch.long)
    padding_mask = torch.BoolTensor(len(input_ids), length).fill_(False)

    for i, (size, input_id) in enumerate(zip(sizes, input_ids)):
        size = min(length, size)
        collated_inp_ids[i, :size] = input_id[:size]
        padding_mask[i, size:] = True

    assert collated_inp_ids.shape[0] == len(batch)
    collated = {"input_ids": collated_inp_ids, "padding_mask": padding_mask}

    return collated


def cycle_loader(dataloader: DataLoader, sampler: Sampler = None) -> Tensor:
    while 1:
        if sampler is not None:
            sampler.set_epoch(np.random.randint(0, 100000))
        for data in dataloader:
            yield data


class StatefulSampler(torch.utils.data.Sampler):
    """
    From: https://github.com/pytorch/data/blob/main/torchdata/stateful_dataloader/sampler.py#L132
    """

    _YIELDED = "yielded"

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.yielded = 0
        self.next_yielded = None
        self.shuffle = shuffle
        self.dataset = dataset
        self.seed = seed

        # Initialize indices
        self.indices = list(range(len(self.dataset)))
        if not self.shuffle:
            # If not shuffling, they stay 0, 1, 2...
            pass
        else:
            # Initial shuffle with provided seed
            random.seed(self.seed)
            random.shuffle(self.indices)

    def __iter__(self):
        self.yielded = 0
        if self.next_yielded is not None:
            self.yielded = self.next_yielded
            self.next_yielded = None
        elif self.shuffle:
            random.shuffle(self.indices)

        it = iter(self.indices)
        for idx in itertools.islice(it, self.yielded, None):
            self.yielded += 1
            yield idx

    def state_dict(self) -> Dict[str, Any]:
        return {self._YIELDED: self.yielded}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if self._YIELDED not in state_dict:
            raise ValueError("Invalid state_dict")
        if state_dict[self._YIELDED] < 0:
            raise ValueError("Cannot load state_dict with negative yielded value")
        self.next_yielded = state_dict[self._YIELDED]
