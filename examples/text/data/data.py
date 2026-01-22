# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

from datasets import DatasetDict, load_dataset, load_dataset_builder
from omegaconf import OmegaConf
import os 
from fairseq.data import (
Dictionary,
data_utils,
StripTokenDataset,
)

import torch
from torch.utils.data import DataLoader
from transformers import SpeechT5Tokenizer
from transformers import WhisperProcessor

from data.tokenizer import wt_detokenizer
from data.utils import cycle_loader, StatefulDistributedSampler, collate_fn_unpaired
from data.extracted_features_dataset import ExtractedFeaturesDataset
from data.random_input_dataset import RandomInputDataset
import logging

logger = logging.getLogger(__name__)

@dataclass
class Dataset:
    dataset: DatasetDict = field(metadata={"help": "Huggingface dataset"})
    sampler: StatefulDistributedSampler = field(
        metadata={"help": "Stateful sampler for `dataset`"}
    )


@dataclass
class DataState:
    train: RandomInputDataset = field(metadata={"help": "train dataset"})
    valid: RandomInputDataset = field(metadata={"help": "valid dataset"})


def get_data_state(config: OmegaConf) -> DataState:

    dict_path = os.path.join(config.data.text_data, "dict.txt")
    target_dictionary = Dictionary.load(dict_path)

    audio = ExtractedFeaturesDataset(
        path=config.data.features_path,
        split="train",
        max_length=config.model.length,
        aux_target_postfix='km',
    )


    text_dataset = data_utils.load_indexed_dataset(
        os.path.join(config.data.text_data, "train"), target_dictionary
    )

    text_dataset = StripTokenDataset(text_dataset, target_dictionary.eos())

    train = RandomInputDataset(
        audio,
        text_dataset,
        ["random_label"],
        add_to_input=True,
        pad_idx=target_dictionary.pad(),
    )

    train.sampler = StatefulDistributedSampler(dataset=train, seed=0)
    train.target_dictionary = target_dictionary

    test_audio = ExtractedFeaturesDataset(
        path=config.data.features_path,
        split="valid",
        max_length=config.model.length,
        aux_target_postfix='km',
    )

    # TODO implement valid text dataset of some sort
    valid = RandomInputDataset(
        test_audio,
        text_dataset,
        ["random_label"],
        add_to_input=True,
        pad_idx=target_dictionary.pad(),
    )
    valid.sampler = StatefulDistributedSampler(dataset=valid, seed=0)
    valid.target_dictionary = target_dictionary

    return DataState(train=train, valid=valid)




def get_data_loaders(
    config: OmegaConf,
    data_state: DataState,
) -> Tuple[Iterable, Iterable]:
    train = cycle_loader(
        DataLoader(
            data_state.train,
            batch_size=(config.training.batch_size // 2) // config.compute.ngpus,
            collate_fn=data_state.train.collater,
            sampler=data_state.train.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=config.data.num_workers > 0,
            drop_last=True,
        )
    )

    valid = cycle_loader(
        DataLoader(
            data_state.valid,
            batch_size=(config.training.batch_size // 2) // config.compute.ngpus,
            collate_fn=data_state.train.collater,
            sampler=data_state.valid.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=config.data.num_workers > 0,
            drop_last=True,
        )
    )

    return (
        iter(train),
        iter(valid),
        valid,
    )
