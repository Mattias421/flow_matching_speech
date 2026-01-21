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


def _get_extracted_features_dataset(
    path: str,
    split: str,
    max_length: int,
    seed: int,
    supervised: bool,
    labels: str = None,
):
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    tokenizer = SpeechT5Tokenizer.from_pretrained("microsoft/speecht5_tts")

    def get_tokens(text):
        text = processor.tokenizer.basic_normalize(text)
        text_tokens = tokenizer(text, return_attention_mask=False)
        return torch.tensor(text_tokens['input_ids'])


    dataset = ExtractedFeaturesDataset(
        path=path,
        split=split,
        max_length=max_length,
        labels=labels,
        tokenizer=get_tokens,
    )
    sampler = (
        StatefulDistributedSampler(dataset=dataset, seed=seed)
        if not supervised
        else None
    )
    dataset.sampler = sampler
    return dataset


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

    import os 
    from fairseq.data import (
    Dictionary,
    data_utils,
    StripTokenDataset,
)

    dict_path = os.path.join(config.data.text_data, "dict.txt")
    target_dictionary = Dictionary.load(dict_path)

    audio = _get_extracted_features_dataset(
        path=config.data.features_path,
        split="train",
        labels="wrd",
        max_length=config.model.length,
        supervised=config.data.supervised,
        seed=0,
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

    test_audio = _get_extracted_features_dataset(
        path=config.data.features_path,
        labels="wrd",
        split="valid",
        max_length=config.model.length,
        supervised=config.data.supervised,
        seed=0,
    )

    # TODO implement valid text dataset
    valid = RandomInputDataset(
        test_audio,
        text_dataset,
        ["random_label"],
        add_to_input=True,
        pad_idx=target_dictionary.pad(),
    )

    return DataState(train=train, valid=valid)




def get_data_loaders(
    config: OmegaConf,
    data_state: DataState,
) -> Tuple[Iterable, Iterable]:
    audio_loader = cycle_loader(
        DataLoader(
            data_state.audio,
            batch_size=(config.training.batch_size // 2) // config.compute.ngpus,
            collate_fn=data_state.audio.collater,
            sampler=data_state.audio.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=config.data.num_workers > 0,
            drop_last=True,
        )
    )

    text_loader = cycle_loader(
        DataLoader(
            data_state.text.dataset,
            batch_size=(config.training.batch_size // 2) // config.compute.ngpus,
            collate_fn=lambda x: collate_fn_unpaired(x, config.model.length),
            sampler=data_state.text.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=config.data.num_workers > 0,
            drop_last=True,
        )
    )

    valid_loader = cycle_loader(
        DataLoader(
            data_state.test_text.dataset,
            batch_size=config.eval.batch_size // config.compute.ngpus,
            collate_fn=lambda x: collate_fn_unpaired(x, config.model.length),
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=False,
        )
    )
    valid_loader_text = DataLoader(
        data_state.test_text.dataset,
        batch_size=config.eval.batch_size // config.compute.ngpus,
        collate_fn=lambda x: collate_fn_unpaired(x, config.model.length),
        num_workers=config.data.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    valid_loader_audio = DataLoader(
        data_state.test_audio,
        batch_size=config.eval.batch_size // config.compute.ngpus,
        collate_fn=data_state.test_audio.collater,
        num_workers=config.data.num_workers,
        pin_memory=True,
        shuffle=False,
    )

    return (
        iter(audio_loader),
        iter(text_loader),
        iter(valid_loader),
        valid_loader_text,
        valid_loader_audio,
    )
