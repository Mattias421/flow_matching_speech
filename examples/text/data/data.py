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
from data.utils import cycle_loader, StatefulDistributedSampler
from data.extracted_features_dataset import ExtractedFeaturesDataset
import logging

logger = logging.getLogger(__name__)


def _get_hf_dataset(
    name: str,
    mode: str,
    split: str,
    codec_name: str,
    cache_dir: str = None,
    block_size: int = 1024,
    num_proc: int = 8,
    text_tokenizer: str = "speecht5",
) -> DatasetDict:
    detokenizer = None

    logger.info(f"preparing {name}-{mode}")

    if name == "wikitext103":
        data = load_dataset(
            "wikitext", name="wikitext-103-raw-v1", cache_dir=cache_dir
        )[mode]
        detokenizer = wt_detokenizer
    elif name == "fineweb-edu":
        data = load_dataset(
            "HuggingFaceFW/fineweb-edu", name="CC-MAIN-2024-10", cache_dir=cache_dir
        )[mode]
    elif name == "librispeech_lm":
        builder = load_dataset_builder(
            "openslr/librispeech_lm", cache_dir=cache_dir, trust_remote_code=True
        )
        builder.download_and_prepare()
        data = builder.as_dataset(split="train")
        data = data.train_test_split(train_size=0.01, seed=42)[
            "train"
        ]  # trim because 80m rows is far too many
        data = data.filter(lambda example: len(example["text"]) <= 510)
    elif name == "librispeech_dummy":
        data = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
    else:
        data = load_dataset(name, cache_dir=cache_dir)[mode]

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                text[i] = detokenizer(t)
            return text

        return detok

    logger.info("loading tokenizer")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    if text_tokenizer == "speecht5":
        tokenizer = SpeechT5Tokenizer.from_pretrained("microsoft/speecht5_tts")
    else:
        tokenizer = processor.tokenizer

    def preprocess_and_tokenize_text(example: Dict):
        text = example["text"]

        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        text = [processor.tokenizer.basic_normalize(t) for t in text]
        text_tokens = tokenizer(text, return_attention_mask=False)

        return text_tokens

    logger.info("Tokenizing data")

    tokenized_dataset = data.map(
        preprocess_and_tokenize_text,
        batched=True,
        batch_size=1000,
        num_proc=8,
        load_from_cache_file=True,
    )

    tokenized_dataset = tokenized_dataset.with_format("torch")

    return tokenized_dataset


def _get_extracted_features_dataset(
    path: str,
    split: str,
    max_length: int,
    seed: int,
    supervised: bool,
    labels: str = None,
):
    dataset = ExtractedFeaturesDataset(
        path=path,
        split=split,
        max_length=max_length,
        labels=labels,
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
    audio: ExtractedFeaturesDataset = field(metadata={"help": "Audio dataset"})
    text: Dataset = field(metadata={"help": "text dataset"})
    test_text: Dataset = field(metadata={"help": "Test dataset"})
    test_audio: ExtractedFeaturesDataset = field(metadata={"help": "Test dataset"})


def _get_dataset(
    name: str,
    mode: str,
    split: str,
    cache_dir: str,
    block_size: int,
    num_proc: int,
    batch_size: int,
    ngpus: int,
    codec_name: str,
    supervised: bool = False,
    seed: int = 0,
) -> Dataset:
    assert batch_size % ngpus == 0, (
        f"{mode} batch size must be divisible by number of gpus."
    )

    dataset = _get_hf_dataset(
        name=name,
        mode=mode,
        split=split,
        cache_dir=cache_dir,
        block_size=block_size,
        num_proc=num_proc,
        codec_name=codec_name,
    )

    sampler = (
        StatefulDistributedSampler(dataset=dataset, seed=seed)
        if not supervised
        else None
    )

    return Dataset(dataset=dataset, sampler=sampler)


def get_data_state(config: OmegaConf) -> DataState:
    text = _get_dataset(
        name=config.data.text,
        mode="text",
        split="train",
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        num_proc=config.data.num_workers,
        batch_size=config.training.batch_size,
        ngpus=config.compute.ngpus,
        codec_name=config.data.codec_name,
        supervised=config.data.supervised,
        seed=0,
    )

    audio = _get_extracted_features_dataset(
        path=config.data.features_path,
        split="train",
        max_length=config.model.length,
        supervised=config.data.supervised,
        seed=0,
    )

    test_text = _get_dataset(
        name=config.data.valid,
        mode="text",
        split="validation",
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        num_proc=config.data.num_workers,
        batch_size=config.eval.batch_size,
        ngpus=config.compute.ngpus,
        codec_name=config.data.codec_name,
        supervised=True,
    )

    test_audio = _get_extracted_features_dataset(
        path=config.data.features_path,
        labels="wrd",
        split="valid",
        max_length=config.model.length,
        supervised=config.data.supervised,
        seed=0,
    )

    return DataState(audio=audio, text=text, test_text=test_text, test_audio=test_audio)


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
