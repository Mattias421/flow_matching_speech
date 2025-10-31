# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, Iterable, Tuple
from pathlib import Path

from datasets import DatasetDict, load_dataset, concatenate_datasets, Audio
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
from transformers import MimiModel, AutoFeatureExtractor

from data.tokenizer import wt_detokenizer, train_tokenizer
from data.utils import cycle_loader, StatefulDistributedSampler
import logging

logger = logging.getLogger(__name__)


def _get_hf_dataset(
    name: str,
    mode: str,
    cache_dir: str = None,
    block_size: int = 1024,
    num_proc: int = 8,
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
    elif name == "librispeech":
        data = load_dataset("openslr/librispeech_asr", cache_dir=cache_dir)
        if mode == "train":
            data = concatenate_datasets(
                [
                    data["train.clean.100"],
                    data["train.clean.360"],
                    data["train.other.500"],
                ]
            )
        elif mode == "validation":
            data = concatenate_datasets(
                [data["validation.clean"], data["validation.other"]]
            )
        else:
            # test clean or other
            data = data[mode]
        data = data.cast_column(
            "audio", 
            Audio(sampling_rate=24000) # mimi expects 24khz
        )
    elif name == "librispeech_dummy":
        data = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    else:
        data = load_dataset(name, cache_dir=cache_dir)[mode]

    def _apply_detokenizer(detokenizer):
        def detok(text):
            for i, t in enumerate(text, 0):
                text[i] = detokenizer(t)
            return text

        return detok

    logger.info("loading tokenizer")
    if "librispeech" in name:
        if mode == "train" and not Path("outputs/tokenizer-librispeech.json").exists():
            logger.info("training new tokenizer")
            train_tokenizer(data, "outputs/tokenizer-librispeech.json")

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file="outputs/tokenizer-librispeech.json"
        )
    else:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # added tokens
    EOS = 2048
    S2T = 2049

    # load the model + feature extractor (for pre-processing the audio)
    model = MimiModel.from_pretrained("kyutai/mimi").to('cuda')
    feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

    def preprocess_and_tokenize(example: Dict):
        text = example["text"]
        audio = example["audio"]
        audio = [a['array'] for a in audio]
        lens = [a.shape[0] for a in audio]
        max_len = max(lens)
        lens = [l / max_len for l in lens]

        inputs = feature_extractor(raw_audio=audio, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt").to('cuda')

        audio_tokens = model.encode(inputs["input_values"]).audio_codes[:,0,:].cpu().tolist() # 0th codebook is semantic

        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        text_tokens = tokenizer(text, return_attention_mask=False)

        input_ids = []

        for text, audio, audio_len in zip(text_tokens['input_ids'], audio_tokens, lens):
            seq = []
            seq += audio[:int(audio_len*len(audio))]
            seq.append(S2T)
            seq += text
            seq.append(EOS)

            input_ids.append(seq)

        return {"input_ids":input_ids}

    logger.info("Tokenizing data")
    tokenized_dataset = data.map(
        preprocess_and_tokenize,
        batched=True,
        batch_size=8,
        num_proc=1,
        load_from_cache_file=True,
    )

    model = model.cpu()

    keep_columns = ["input_ids", "id"]

    if name == "fineweb-edu" or "librispeech" in name:
        features = tokenized_dataset.features.keys()
        for k in features:
            if k not in keep_columns:
                tokenized_dataset = tokenized_dataset.remove_columns(k)
    else:
        tokenized_dataset = tokenized_dataset.remove_columns("text")

    def group_texts(examples: Dict):

        result = {'id':[], 'input_ids':[]}
        current_chunk = {'id':[], 'input_ids':[]}

        for utt_id, input_ids in zip(examples['id'], examples['input_ids']):
            if len(current_chunk['input_ids']) + len(input_ids) > block_size:
                # block is ready to be added to result
                current_chunk['input_ids'] += [EOS] * (block_size - len(current_chunk['input_ids'])) # pad remainder with EOS

                result['input_ids'].append(current_chunk['input_ids'])
                result['id'].append(current_chunk['id'])

                current_chunk = {'id':[], 'input_ids':[]}

            current_chunk['id'].append(utt_id)
            current_chunk['input_ids'] += input_ids

        current_chunk['input_ids'] += [EOS] * (block_size - len(current_chunk['input_ids'])) # pad remainder with EOS
        result['input_ids'].append(current_chunk['input_ids'])
        result['id'].append(current_chunk['id'])

        return result

    logger.info("Chunking data")
    chunked_dataset = tokenized_dataset.map(
        group_texts, batched=True, num_proc=num_proc, load_from_cache_file=True
    )
    chunked_dataset = chunked_dataset.with_format("torch")

    return chunked_dataset


@dataclass
class Dataset:
    dataset: DatasetDict = field(metadata={"help": "Huggingface dataset"})
    sampler: StatefulDistributedSampler = field(
        metadata={"help": "Stateful sampler for `dataset`"}
    )


@dataclass
class DataState:
    train: Dataset = field(metadata={"help": "Train dataset"})
    test: Dataset = field(metadata={"help": "Test dataset"})


def _get_dataset(
    name: str,
    mode: str,
    cache_dir: str,
    block_size: int,
    num_proc: int,
    batch_size: int,
    ngpus: int,
) -> Dataset:
    assert batch_size % ngpus == 0, (
        f"{mode} batch size must be divisible by number of gpus."
    )

    dataset = _get_hf_dataset(
        name=name,
        mode=mode,
        cache_dir=cache_dir,
        block_size=block_size,
        num_proc=num_proc,
    )

    sampler = StatefulDistributedSampler(dataset=dataset)

    return Dataset(dataset=dataset, sampler=sampler)


def get_data_state(config: OmegaConf) -> DataState:
    train = _get_dataset(
        name=config.data.train,
        mode="train",
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        num_proc=config.data.num_workers,
        batch_size=config.training.batch_size,
        ngpus=config.compute.ngpus,
    )
    test = _get_dataset(
        name=config.data.valid,
        mode="validation",
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        num_proc=config.data.num_workers,
        batch_size=config.eval.batch_size,
        ngpus=config.compute.ngpus,
    )

    return DataState(train=train, test=test)

def collate_fn(batch):
    utt_ids = [item['id'] for item in batch]

    input_ids = torch.stack([item['input_ids'] for item in batch])

    return {"id":utt_ids, "input_ids":input_ids}


def get_data_loaders(
    config: OmegaConf,
    data_state: DataState,
) -> Tuple[Iterable, Iterable]:
    train_loader = cycle_loader(
        DataLoader(
            data_state.train.dataset,
            batch_size=config.training.batch_size // config.compute.ngpus,
            collate_fn=collate_fn,
            sampler=data_state.train.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=(data_state.train.sampler is None),
            persistent_workers=True,
        )
    )

    valid_loader = cycle_loader(
        DataLoader(
            data_state.test.dataset,
            batch_size=config.eval.batch_size // config.compute.ngpus,
            collate_fn=collate_fn,
            sampler=data_state.test.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=(data_state.test.sampler is None),
        )
    )

    valid_loader_no_cycle = DataLoader(
        data_state.test.dataset,
        batch_size=config.eval.batch_size // config.compute.ngpus,
        collate_fn=collate_fn,
        sampler=data_state.test.sampler,
        num_workers=config.data.num_workers,
        pin_memory=True,
        shuffle=(data_state.test.sampler is None),
    )

    return iter(train_loader), iter(valid_loader), valid_loader_no_cycle
