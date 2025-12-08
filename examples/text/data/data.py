# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple
from pathlib import Path

from datasets import DatasetDict, load_dataset, concatenate_datasets, Audio, load_dataset_builder
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
    codec_name: str,
    cache_dir: str = None,
    block_size: int = 1024,
    num_proc: int = 8,
    train_percent: int = 100,
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
        if mode == "audio" or mode == "text":
            data = concatenate_datasets(
                [
                    data["train.clean.100"],
                    data["train.clean.360"],
                    data["train.other.500"],
                ]
            )
        elif mode == "train":
            data = load_dataset("openslr/librispeech_asr", cache_dir=cache_dir, split=f"train.clean.100[:{train_percent}%]")
        elif mode == "validation":
            data = concatenate_datasets(
                [data["validation.clean"], data["validation.other"]]
            )
        else:
            # test clean or other
            data = data[mode]
        if codec_name == 'mimi':
            data = data.cast_column(
                "audio",
                Audio(sampling_rate=24000),  # mimi expects 24khz
            )

    elif name == "librispeech_lm":
        builder = load_dataset_builder('openslr/librispeech_lm', cache_dir=cache_dir, trust_remote_code=True)
        builder.download_and_prepare()
        data = builder.as_dataset(split='train')
        data = data.train_test_split(train_size=0.01, seed=42)['train'] # trim because 80m rows is far too many
        data = data.filter(lambda example : len(example['text']) <= 510)
    elif name == "librispeech_dummy":

        if mode == 'train':
            data = load_dataset(
                    "hf-internal-testing/librispeech_asr_dummy", "clean", split=f"validation[:{train_percent}%]"
            )
        else:
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


    # load the model + feature extractor (for pre-processing the audio)
    if codec_name == 'mimi':
        model = MimiModel.from_pretrained("kyutai/mimi").to("cuda")
        feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

        n_vocab = 2048

        def get_audio_tokens(audio):
            inputs = feature_extractor(
                raw_audio=audio,
                sampling_rate=feature_extractor.sampling_rate,
                return_tensors="pt",
            ).to("cuda")

            audio_tokens = (
                model.encode(inputs["input_values"]).audio_codes[:, 0, :].cpu().tolist()
            )  # 0th codebook is semantic
            return audio_tokens

    elif codec_name == 'focalcodec':
        model = torch.hub.load(
            repo_or_dir="lucadellalib/focalcodec",
            model="focalcodec",
            config="lucadellalib/focalcodec_12_5hz",
            force_reload=True,  # Fetch the latest FocalCodec version from Torch Hub
        )
        model.eval().requires_grad_(False).to("cuda")

        n_vocab = 8192

        def get_audio_tokens(audio):
            batch_size = len(audio)
            lens = [a.shape[0] for a in audio]
            max_len = max(lens)

            audio_batch = torch.zeros((batch_size, max_len)).to("cuda")

            for i, (a, le) in enumerate(zip(audio, lens)):
                audio_batch[i,:le] = torch.tensor(a)

            return model.sig_to_toks(audio_batch)

    logger.info("loading tokenizer")
    if "librispeech" in name:
        if mode == "text" and not Path(f"outputs/tokenizer-librispeech-{n_vocab}.json").exists():
            logger.info("training new tokenizer")
            train_tokenizer(data, f"outputs/tokenizer-librispeech-{n_vocab}.json", n_vocab)

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=f"outputs/tokenizer-librispeech-{n_vocab}.json"
        )

    else:
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

    # added tokens
    EOS = n_vocab
    S2T = n_vocab + 1
    PAD = n_vocab + 2


    def preprocess_and_tokenize_text(example: Dict):
        text = example["text"]

        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        text_tokens = tokenizer(text, return_attention_mask=False)

        input_ids = []

        for text in text_tokens["input_ids"]:
            seq = []
            seq += [PAD] * ((block_size // 2))
            seq.append(S2T)
            seq += text
            seq.append(EOS)
            assert (block_size) > len(seq), (
                f"Sequence length {len(seq)} greater than block size, consider increasing block_size"
            )
            seq += [PAD] * (block_size - len(seq))

            input_ids.append(seq)

        return {"input_ids": input_ids}

    def preprocess_and_tokenize_audio(example: Dict):
            audio = example["audio"]
            audio = [a["array"] for a in audio]
            lens = [a.shape[0] for a in audio]
            max_len = max(lens)
            lens = [l / max_len for l in lens]

            audio_tokens = get_audio_tokens(audio)

            input_ids = []

            for audio, audio_len in zip(audio_tokens, lens):
                seq = []
                seq += audio[: int(audio_len * len(audio))]
                seq.append(EOS)
                assert (block_size // 2) > len(seq), (
                    "Audio sequence length greater than half block size, consider increasing block_size"
                )
                seq += [PAD] * ((block_size // 2) - len(seq))
                seq.append(S2T)
                assert (block_size) > len(seq), (
                    "Sequence length greater than block size, consider increasing block_size"
                )
                seq += [PAD] * (block_size - len(seq))

                input_ids.append(seq)

            return {"input_ids": input_ids}



    def preprocess_and_tokenize(example: Dict):
        text = example["text"]
        audio = example["audio"]
        audio = [a["array"] for a in audio]
        lens = [a.shape[0] for a in audio]
        max_len = max(lens)
        lens = [l / max_len for l in lens]

        audio_tokens = get_audio_tokens(audio)


        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        text_tokens = tokenizer(text, return_attention_mask=False)

        input_ids = []

        for text, audio, audio_len in zip(text_tokens["input_ids"], audio_tokens, lens):
            seq = []
            seq += audio[: int(audio_len * len(audio))]
            seq.append(EOS)
            assert (block_size // 2) > len(seq), (
                "Audio sequence length greater than half block size, consider increasing block_size"
            )
            seq += [PAD] * ((block_size // 2) - len(seq))
            seq.append(S2T)
            seq += text
            seq.append(EOS)
            assert (block_size) > len(seq), (
                "Sequence length greater than block size, consider increasing block_size"
            )
            seq += [PAD] * (block_size - len(seq))

            input_ids.append(seq)

        return {"input_ids": input_ids}

    logger.info("Tokenizing data")
    if mode == 'audio':
        tokenized_dataset = data.map(
            preprocess_and_tokenize_audio,
            batched=True,
            batch_size=8,
            num_proc=1,
            load_from_cache_file=True,
        )
    elif mode == 'text':
        tokenized_dataset = data.map(
            preprocess_and_tokenize_text,
            batched=True,
            batch_size=8,
            num_proc=1,
            load_from_cache_file=True,
        )
    else:
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

    tokenized_dataset = tokenized_dataset.with_format("torch")

    return tokenized_dataset


@dataclass
class Dataset:
    dataset: DatasetDict = field(metadata={"help": "Huggingface dataset"})
    sampler: StatefulDistributedSampler = field(
        metadata={"help": "Stateful sampler for `dataset`"}
    )


@dataclass
class DataState:
    train: Dataset = field(metadata={"help": "train dataset"})
    audio: Dataset = field(metadata={"help": "Audio dataset"})
    text: Dataset = field(metadata={"help": "text dataset"})
    test: Dataset = field(metadata={"help": "Test dataset"})


def _get_dataset(
    name: str,
    mode: str,
    cache_dir: str,
    block_size: int,
    num_proc: int,
    batch_size: int,
    ngpus: int,
    codec_name: str,
    train_percent: int = 100,
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
        codec_name=codec_name,
        train_percent=train_percent,
    )

    sampler = StatefulDistributedSampler(dataset=dataset)

    return Dataset(dataset=dataset, sampler=sampler)

def get_data_state(config: OmegaConf) -> DataState:
    text = _get_dataset(
        name=config.data.text,
        mode="text",
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        num_proc=config.data.num_workers,
        batch_size=config.training.batch_size,
        ngpus=config.compute.ngpus,
        codec_name=config.data.codec_name,
    )

    audio = _get_dataset(
        name=config.data.audio,
        mode="audio",
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        num_proc=config.data.num_workers,
        batch_size=config.training.batch_size,
        ngpus=config.compute.ngpus,
        codec_name=config.data.codec_name,
    )

    train = _get_dataset(
        name=config.data.train,
        mode="train",
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        num_proc=config.data.num_workers,
        batch_size=config.training.batch_size,
        ngpus=config.compute.ngpus,
        codec_name=config.data.codec_name,
        train_percent=config.data.train_percent,
    )

    test = _get_dataset(
        name=config.data.valid,
        mode="validation",
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        num_proc=config.data.num_workers,
        batch_size=config.eval.batch_size,
        ngpus=config.compute.ngpus,
        codec_name=config.data.codec_name,
    )

    return DataState(train=train, audio=audio, text=text, test=test)


def collate_fn(batch):
    utt_ids = [item["id"] for item in batch]

    input_ids = torch.stack([item["input_ids"] for item in batch])

    return {"id": utt_ids, "input_ids": input_ids}

def collate_fn_unpaired(batch):

    input_ids = torch.stack([item["input_ids"] for item in batch])

    return {"input_ids": input_ids}


def get_data_loaders(
    config: OmegaConf,
    data_state: DataState,
) -> Tuple[Iterable, Iterable]:
    audio_loader = cycle_loader(
        DataLoader(
            data_state.audio.dataset,
            batch_size=(config.training.batch_size // 2) // config.compute.ngpus,
            collate_fn=collate_fn_unpaired,
            sampler=data_state.audio.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=(data_state.audio.sampler is None),
            persistent_workers=True,
        )
    )

    text_loader = cycle_loader(
        DataLoader(
            data_state.text.dataset,
            batch_size=(config.training.batch_size // 2) // config.compute.ngpus,
            collate_fn=collate_fn_unpaired,
            sampler=data_state.text.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=(data_state.text.sampler is None),
            persistent_workers=True,
        )
    )

    train_loader = cycle_loader(
        DataLoader(
            data_state.train.dataset,
            batch_size=config.training.batch_size // config.compute.ngpus,
            collate_fn=collate_fn,
            sampler=data_state.train.sampler,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=(data_state.test.sampler is None),
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

    return iter(train_loader), iter(audio_loader), iter(text_loader), iter(valid_loader), valid_loader_no_cycle


