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
from sklearn.cluster import MiniBatchKMeans
import numpy as np

import torch
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast, PreTrainedTokenizerFast
from transformers import MimiModel, AutoFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration

from data.tokenizer import wt_detokenizer, train_tokenizer
from data.utils import cycle_loader, StatefulDistributedSampler
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
        if split == 'train':
            data = load_dataset("openslr/librispeech_asr", cache_dir=cache_dir)
            data = concatenate_datasets(
                [
                    data["train.clean.100"],
                    data["train.clean.360"],
                ]
            )
        elif split == "validation":
            data = load_dataset("openslr/librispeech_asr", cache_dir=cache_dir, split="validation.clean")
        else:
            # test clean or other
            data = data[mode]
        # if codec_name == 'mimi':
        #     data = data.cast_column(
        #         "audio",
        #         Audio(sampling_rate=24000),  # mimi expects 24khz
        #     )

    elif name == "librispeech_lm":
        builder = load_dataset_builder('openslr/librispeech_lm', cache_dir=cache_dir, trust_remote_code=True)
        builder.download_and_prepare()
        data = builder.as_dataset(split='train')
        data = data.train_test_split(train_size=0.01, seed=42)['train'] # trim because 80m rows is far too many
        data = data.filter(lambda example : len(example['text']) <= 510)
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
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    del model.model.decoder
    model = model.model.encoder.to("cuda").eval()

    def preprocess_and_tokenize_text(example: Dict):
        text = example["text"]

        if detokenizer is not None:
            text = _apply_detokenizer(detokenizer)(text)

        normalized_text = [processor.tokenizer.normalize(t) for t in text]
        text_tokens = processor.tokenizer(normalized_text, return_attention_mask=False)

        return text_tokens

    def encode_audio(example):
        audio = example["audio"]["array"]
        length = len(audio) / (16000 * 30)
        mel_features = processor(audio, sampling_rate=16000).input_features
        input_features = torch.tensor(mel_features).to("cuda")
        with torch.no_grad():
            neural_features = model(input_features)
        input_features = input_features.cpu()
        neural_features = neural_features.last_hidden_state.cpu()

        mel_features = mel_features[0,:,:int(length * mel_features.shape[-1])]
        return {"mel_features":mel_features, "neural_features":neural_features}


    logger.info("Tokenizing data")
    if mode == 'audio':
        tokenized_dataset = data.map(
            encode_audio,
            batched=False,
            batch_size=1,
            num_proc=1,
            load_from_cache_file=True,
        )

        k_means_data = np.concatenate([np.array(example["mel_features"]).T for example in tokenized_dataset])

        kmeans = MiniBatchKMeans(
                n_clusters=64,
                init="k-means++",
                batch_size=10000,
                tol=0.0,
                max_no_improvement=100,
                n_init=20,
                reassignment_ratio=0.0,
                )

        print("training and labelling kmeans model")
        kmeans_labels = kmeans.fit_predict(k_means_data)
        idx = 0

        def label_feature(example):
            global idx
            length = len(example["mel_features"])
            labels = kmeans_labels[idx:length]
            idx += length

            return {"kmeans_labels":labels}

        tokenized_dataset = tokenized_dataset.map(
            label_feature,
            batched=False,
            batch_size=1,
            num_proc=1,
            load_from_cache_file=True,
        )

            
    elif mode == 'text':
        tokenized_dataset = data.map(
            preprocess_and_tokenize_text,
            batched=True,
            batch_size=1000,
            num_proc=8,
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
    audio: Dataset = field(metadata={"help": "Audio dataset"})
    text: Dataset = field(metadata={"help": "text dataset"})
    test_text: Dataset = field(metadata={"help": "Test dataset"})
    test_audio: Dataset = field(metadata={"help": "Test dataset"})


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

    sampler = StatefulDistributedSampler(dataset=dataset, seed=seed) if not supervised else None

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

    audio = _get_dataset(
        name=config.data.audio,
        mode="audio",
        split="train",
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        num_proc=config.data.num_workers,
        batch_size=config.training.batch_size,
        ngpus=config.compute.ngpus,
        codec_name=config.data.codec_name,
        supervised=config.data.supervised,
        seed=1,
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

    test_audio = _get_dataset(
        name=config.data.valid,
        mode="audio",
        split="validation",
        cache_dir=config.data.cache_dir,
        block_size=config.model.length,
        num_proc=config.data.num_workers,
        batch_size=config.eval.batch_size,
        ngpus=config.compute.ngpus,
        codec_name=config.data.codec_name,
        supervised=True,
    )


    return DataState(audio=audio, text=text, test_text=test_text, test_audio=test_audio)


def collate_fn(batch):
    utt_ids = [item["id"] for item in batch]

    input_ids = torch.stack([item["input_ids"] for item in batch])

    return {"id": utt_ids, "input_ids": input_ids}

def collate_fn_unpaired(batch):

    utt_ids = [item["id"] for item in batch]
    input_ids = torch.stack([item["input_ids"] for item in batch])

    return {"input_ids": input_ids, "id":utt_ids}


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
            shuffle=False,
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
            shuffle=False,
            persistent_workers=True,
        )
    )

    valid_loader = cycle_loader(
        DataLoader(
            data_state.test_text.dataset,
            batch_size=config.eval.batch_size // config.compute.ngpus,
            collate_fn=collate_fn,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=False
        )
    )
    valid_loader_text = DataLoader(
            data_state.test_text.dataset,
            batch_size=config.eval.batch_size // config.compute.ngpus,
            collate_fn=collate_fn,
            num_workers=config.data.num_workers,
            pin_memory=True,
            shuffle=False,
        )

    valid_loader_audio = DataLoader(
                data_state.test_audio.dataset,
                batch_size=config.eval.batch_size // config.compute.ngpus,
                collate_fn=collate_fn,
                num_workers=config.data.num_workers,
                pin_memory=True,
                shuffle=False,
            )

    return iter(audio_loader), iter(text_loader), iter(valid_loader), valid_loader_text, valid_loader_audio


