# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

import re
from pathlib import Path
from typing import Optional
from itertools import chain

import jiwer
import torch
from flow_matching.path import ProbPath
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper
from torch import nn, Tensor
from transformers.tokenization_utils import PreTrainedTokenizer
from tqdm import tqdm

from .flow import SourceDistribution


class WrappedModel(ModelWrapper):
    def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
        # Note: logit's precision is important.
        return torch.softmax(self.model(x_t=x, time=t).float(), -1)


def generate_samples(
    model: nn.Module,
    step: int,
    vocab_size: int,
    tokenizer: PreTrainedTokenizer,
    rank: int,
    device: torch.device,
    path: ProbPath,
    source_distribution: SourceDistribution,
    sample_batch_size: int,
    sequence_length: int,
    sampling_steps: int,
    n_gen_iter: int = 1,
    time_epsilon: float = 0.0,
    sample_dir: Optional[Path] = None,
    dtype_categorical: torch.dtype = torch.float64,
) -> Tensor:
    wrapped_probability_denoiser = WrappedModel(model=model)

    add_token = 1 if source_distribution.masked else 0
    solver = MixtureDiscreteEulerSolver(
        model=wrapped_probability_denoiser,
        path=path,
        vocabulary_size=vocab_size + add_token,
    )

    x_init = source_distribution.sample(
        tensor_size=(sample_batch_size, sequence_length), device=device
    )

    sentences = []

    for i in range(n_gen_iter):
        sample = solver.sample(
            x_init=x_init,
            step_size=1 / sampling_steps,
            verbose=True,
            dtype_categorical=dtype_categorical,
            time_grid=torch.tensor([0.0, 1.0 - time_epsilon]),
        )

        sentences.extend(tokenizer.batch_decode(sample))

    if sample_dir is not None:
        file_name = sample_dir / f"iter_{step}" / f"sample_{rank}.txt"
        file_name.parents[0].mkdir(exist_ok=True, parents=True)

        with open(file_name, "w") as file:
            for sentence in sentences:
                file.write(f"{sentence}\n")

    return sample

@torch.no_grad()
def generate_transcription(
    model: nn.Module,
    step: int,
    vocab_size: int,
    dataloader,
    tokenizer: PreTrainedTokenizer,
    rank: int,
    device: torch.device,
    path: ProbPath,
    source_distribution: SourceDistribution,
    sample_batch_size: int,
    sequence_length: int,
    sampling_steps: int,
    time_epsilon: float = 0.0,
    sample_dir: Optional[Path] = None,
    dtype_categorical: torch.dtype = torch.float64,
) -> Tensor:

    wrapped_probability_denoiser = WrappedModel(model=model)

    add_token = 1 if source_distribution.masked else 0
    solver = MixtureDiscreteEulerSolver(
        model=wrapped_probability_denoiser,
        path=path,
        vocabulary_size=vocab_size + add_token,
    )

    hyp_trn = []
    ref_trn = []
    raw_hypotheses = []
    raw_references = []


    for batch in tqdm(dataloader, total=len(dataloader)):
        x_1 = batch['input_ids'].to(device)

        x_0 = source_distribution.sample_like(x_1, speech_noise_prob=0.0, text_noise_prob=1.0)

        sample = solver.sample(
            x_init=x_0,
            step_size=1 / sampling_steps,
            verbose=True,
            dtype_categorical=dtype_categorical,
            time_grid=torch.tensor([0.0, 1.0 - time_epsilon]),
        )


        toggle = torch.zeros_like(sample)
        toggle[sample == 2049] = 1
        toggle[sample == 2048] = -1
        text_ids = torch.cumsum(toggle, dim=1)

        text = ''.join(tokenizer.convert_ids_to_tokens(sample[text_ids==1]))
        text = text[5:] # remove first s2t token
        text += "[S2T]" # add it to end
        text = text.replace("[PAD]",'') # remove padding

        clean_hyp = text.replace("[S2T]", "").strip()
        raw_hypotheses.append(clean_hyp)

        utt_ids = chain(*batch["id"])

        hyp_trn.append(re.sub(r"\[S2T\]", lambda m : f" ({next(utt_ids)})\n", text))

        text = ''.join(tokenizer.convert_ids_to_tokens(x_1[text_ids==1]))
        text = text[5:] # remove first s2t token
        text += "[S2T]" # add it to end
        text = text.replace("[PAD]",'') # remove padding

        clean_ref = text.replace("[S2T]", "").strip()
        raw_references.append(clean_ref)

        utt_ids = chain(*batch["id"])

        ref_trn.append(re.sub(r"\[S2T\]", lambda m : f" ({next(utt_ids)})\n", text))

    if sample_dir is not None:
        hyp_file_name = sample_dir / f"iter_{step}" / "hyp.trn"
        ref_file_name = sample_dir / f"iter_{step}" / "ref.trn"

        hyp_file_name.parents[0].mkdir(exist_ok=True, parents=True)

        with open(hyp_file_name, "w") as hyp_file, open(ref_file_name, "w") as ref_file:
            for hyp, ref in zip(hyp_trn, ref_trn):
                hyp_file.write(hyp)
                ref_file.write(ref)

    cer = None
    if raw_references and raw_hypotheses:
        cer = jiwer.cer(raw_references, raw_hypotheses)

    return cer
