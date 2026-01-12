# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional

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


@torch.no_grad()
def generate_transcription(
    model: nn.Module,
    step: int,
    vocab_size: int,
    dataloader,
    tokenizer: PreTrainedTokenizer,
    normalize,
    rank: int,
    device: torch.device,
    path: ProbPath,
    source_distribution: SourceDistribution,
    sample_batch_size: int,
    sequence_length: int,
    sampling_steps: int,
    pad_id: int,
    time_epsilon: float = 0.0,
    sample_dir: Optional[Path] = None,
    dtype_categorical: torch.dtype = torch.float64,
    cfg_strength: float = 1.0,
) -> Tensor:
    add_token = 1 if source_distribution.masked else 0

    model = model.eval()

    hyp_trn = []
    ref_trn = []
    raw_hypotheses = []
    raw_references = []

    for text_batch, audio_batch in tqdm(dataloader):
        assert text_batch['id'] == audio_batch['id']

        audio_embeddings = audio_batch["neural_features"].to(device)
        audio_cache = model.build_audio_cache(audio_embeddings)

        x_1 = text_batch["input_ids"]
        x_0 = source_distribution.sample_like(x_1).to(device)

        class WrappedASRModel(ModelWrapper):
            def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
                # Note: logit's precision is important.
                probs = torch.softmax(self.model(x_t=x, time=t, audio_embeddings=audio_embeddings, **audio_cache, cfg_strength=cfg_strength).float(), -1)
                return probs

        wrapped_probability_denoiser = WrappedASRModel(model)

        solver = MixtureDiscreteEulerSolver(
            model=wrapped_probability_denoiser,
            path=path,
            vocabulary_size=vocab_size + add_token,
        )

        time_grid = torch.linspace(0.0,1.0-time_epsilon, sampling_steps)
        sample = solver.sample(
            x_init=x_0,
            step_size=None,
            verbose=False,
            dtype_categorical=dtype_categorical,
            time_grid=time_grid,
            return_intermediates=True,
        )

        text_sample = sample[-1]
        text_ref = text_batch["input_ids"]

        for hyp_text_ids, ref_text_ids, utt_id in zip(
            text_sample, text_ref, text_batch["id"]
        ):
            text = tokenizer.decode(hyp_text_ids, skip_special_tokens=True)
            text = normalize(text)
            raw_hypotheses.append(text)
            trn_hyp = text + f" ({utt_id})\n"
            hyp_trn.append(trn_hyp)

            text = tokenizer.decode(ref_text_ids, skip_special_tokens=True)
            text = normalize(text)
            raw_references.append(text)
            trn_ref = text + f" ({utt_id})\n"
            ref_trn.append(trn_ref)

    if sample_dir is not None:
        hyp_file_name = sample_dir / f"iter_{step}" / "hyp.trn"
        ref_file_name = sample_dir / f"iter_{step}" / "ref.trn"

        hyp_file_name.parents[0].mkdir(exist_ok=True, parents=True)

        with open(hyp_file_name, "w", encoding='utf-8') as hyp_file, open(ref_file_name, "w", encoding='utf-8') as ref_file:
            for hyp, ref in zip(hyp_trn, ref_trn):
                hyp_file.write(hyp)
                ref_file.write(ref)

    for hyp in hyp_trn[:10]:
        print(hyp)

    cer = None
    if raw_references and raw_hypotheses:
        cer = jiwer.cer(raw_references, raw_hypotheses)
    else:
        print("something went wrong with CER calculation")

    model = model.train()

    return cer
