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
) -> Tensor:
    add_token = 1 if source_distribution.masked else 0

    hyp_trn = []
    ref_trn = []
    raw_hypotheses = []
    raw_references = []

    for text_batch, audio_batch in tqdm(dataloader):
        assert text_batch['id'] == audio_batch['id']

        x_0 = audio_batch["input_ids"].to(device)

        class WrappedASRModel(ModelWrapper):
            def forward(self, x: Tensor, t: Tensor, **extras) -> Tensor:
                # Note: logit's precision is important.
                return torch.softmax(self.model(x_t=x, time=t, x_source=x_0).float(), -1)

        wrapped_probability_denoiser = WrappedASRModel(model)

        solver = MixtureDiscreteEulerSolver(
            model=wrapped_probability_denoiser,
            path=path,
            vocabulary_size=vocab_size + add_token,
        )

        x_init = torch.randint_like(x_0, pad_id)
        x_init=torch.full_like(x_0, 0)

        time_grid = torch.linspace(0.0,1.0-time_epsilon, sampling_steps)
        sample = solver.sample(
            # x_init=x_0,
            x_init = x_init,
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
            text = "".join(tokenizer.convert_ids_to_tokens(hyp_text_ids))
            text = text.replace("[PAD]", "")  # remove padding
            clean_hyp = text.replace("[EOS]", "").strip()
            raw_hypotheses.append(clean_hyp)
            trn_hyp = clean_hyp + f" ({utt_id})\n"
            hyp_trn.append(trn_hyp)

            text = "".join(tokenizer.convert_ids_to_tokens(ref_text_ids))
            text = text.replace("[PAD]", "")  # remove padding
            clean_ref = text.replace("[EOS]", "").strip()
            raw_references.append(clean_ref)
            trn_ref = clean_ref + f" ({utt_id})\n"
            ref_trn.append(trn_ref)

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
    else:
        print("something went wrong with CER calculation")

    return cer
