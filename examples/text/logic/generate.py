# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from typing import Optional

import jiwer
import torch
import kenlm
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
    audioloader,
    rank: int,
    device: torch.device,
    path: ProbPath,
    source_distribution: SourceDistribution,
    sample_batch_size: int,
    sequence_length: int,
    sampling_steps: int,
    pad_id: int,
    target_dictionary,
    inference_block: int,
    kenlm_path: str,
    time_epsilon: float = 0.0,
    sample_dir: Optional[Path] = None,
    dtype_categorical: torch.dtype = torch.float64,
    time_conditioning: bool = True,
) -> Tensor:
    add_token = 1 if source_distribution.masked else 0

    model = model.eval()
    kenlm_model = kenlm.Model(kenlm_path)
    total_log_score = 0
    total_units = 1e-9

    hyp_trn = []
    ref_trn = []
    raw_hypotheses = []

    raw_references = []
    utt_ids = []

    for audio_batch in tqdm(audioloader):
        # assert text_batch["id"] == audio_batch["id"]

        refs = [' '.join(phn_seq) for phn_seq in audio_batch["target"]]
        raw_references.extend(refs)

        utt_ids.extend(audio_batch["id"])

        speech = audio_batch["net_input"]["features"].to(device)

        padding_mask_speech = audio_batch["net_input"]["padding_mask"].to(device)

        t = torch.ones(speech.shape[0], device=speech.device) * (1.0 - time_epsilon)
        probs = model(
                x_t_speech=speech,
                padding_mask_speech=padding_mask_speech,
                time=t,
                inference_block=inference_block,
            )

        greedy = probs.argmax(-1)
        greedy[padding_mask_speech] = pad_id

        for g in greedy:
            g = g[g >= target_dictionary.nspecial]
            pred_units_arr = g
            # ctc eval
            pred_units_arr = pred_units_arr[pred_units_arr != 0]  # remove pad
            pred_units_arr = pred_units_arr[pred_units_arr != vocab_size - 1]  # remove silence
            pred_units_arr = pred_units_arr.unique_consecutive()
            pred_units_arr = pred_units_arr.tolist()
            hyp = target_dictionary.string(pred_units_arr)
            total_log_score += kenlm_model.score(hyp)
            total_units += len(hyp.split()) + 1
            raw_hypotheses.append(hyp)

    for ref, hyp, utt_id in zip(
        raw_references, raw_hypotheses, utt_ids
    ):
        trn_hyp = hyp + f" ({utt_id})\n"
        hyp_trn.append(trn_hyp)

        trn_ref = ref + f" ({utt_id})\n"
        ref_trn.append(trn_ref)

    if sample_dir is not None:
        hyp_file_name = sample_dir / f"iter_{step}" / "hyp.trn"
        ref_file_name = sample_dir / f"iter_{step}" / "ref.trn"

        hyp_file_name.parents[0].mkdir(exist_ok=True, parents=True)

        with (
            open(hyp_file_name, "w", encoding="utf-8") as hyp_file,
            open(ref_file_name, "w", encoding="utf-8") as ref_file,
        ):
            for hyp, ref in zip(hyp_trn, ref_trn):
                hyp_file.write(hyp)
                ref_file.write(ref)

    for hyp, ref in zip(hyp_trn[:10], ref_trn[:10]):
        print(hyp)
        print(ref)

    uer = 10.0
    if raw_references and raw_hypotheses:
        uer = jiwer.wer(raw_references, raw_hypotheses)
    else:
        print("something went wrong with UER calculation")

    model = model.train()
    ppl = 10.0 ** (-total_log_score / total_units)
    return uer, ppl
