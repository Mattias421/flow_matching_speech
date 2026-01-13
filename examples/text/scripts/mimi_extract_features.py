#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import os.path as osp
import tqdm
import torch
import torch.nn.functional as F
import torchaudio
from shutil import copyfile
from transformers import MimiModel, AutoFeatureExtractor

from npy_append_array import NpyAppendArray

import soundfile as sf


def get_parser():
    parser = argparse.ArgumentParser(
        description="compute kmeans codebook from kaldi-computed feats"
    )
    # fmt: off
    parser.add_argument('data', help='location of tsv files')
    parser.add_argument('--split', help='which split to read', required=True)
    parser.add_argument('--save-dir', help='where to save the output', required=True)
    # parser.add_argument('--layer', type=int, default=14, help='which layer to use')
    # fmt: on


    return parser


class MimiFeatureReader(object):
    def __init__(self):

        self.model = MimiModel.from_pretrained("kyutai/mimi").to("cuda").eval()
        self.feature_extractor = AutoFeatureExtractor.from_pretrained("kyutai/mimi")

        self.resample = torchaudio.transforms.Resample(16000,24000)

    def read_audio(self, fname):
        """Load an audio file and return PCM along with the sample rate"""
        wav, sr = sf.read(fname)
        assert sr == 16e3

        return wav

    def get_feats(self, loc):
        x = self.read_audio(loc)
        with torch.no_grad():
            audio_sample = torch.from_numpy(x).float()
            audio_sample = self.resample(audio_sample)
            # pre-process the inputs
            inputs = self.feature_extractor(raw_audio=audio_sample, sampling_rate=self.feature_extractor.sampling_rate, return_tensors="pt")
            inputs = inputs["input_values"].cuda()

            # explicitly encode the audio inputs
            encoder_outputs = self.model.encode(inputs, num_quantizers=1).audio_codes[0,0].cpu()
            return encoder_outputs

def get_iterator(args):
    with open(osp.join(args.data, args.split) + ".tsv", "r") as fp:
        lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        files = [osp.join(root, line.split("\t")[0]) for line in lines if len(line) > 0]

        num = len(files)
        reader = MimiFeatureReader()

        def iterate():
            for fname in files:
                mimi_feats = reader.get_feats(fname)

                basename = osp.basename(fname)
                file_id = osp.splitext(basename)[0]
                yield mimi_feats,file_id

    return iterate, num


def main():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    def create_files(dest):
        copyfile(osp.join(args.data, args.split) + ".tsv", dest + ".tsv")
        if osp.exists(osp.join(args.data, args.split) + ".wrd"):
            copyfile(osp.join(args.data, args.split) + ".wrd", dest + ".wrd")
        if osp.exists(osp.join(args.data, args.split) + ".phn"):
            copyfile(osp.join(args.data, args.split) + ".phn", dest + ".phn")

        if osp.exists(dest + ".npy"):
            os.remove(dest + ".npy")
        npaa = NpyAppendArray(dest + ".npy")
        return npaa

    save_path = osp.join(args.save_dir, args.split)
    npaa = create_files(save_path)

    generator, num = get_iterator(args)
    iterator = generator()

    with open(save_path + ".lengths", "w") as l_f, open(save_path + ".ids","w") as i_f:
        for mimi_feats, file_id in tqdm.tqdm(iterator, total=num):
            print(len(mimi_feats), file=l_f)
            print(file_id, file=i_f)

            if len(mimi_feats) > 0:
                npaa.append(mimi_feats.numpy())


if __name__ == "__main__":
    main()
