# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import logging
import os
import contextlib

import numpy as np
import torch
from torch.utils.data import Dataset

from data.utils import collate_fn_unpaired


logger = logging.getLogger(__name__)


class ExtractedFeaturesDataset(Dataset):
    def __init__(
        self,
        path,
        split,
        min_length=3,
        max_length=None,
        labels=None,
        shuffle=True,
        sort_by_length=True,
        aux_target_postfix=None,
        tokenizer=None,
    ):
        super().__init__()

        self.min_length = min_length
        self.max_length = max_length
        self.shuffle = shuffle
        self.sort_by_length = sort_by_length

        self.sizes = []
        self.offsets = []
        self.labels = []
        self.aux_tgt = None
        self.tokenizer = tokenizer

        path = os.path.join(path, split)
        data_path = path
        self.data = np.load(data_path + ".npy", mmap_mode="r")
        with open(data_path + ".ids", "r") as f:
            self.ids = f.readlines()

        offset = 0
        skipped = 0

        if not os.path.exists(path + f".{labels}"):
            labels = None

        with (
            open(data_path + ".lengths", "r") as len_f,
            open(path + f".{labels}", "r")
            if labels is not None
            else contextlib.ExitStack() as lbl_f,
        ):
            for line in len_f:
                length = int(line.rstrip())
                lbl = None if labels is None else next(lbl_f).strip()
                if length >= min_length and (
                    max_length is None or length <= max_length
                ):
                    self.sizes.append(length)
                    self.offsets.append(offset)
                    if lbl is not None:
                        self.labels.append(lbl)
                offset += length

        self.sizes = np.asarray(self.sizes)
        self.offsets = np.asarray(self.offsets)

        if aux_target_postfix is not None:
            if not os.path.exists(path + f".{aux_target_postfix}"):
                logger.info(f"auxaliry target for {split} missing")
            else:
                with open(path + f".{aux_target_postfix}", "r") as t_f:
                    self.aux_tgt = [
                        torch.LongTensor(list(map(int, seg.strip().split())))
                        for seg in t_f
                    ]

        logger.info(f"loaded {len(self.offsets)}, skipped {skipped} samples")

    def __getitem__(self, index):
        offset = self.offsets[index]
        end = self.sizes[index] + offset
        feats = torch.from_numpy(self.data[offset:end].copy()).float()

        res = {"id": self.ids[index], "features": feats}
        if len(self.labels) > 0:
            res["target"] = self.labels[index]

            if self.tokenizer is not None:
                res["input_ids"] = self.tokenizer(res["target"])


        if self.aux_tgt:
            res["aux_target"] = self.aux_tgt[index]

        return res

    def __len__(self):
        return len(self.sizes)

    def collater(self, samples):
        if len(samples) == 0:
            return {}

        features = [s["features"] for s in samples]
        sizes = [len(s) for s in features]

        assert max(sizes) <= self.max_length
        target_size = max(sizes)

        if len(features[0].shape) == 2:
            collated_features = features[0].new_zeros(
                len(features), target_size, features[0].size(-1), dtype=torch.long,
            )
            padding_mask = torch.BoolTensor(collated_features.shape[:-1]).fill_(False)
        elif len(features[0].shape) == 1:
            collated_features = features[0].new_zeros(
                len(features),
                target_size,
                dtype=torch.long,
            )
            padding_mask = torch.BoolTensor(collated_features.shape).fill_(False)

        for i, (f, size) in enumerate(zip(features, sizes)):
            collated_features[i, :size] = f
            padding_mask[i, size:] = True

        assert collated_features.dtype == torch.long

        res = {
            "id": [s["id"].strip() for s in samples],
            "input_ids": collated_features,
            "padding_mask": padding_mask,
        }

        if len(self.labels) > 0:
            res["target"] = [s["target"] for s in samples]
            collate_toks = collate_fn_unpaired(samples, self.max_length)
            res["input_ids_text"] = collate_toks["input_ids"]
            res["padding_mask_text"] = collate_toks["padding_mask"]

        if self.aux_tgt:
            idxs = torch.nn.utils.rnn.pad_sequence(
                [s["aux_target"] for s in samples],
                batch_first=True,
                padding_value=-1,
            )
            res["aux_target"] = idxs

        return res

    def num_tokens(self, index):
        return self.size(index)

    def size(self, index):
        return self.sizes[index]

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]

        if self.sort_by_length:
            order.append(self.sizes)
            return np.lexsort(order)[::-1]
        else:
            return order[0]
