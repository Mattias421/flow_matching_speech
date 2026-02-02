# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

import os

import hydra
import torch.multiprocessing as mp

from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import open_dict
from omegaconf.dictconfig import DictConfig
from train import run_mp_training
import socket
from contextlib import closing

from utils import checkpointing


@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    if "load_dir" in cfg:
        work_dir = cfg.load_dir
        cfg = checkpointing.load_cfg_from_path(cfg.load_dir)
    else:
        work_dir = os.getcwd()
        os.makedirs(work_dir, exist_ok=True)

    with open_dict(cfg):
        cfg.work_dir = work_dir


    cer = run_mp_training(rank=0, world_size=1, cfg=cfg)
    print(f"run_mp_training returns {cer}")
    return cer


if __name__ == "__main__":
    main()
