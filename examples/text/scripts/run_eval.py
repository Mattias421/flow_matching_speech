# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
# Part of this implementation is adapted from https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
# which is released under MIT license

import argparse

import torch.multiprocessing as mp
import socket
from contextlib import closing

from eval import run_mp_eval

def find_free_port():
    """Finds a free port on the host machine."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))  # Bind to port 0 to let the OS pick a free port
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]  # Return the port number assigned by the OS


def main(args: argparse.Namespace):
    port = find_free_port()

    assert args.batch_size % args.ngpus == 0

    if args.ngpus == 1:
        run_mp_eval(
            rank=0,
            world_size=1,
            seed=args.seed,
            work_dir=args.work_dir,
            batch_size=args.batch_size // args.ngpus,
            split=args.split,
            sampling_steps=args.sampling_steps,
            transcribe=args.transcribe,
            eval_elbo=args.eval_elbo,
            data_name=args.data_name,
            port=port,
        )
    else:
        mp.set_start_method("forkserver")

        mp.spawn(
            run_mp_eval,
            args=(
                args.ngpus,
                args.seed,
                args.work_dir,
                args.batch_size // args.ngpus,
                args.sampling_steps,
                args.eval_elbo,
                args.eval_perplexity,
                args.elbo_data,
                args.perplexity_n_samples // args.ngpus,
                port,
            ),
            nprocs=args.ngpus,
            join=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--work_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--ngpus", type=int, default=1)

    parser.add_argument("--eval_elbo", action="store_true")
    parser.add_argument("--transcribe", action="store_true")

    # Perplexity parameters
    parser.add_argument("--sampling_steps", type=int, default=1024)

    # ELBO parameters
    parser.add_argument("--data_name", type=str, default="librispeech")
    parser.add_argument("--split", type=str, default="test.clean")

    args = parser.parse_args()
    main(args)
