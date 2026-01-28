#!/bin/bash
uv run run_train.py --config-name dummy_libri hydra_dir=./outputs training.time_conditioning=false flow.loss_function=cross_entropy
