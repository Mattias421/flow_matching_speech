#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/slurm/%x.out
#SBATCH --export=NONE

echo "starting experiment"

cd $EXP/flow_matching_speech/examples/text
ml GCC GCCcore binutils libsndfile
source .venv/bin/activate
export SUBMITIT_EXECUTOR=slurm
export WANDB_TAGS="long_boi"

python run_train.py --config-name librispeech -m \
    data.cache_dir=$HF_DATASETS_CACHE \
    hydra_dir=./outputs \
    flow.loss_function=generalized_kl \
    hydra.launcher.name="long_boi_grid" \
    optim.n_iters=80000 \
    optim.lr=0.0001,0.001 \
    optim.warmup=2500,5000,8000 \
    optim.weight_decay=0.003,0.01 \
    model.dropout=0.01,0.1 \
    flow.partial_noise_prob=0.01

echo "multirun complete"


