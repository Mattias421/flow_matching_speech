#!/bin/bash
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/slurm/%x-%a-2.out
#SBATCH --export=NONE

echo "starting experiment"

cd $EXP/flow_matching_speech/examples/text
ml GCC GCCcore binutils libsndfile
source .venv/bin/activate
export SUBMITIT_EXECUTOR=slurm
export WANDB_TAGS="lr_sweep"
python run_train.py --config-name librispeech data.cache_dir=$HF_DATASETS_CACHE hydra_dir=./outputs flow.loss_function=cross_entropy,generalized_kl optim.lr=0.001 optim.warmup=400,1000 -m

# python run_train.py --config-name librispeech data.cache_dir=$HF_DATASETS_CACHE hydra_dir=./outputs flow.loss_function=cross_entropy,generalized_kl optim.lr=0.002,0.0005 optim.warmup=100,200,400,1000 -m

echo "multirun complete"


