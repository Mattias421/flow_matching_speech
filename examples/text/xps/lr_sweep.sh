#!/bin/bash
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/slurm/%x-%a-2.out
#SBATCH --export=NONE

echo "starting experiment"

cd $EXP/flow_matching_speech/examples/text
ml GCC GCCcore binutils libsndfile
source .venv/bin/activate
export SUBMITIT_EXECUTOR=slurm
python run_train.py --config-name librispeech data.cache_dir=$HF_DATASETS_CACHE hydra_dir=./outputs optim.lr=1e-4,1e-3,1e-5 optim.warmup=200,500,1000,100 -m 

echo "multirun complete"


