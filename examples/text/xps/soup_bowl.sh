#!/bin/bash
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/slurm/%x-%a-2.out
#SBATCH --export=NONE

echo "starting experiment"

cd $EXP/flow_matching_speech/examples/text
ml GCC GCCcore binutils libsndfile
source .venv/bin/activate
export SUBMITIT_EXECUTOR=slurm
python run_train.py --config-name librispeech data.cache_dir=$HF_DATASETS_CACHE hydra_dir=./outputs flow.loss_function=generalized_kl training.unsupervised=True -m


echo "multirun complete"


