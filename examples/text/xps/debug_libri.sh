#!/bin/bash
#SBATCH --partition=gpu,gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/slurm/%x-%a-2.out

echo "starting experiment"

cd $EXP/flow_matching_speech/examples/text

ml binutils
ml GCCcore
ml GCC
ml libsndfile

source .venv/bin/activate
python run_train.py --config-name librispeech data.cache_dir=$DATA

