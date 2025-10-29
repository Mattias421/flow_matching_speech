#!/bin/bash
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/slurm/%x-%a-2.out
#SBATCH --export=NONE
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

echo "starting experiment"

cd $EXP/flow_matching_speech/examples/text
ml binutils GCCcore GCC libsndfile cuDNN bzip2

source .venv/bin/activate
python run_train.py --config-name librispeech hydra_dir=outputs +load_dir=$1

echo "run complete"


