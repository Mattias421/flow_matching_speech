#!/bin/bash
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --output=logs/slurm/%x-%a-2.out
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

work_dir=$1

echo "starting experiment"

# cd $EXP/flow_matching_speech/examples/text

# ml binutils GCCcore GCC libsndfile cuDNN bzip2

source .venv/bin/activate

export PYTHONPATH="."
eval_cmd="python scripts/run_eval.py --work_dir $work_dir --data_name librispeech_dummy --transcribe --split validation.clean --sampling_steps 64 --batch_size 16 --cfg_strength"

# $eval_cmd 2
# $eval_cmd 8
$eval_cmd 0.1
$eval_cmd 0.25
$eval_cmd 0.5
$eval_cmd 0.75
$eval_cmd 1.0
# $eval_cmd 1024

