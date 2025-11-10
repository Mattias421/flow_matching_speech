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

cd $EXP/flow_matching_speech/examples/text

ml binutils GCCcore GCC libsndfile cuDNN bzip2

source .venv/bin/activate

export PYTHONPATH="."
eval_cmd="python scripts/run_eval.py --work_dir $work_dir --data_name librispeech --transcribe --split test.clean --sampling_steps"

$eval_cmd 2
$eval_cmd 8
$eval_cmd 64
$eval_cmd 256
$eval_cmd 512
# $eval_cmd 1024

eval_cmd="python scripts/run_eval.py --work_dir $work_dir --data_name librispeech --transcribe --split test.other --sampling_steps"

$eval_cmd 2
$eval_cmd 8
$eval_cmd 64
$eval_cmd 256
$eval_cmd 512
# $eval_cmd 1024

# python scripts/run_eval.py --work_dir $work_dir --data_name librispeech --split test.other --eval_elbo
# python scripts/run_eval.py --work_dir $work_dir --data_name librispeech --split test.clean --eval_elbo
