#!/bin/bash
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=96G
#SBATCH --output=logs/slurm/%x-%a-2.out
#SBATCH --export=NONE

echo "starting experiment"

cd $EXP/flow_matching_speech/examples/text
ml GCC GCCcore binutils libsndfile
source .venv/bin/activate
export SUBMITIT_EXECUTOR=slurm
python run_train.py --config-name librispeech data.cache_dir=$HF_DATASETS_CACHE data.features_path=$DATA/LibriSpeech-Clean-NoSil/features/mimi/ hydra_dir=./outputs  study_name=the_big_rando -m


echo "multirun complete"


