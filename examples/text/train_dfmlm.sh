#!/bin/bash
#SBATCH --partition=gpu,gpu-h100
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=80:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8G
#SBATCH --output=logs/slurm/%x-%a-2.out

#echo "Copying data tar to $TMPDIR"
#cp $DATA/fineweb-edu.tar $TMPDIR
#cp $DATA/wikitext.tar $TMPDIR
#echo "decompressing"
#cd $TMPDIR
#tar -xf fineweb-edu.tar
#tar -xf wikitext.tar

echo "starting experiment"

cd $EXP/flow_matching_speech/examples/text

ml binutils
ml GCCcore
ml GCC
ml libsndfile

source .venv/bin/activate
python run_train.py data.cache_dir=$DATA training.batch_size=256 eval.batch_size=256 optim.lr=1.5e-4 optim.warmup=5000 hydra.run.dir=./outputs/2025-10-17/15-21-21

