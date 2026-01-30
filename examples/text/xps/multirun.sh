#!/bin/bash
#SBATCH --array=2-960
#SBATCH --time=5:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=96G
#SBATCH --export=NONE
#SBATCH --output=logs/slurm/%x-%a-2.out
#SBATCH --partition=gpu,gpu-h100,gpu-h100-nvl
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1

echo "starting experiment"

cd $EXP/flow_matching_speech/examples/text
ml binutils GCCcore GCC libsndfile cuDNN bzip2
source .venv/bin/activate
export SUBMITIT_EXECUTOR=slurm

counter=0

for lr in 0.0001 0.0002 0.0003 0.0004; do
for warmup in 500 1000 1500 2000; do
    for t_min in 0.0 0.5 0.75 0.9 1.0; do
        for n_quantizers in 32 64 100 200; do
            for quantizer_groups in 1 2 3; do
    if [ "$SLURM_ARRAY_TASK_ID" == "$counter" ]; then

        # disabling eval because this crashed
        echo "running job $counter with $lr lr"

        python run_train.py -m --config-name librispeech \
        hydra.job.num=$SLURM_ARRAY_TASK_ID \
        training.eval_freq=1000000 \
        data.text_data=$DATA/variety-text-corpus/LibriLM/text/phones \
        data.features_path=$DATA/LibriSpeech-10hr-rVAD/features/wav2vec_vox \
        hydra_dir=./outputs  \
        study_name=the_big_rando_2 \
        optim.lr=$lr \
        optim.warmup=$warmup \
        training.t_min=$t_min \
        model.n_quantizers=$n_quantizers \
        model.quantizer_groups=$quantizer_groups
fi
    counter=$(( counter + 1 ))

done
done
done
done
done

echo "jobs done"


