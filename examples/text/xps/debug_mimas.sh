CUDA_VISIBLE_DEVICES=0,1,2,3 uv run run_train.py --config-name librispeech data.cache_dir=$HF_DATASETS_CACHE hydra_dir=./outputs eval.kenlm_path=$KENLM
