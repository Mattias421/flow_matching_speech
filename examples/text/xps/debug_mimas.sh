CUDA_VISIBLE_DEVICES=0 uv run run_train.py --config-name dummy_libri data.cache_dir=$HF_DATASETS_CACHE hydra_dir=./outputs 
