uv run run_train.py --config-name dummy_libri data.cache_dir=$HF_DATASETS_CACHE hydra_dir=./outputs training.time_conditioning=false data.supervised=true flow.loss_function=cross_entropy
