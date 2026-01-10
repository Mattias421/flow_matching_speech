uv run run_train.py --config-name dummy_libri data.cache_dir=$HF_DATASETS_CACHE hydra_dir=./outputs training.unsupervised_prob=1.0 data.supervised=false optim.n_iters=10000
