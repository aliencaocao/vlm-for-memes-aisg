export DISABLE_VERSION_CHECK=True
export WANDB_USERNAME='aisg-meme'
export WANDB_PROJECT='qwen2-vl'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export FORCE_TORCHRUN=1
ulimit -n 1048576
CUDA_VISIBLE_DEVICES=0 llamafactory-cli train a100_inference.yaml