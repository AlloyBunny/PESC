export CUDA_VISIBLE_DEVICES=4
conda activate cfbench
vllm serve models/Qwen2.5-7B-Instruct-hybrid-dpo-en --port 8004 --max-model-len 16384