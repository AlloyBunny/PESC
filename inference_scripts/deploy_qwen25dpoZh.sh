export CUDA_VISIBLE_DEVICES=5
conda activate cfbench
vllm serve models/Qwen2.5-7B-Instruct-hybrid-dpo-zh --port 8005 --max-model-len 16384