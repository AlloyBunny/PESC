export CUDA_VISIBLE_DEVICES=1
conda activate cfbench
vllm serve models/Llama-3.1-8B-Instruct-hybrid-dpo-zh --port 8001 --max-model-len 16384