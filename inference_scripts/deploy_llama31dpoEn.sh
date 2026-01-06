export CUDA_VISIBLE_DEVICES=0
conda activate cfbench
vllm serve models/Llama-3.1-8B-Instruct-hybrid-dpo-en --port 8000 --max-model-len 16384