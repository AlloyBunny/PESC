export CUDA_VISIBLE_DEVICES=2
conda activate cfbench
vllm serve models/Llama-3.1-8B-Instruct-sft-en --port 8002 --max-model-len 16384