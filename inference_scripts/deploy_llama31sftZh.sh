export CUDA_VISIBLE_DEVICES=3
conda activate cfbench
vllm serve models/Llama-3.1-8B-Instruct --port 8003 --max-model-len 16384