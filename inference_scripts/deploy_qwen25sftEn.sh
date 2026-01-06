export CUDA_VISIBLE_DEVICES=6
conda activate cfbench
vllm serve models/Qwen2.5-7B-Instruct --port 8006 --max-model-len 16384