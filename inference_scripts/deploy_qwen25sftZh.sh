export CUDA_VISIBLE_DEVICES=7
conda activate cfbench
vllm serve models/Qwen2.5-7B-Instruct --port 8007 --max-model-len 16384