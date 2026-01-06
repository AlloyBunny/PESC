# 注：adapters和 output_dir需要根据实际情况修改
# 其中adapters是checkpoint的路径，output_dir是保存模型的路径
conda activate swift
export CUDA_VISIBLE_DEVICES=0
swift export \
    --adapters checkpoint/Qwen2.5-7B-Instruct-dpo-en \
    --merge_lora true \
    --output_dir models/Qwen2.5-7B-Instruct-dpo-en
swift export \
    --adapters checkpoint/Llama-3.1-8B-Instruct-dpo-en \
    --merge_lora true \
    --output_dir models/Llama-3.1-8B-Instruct-dpo-en
swift export \
    --adapters checkpoint/Qwen2.5-7B-Instruct-dpo-zh \
    --merge_lora true \
    --output_dir models/Qwen2.5-7B-Instruct-dpo-zh
swift export \
    --adapters checkpoint/Llama-3.1-8B-Instruct-dpo-zh \
    --merge_lora true \
    --output_dir models/Llama-3.1-8B-Instruct-dpo-zh