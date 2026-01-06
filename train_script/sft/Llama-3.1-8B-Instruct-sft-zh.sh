conda activate swift
CUDA_VISIBLE_DEVICES=5 \
swift sft \
    --model models/Llama-3.1-8B-Instruct \
    --model_type llama3_1 \
    --train_type lora \
    --dataset "dataset/train_sft_zh.jsonl" \
    --val_dataset "dataset/val_sft_zh.jsonl" \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 8 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 50 \
    --logging_steps 10 \
    --max_length 12288 \
    --output_dir train_output/sft_zh/Llama-3.1-8B-Instruct-sft-zh \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author anonymous \
    --model_name Llama-3.1-8B-Instruct-sft-zh \
    --ddp_find_unused_parameters false