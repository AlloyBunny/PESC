    # 【原参数】
    # --per_device_train_batch_size 2 \
    # --per_device_eval_batch_size 2 \
    # --gradient_accumulation_steps 16 \

    # 【修改后】
    # --per_device_train_batch_size 8 \
    # --per_device_eval_batch_size 8 \
    # --gradient_accumulation_steps 4 \

CUDA_VISIBLE_DEVICES=7 \
swift sft \
    --model models/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset "dataset/train_sft_zh.jsonl" \
    --val_dataset "dataset/val_sft_zh.jsonl" \
    --torch_dtype bfloat16 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 50 \
    --logging_steps 10 \
    --max_length 12288 \
    --output_dir train_output/sft_zh/Qwen2.5-7B-Instruct-sft-zh \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author zhaiyuxuan \
    --model_name Qwen2.5-7B-Instruct-sft-zh