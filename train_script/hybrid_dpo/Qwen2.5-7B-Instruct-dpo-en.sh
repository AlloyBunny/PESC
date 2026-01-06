    # 【原参数】
    # --per_device_train_batch_size 1 \
    # --per_device_eval_batch_size 1 \
    # --gradient_accumulation_steps 8 \

    # 【修改后】
    # --per_device_train_batch_size 4 \
    # --per_device_eval_batch_size 4 \
    # --gradient_accumulation_steps 2 \

CUDA_VISIBLE_DEVICES=2 \
swift rlhf \
    --rlhf_type dpo \
    --model models/Qwen2.5-7B-Instruct \
    --train_type lora \
    --dataset dataset/train_dpo_en.jsonl \
    --val_dataset dataset/val_dpo_en_sample1000.jsonl \
    --torch_dtype bfloat16 \
    --tf32 false \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 5e-5 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 2 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 50 \
    --logging_steps 10 \
    --max_length 12288 \
    --output_dir train_output/dpo_en/Qwen2.5-7B-Instruct-dpo-en \
    --warmup_ratio 0.05 \
    --gradient_checkpointing true \
    --rpo_alpha 1.0 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8