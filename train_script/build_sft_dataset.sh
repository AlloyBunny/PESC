conda activate cfbench

python utils/get_sft_dataset.py \
    --input_file_path dataset/zh/train_raw_zh_with_strategy.jsonl \
    --output_file_path dataset/zh/train_sft_zh.jsonl \
    --language zh

python utils/get_sft_dataset.py \
    --input_file_path dataset/zh/val_raw_zh_with_strategy.jsonl \
    --output_file_path dataset/zh/val_sft_zh.jsonl \
    --language zh

python utils/get_sft_dataset.py \
    --input_file_path dataset/en/train_raw_en_with_strategy.jsonl \
    --output_file_path dataset/en/train_sft_en.jsonl \
    --language en

python utils/get_sft_dataset.py \
    --input_file_path dataset/en/val_raw_en_with_strategy.jsonl \
    --output_file_path dataset/en/val_sft_en.jsonl \
    --language en