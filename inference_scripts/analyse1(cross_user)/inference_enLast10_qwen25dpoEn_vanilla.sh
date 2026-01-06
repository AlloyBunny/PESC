conda activate cfbench
python inference_en/test_npc.py \
    --prt_type none \
    --prt_levels L3,L2,L1 \
    --user_llm_type deepseek \
    --assistant_llm_type local \
    --dataset_path profile/test_en_last10users.jsonl \
    --store_file inference_output/analyse1/qwen25dpoEn_vinilla.jsonl \
    --base_url http://localhost:8004 \
    --num_threads 3