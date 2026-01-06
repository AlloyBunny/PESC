conda activate cfbench
python inference_zh/test_npc.py \
    --prt_type personalized \
    --prt_levels L3,L2,L1 \
    --user_llm_type deepseek \
    --assistant_llm_type local \
    --dataset_path profile/test_zh_last10users.jsonl \
    --store_file inference_output/analyse1/qwen25dpoZh_prt.jsonl \
    --base_url http://localhost:8005 \
    --num_threads 3