conda activate cfbench
python inference_zh/test_npc.py \
    --prt_type personalized \
    --prt_levels L3,L2,L1 \
    --user_llm_type deepseek \
    --assistant_llm_type local \
    --dataset_path profile/test_zh.jsonl \
    --store_file inference_output/zh/llama31dpoZh_prt.jsonl \
    --base_url http://localhost:8001 \
    --num_threads 3