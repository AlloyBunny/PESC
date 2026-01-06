conda activate cfbench
python inference_en/test_npc.py \
    --prt_type personalized \
    --prt_levels L3,L2,L1 \
    --user_llm_type deepseek \
    --assistant_llm_type local \
    --dataset_path profile/test_en.jsonl \
    --store_file inference_output/en/llama31dpoEn_prt.jsonl \
    --base_url http://localhost:8000 \
    --num_threads 3