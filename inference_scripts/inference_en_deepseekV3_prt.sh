conda activate cfbench
python inference_en/test_npc.py \
    --prt_type personalized \
    --prt_levels L3,L2,L1 \
    --user_llm_type deepseek \
    --assistant_llm_type deepseek \
    --dataset_path profile/test_en.jsonl \
    --store_file inference_output/en/deepseekV3_prt.jsonl \
    --num_threads 10