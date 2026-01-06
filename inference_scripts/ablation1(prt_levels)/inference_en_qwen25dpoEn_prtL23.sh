conda activate cfbench
python inference_en/test_npc.py \
    --prt_type personalized \
    --prt_levels L3,L2 \
    --user_llm_type deepseek \
    --assistant_llm_type local \
    --dataset_path profile/test_en_random100.jsonl \
    --store_file inference_output/ablation1/qwen25dpoEn_prtL23.jsonl \
    --base_url http://localhost:8004 \
    --num_threads 3