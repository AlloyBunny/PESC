conda activate cfbench
python inference_en/test_npc.py \
    --prt_type none \
    --prt_levels L3,L2,L1 \
    --user_llm_type deepseek \
    --assistant_llm_type local \
    --dataset_path profile/test_en.jsonl \
    --store_file inference_output/en/qwen25base_prt.jsonl \
    --base_url http://localhost:8006 \
    --num_threads 3