#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
conda activate cfbench

# ------重要参数设置-----

# character_id_start和character_id_end是选用哪些角色数据加入DPO数据集，character_id范围是1~100，对应100个角色卡，详见utils/get_character_id.py
character_id_start=1
character_id_end=100
# 数据集类型
dataset_type=val
dpo_dataset_path=dataset/${dataset_type}_dpo.jsonl
# -----------------------


# -----Step 1: 预处理数据-----
input_file=dataset/${dataset_type}.jsonl
preprocessed_file=dataset/${dataset_type}_${character_id_start}to${character_id_end}_preprocessed.jsonl

echo "-----Step 1: 预处理数据-----"
echo "Input file: ${input_file}"
echo "Output file: ${preprocessed_file}"
echo "=========================="

python checklist/preprocess.py \
    --input_file ${input_file} \
    --output_file ${preprocessed_file} \
    --character_id_start ${character_id_start} \
    --character_id_end ${character_id_end}

# -----Step 2: 推理得到chosen回复-----
# 推理配置
infer_model=gpt41 # 可选gpt41, gpt4o, deekseek_v3
infer_max_threads=10
# infer_out_dir=checklist_output/${dataset_type}_${character_id_start}to${character_id_end}/response
infer_out_file=checklist_output/${dataset_type}_${character_id_start}to${character_id_end}/response/${infer_model}_pairwise_infer.jsonl
temperature=0.7            # 可调温度参数

echo ""
echo "-----Step 2: 推理得到chosen回复-----"
echo "Model: ${infer_model}"
echo "Temperature: ${temperature}"
# echo "Iterations: ${num_iterations}"
echo "Input data: ${preprocessed_file}"
echo "Output file: ${infer_out_file}"
echo "=========================="

python checklist/inference_preference_pair_jsonl.py \
    --infer_model ${infer_model} \
    --input_path ${preprocessed_file} \
    --output_path ${infer_out_file} \
    --max_threads ${infer_max_threads} \
    --temperature ${temperature}

# 检查推理是否成功
if [ $? -eq 0 ]; then
    echo "Inference completed successfully!"
else
    echo "Inference failed!"
    exit 1
fi

# -----Step 3: 获取checklist-----
# checklist模型配置（注意这里用的是inference/call_llm.py中的模型）
checklist_llm_type=deepseek # 可选gpt, deepseek, gemini, local（gpt是gpt4o, deepseek是deepseek_v3, gemini是gemini-1.5-pro, local是vllm部署的本地模型）
# checklist输出路径
checklist_output_file=checklist_output/${dataset_type}_${character_id_start}to${character_id_end}/response/${infer_model}_pairwise_infer_with_checklist.jsonl

echo ""
echo "-----Step 3: 获取checklist-----"
echo "LLM type: ${checklist_llm_type}"
echo "Input file: ${infer_out_file}"
echo "Output file: ${checklist_output_file}"
echo "=========================="

python checklist/get_checklist_dataset_multi_thread.py \
    --llm_type ${checklist_llm_type} \
    --input_file ${infer_out_file} \
    --output_file ${checklist_output_file}
    

# -----Step 4: 评估chosen和rejected哪个更优-----
# 评估配置（没有温度选项，因为温度默认固定为0.01）
eval_model=deepseek_v3 # 可选gpt4o, deekseek_v3
eval_out_file=checklist_output/${dataset_type}_${character_id_start}to${character_id_end}/judge/${infer_model}_pairwise_eval.jsonl
# eval_score_path=checklist_output/${dataset_type}_${character_id_start}to${character_id_end}/scores.xlsx
eval_max_threads=10

echo ""
echo "-----Step 4: 评估chosen和rejected哪个更优-----"
echo "Input file: ${checklist_output_file}"
echo "Output file: ${eval_out_file}"
echo "=========================="

python checklist/evaluate_preference_pair.py \
    --input_path ${checklist_output_file} \
    --output_path ${eval_out_file} \
    --max_threads ${eval_max_threads} \
    --eval_model ${eval_model}

# 检查评估是否成功
if [ $? -eq 0 ]; then
    echo "Evaluation completed successfully!"
    echo ""
    echo "Results saved to:"
    echo "  - Pairwise results: ${eval_out_dir}/${infer_model}_pairwise_eval.json"
    echo "  - Scores: ${eval_score_path}"
else
    echo "Evaluation failed!"
    exit 1
fi

# -----Step 5: 构建DPO数据集-----
echo ""
echo "-----Step 5: 构建DPO数据集-----"
echo "Input file: ${eval_out_file}"
echo "Output file: ${dpo_dataset_path}"
echo "=========================="

python utils/get_dpo_dataset.py \
    --input_file ${eval_out_file} \
    --output_file ${dpo_dataset_path}

echo "整个.sh脚本执行完成！"