import argparse
import json
# python checklist/get_dpo_dataset.py --judge_path checklist_output/dpo/judge/Qwen2_5_7B_Instruct_sft_pairwise_eval.json --output_path dataset/train_dpo.jsonl
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt4o")
parser.add_argument("--judge_path", type=str, default="checklist_output/dpo/judge/gpt4o_pairwise_eval.jsonl")
parser.add_argument("--output_path", type=str, default="dataset/train_dpo.jsonl")
args = parser.parse_args()
model_name = args.model_name
judge_path = args.judge_path
output_path = args.output_path

with open(judge_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
    f.close()

avg_score_0, avg_score_1 = 0, 0
cnt0, cnt1, cnt2, cnt3 = 0, 0, 0, 0
score_difference_list = []
for item in data:
    if float(item["score_difference"]) == 0:
        cnt0 += 1
    elif item["best_result"]["iteration"] != 0:
        print(f"第{item['idx']}项数据，有个性化记忆的分数更差")
        cnt1 += 1
    elif float(item["score_difference"]) < 3:
        print(f"第{item['idx']}项数据，有个性化记忆没有明显更优")
        cnt2 += 1
        score_difference_list.append(float(item["score_difference"]))
    else:
        result = {
            "messages": item["dialog_history"] + [{"role": "assistant", "content": item["best_result"]["response"]}],
            "rejected_response": item["worst_result"]["response"]
        }
        cnt3 += 1
        score_difference_list.append(float(item["score_difference"]))
    
    if item["best_result"]["iteration"]==0:
        avg_score_0 += float(item["best_result"]["judge_metric"]["custom_score"])
        avg_score_1 += float(item["worst_result"]["judge_metric"]["custom_score"])
    else:
        avg_score_0 += float(item["worst_result"]["judge_metric"]["custom_score"])
        avg_score_1 += float(item["best_result"]["judge_metric"]["custom_score"])
    
avg_score_0 /= len(data)
avg_score_1 /= len(data)

print("================================================")
print(f"{model_name}执行结束")
print(f"有/无个性化记忆分数相等: {cnt0}个")
print(f"有个性化记忆的分数更差: {cnt1}个")
print(f"有个性化记忆稍微更优: {cnt2}个（分数差值<3）")
print(f"有个性化记忆明显更优: {cnt3}个（分数差值>=3）")
print(f"有个性化平均分: {avg_score_0:.2f}，无个性化平均分: {avg_score_1:.2f}")
print(f"分数差值列表: {score_difference_list}")

    
"""
【输入格式: json】
{
    "idx": 0,
    "dialog_history": [
        {
          "role": "system",
          "content": "你是一个智能聊天伙伴，你擅长根据用户的性格和偏好..."
        },
        {
            "role": "user",
            "content": "最近工作上的压力让我有些喘不过气来，真的需要一些建议。"
        }
    ],
    "criteria": [
        [
            "回应需通过细腻的语言和比喻表达方式来传达对用户的理解和共情（例如，使用隐喻来描绘用户的情感状态）",
            "主需",
            "风格约束",
            "间接表达约束"
        ],
        [
            "提供的建议必须具体、细节明确，并与用户表达的情感状态相匹配（例如，通过艺术活动如参观博物馆来舒缓压力）",
            "主需",
            "建议方式约束",
            "具体与情境相关约束"
        ],
        ...
    ],
    "split": "hard",
    "infer_model": "Qwen2_5_7B_Instruct_sft",
    "temperature": 0.7,
    "worst_result": {
        "iteration": 1,
        "response": "听起来你确实面临着一些挑战，压力有时候真的很难处理。首先，给自己一点时间休息和恢复是很重要的。可以试着以下几个方法：\n\n1. **分解任务**：把大任务分解成小步骤，逐步完成，这样会让事情看起来更容易掌控。\n\n2. **优先级管理**：列出工作任务，优先处理重要和紧急的事项。\n\n3. **适当休息**：短暂的休息有助于恢复精力，记得每天给自己安排一些放松的时间。\n\n4. **寻求支持**：跟同事、朋友或家人聊聊，有时分享问题本身就是一种缓解压力的方式。\n\n5. **自我关怀**：做一些你喜欢的事情，比如运动、冥想或阅读，帮助自己放松心情。\n\n记住，工作虽然重要，但健康永远是第一位的。如果你觉得需要更长时间的休息或咨询，请随时告知领导或相关负责人。希望你能慢慢找到平衡，早日恢复轻松的状态。",
        "judge_response": "标准1: 回应需通过细腻的语言和比喻表达方式来传达对用户的理解和共情（例如，使用隐喻来描绘用户的情感状态）\t0\n标准2: 提供的建议必须具体、细节明确，并与用户表达的情感状态相匹配（例如，通过艺术活动如参观博物馆来舒缓压力）\t1\n标准3: 避免使用空泛、无细节的鼓励性语言，如“加油”或“多努力”，应提供具有情感深度的支持\t1\n标准4: 在情感共鸣方面，应深度理解用户的情感复杂性，使用比喻或故事来展示这一理解\t0\n标准5: 提供的支持应避免单纯的策略性建议，而需包含对用户情感状态的深刻共鸣，以此为前提展开具体建议\t0\n标准6: 当用户提到孤独感时，回应应优先考虑在情感上给予支持和理解，并通过细腻语言和比喻表达共情\t0\n标准7: 在提供建议时，需强调其可操作性和与用户当前情感状态的相关性，防止情绪进一步下降\t1",
        "judge_score": [
            "0",
            "1",
            "1",
            "0",
            "0",
            "0",
            "1"
        ],
        "judge_parsing": "1",
        "judge_metric": {
            "CSR": 0.43,
            "ISR": 0,
            "PSR": 0,
            "custom_score": 5.0
        }
    },
    "best_result": {
        "iteration": 0,
        "response": "听起来你确实面临着一些挑战，工作压力有时候真的很难处理。首先，我想说能够意识到自己需要帮助是非常好的第一步。以下是一些可能对你有帮助的方法：\n\n1. **优先排序**：试着列出你的任务清单，区分哪些事情是紧急且重要的，优先处理这些任务。这样可以帮助你更好地安排时间，减少压力。\n\n2. **合理休息**：确保自己有足够的休息时间，哪怕是短暂的休息也能让你恢复精力。尝试每天留出一段时间做自己喜欢的事情，无论是读书、运动还是简单的冥想。\n\n3. **与人交流**：有时候和同事或朋友分享你的感受可以减轻负担，他们可能也会给你一些有用的建议或者只是提供一个倾诉的空间。\n\n4. **设定界限**：学会说“不”，避免承担过多的任务。确保你的工作量是合理的，避免过度加班，以免影响身心健康。\n\n5. **保持健康的生活方式**：均衡饮食、定期运动和充足睡眠对减轻压力都很有帮助。\n\n记住，压力是生活的一部分，但我们可以学习如何更好地应对。希望这些建议能对你有所帮助，如果需要更多的支持，请随时告诉我。",
        "judge_response": "标准1: 回应需通过细腻的语言和比喻表达方式来传达对用户的理解和共情（例如，使用隐喻来描绘用户的情感状态）\t0\n标准2: 提供的建议必须具体、细节明确，并与用户表达的情感状态相匹配（例如，通过艺术活动如参观博物馆来舒缓压力）\t1\n标准3: 避免使用空泛、无细节的鼓励性语言，如“加油”或“多努力”，应提供具有情感深度的支持\t1\n标准4: 在情感共鸣方面，应深度理解用户的情感复杂性，使用比喻或故事来展示这一理解\t0\n标准5: 提供的支持应避免单纯的策略性建议，而需包含对用户情感状态的深刻共鸣，以此为前提展开具体建议\t0\n标准6: 当用户提到孤独感时，回应应优先考虑在情感上给予支持和理解，并通过细腻语言和比喻表达共情\t1\n标准7: 在提供建议时，需强调其可操作性和与用户当前情感状态的相关性，防止情绪进一步下降\t1",
        "judge_score": [
            "0",
            "1",
            "1",
            "0",
            "0",
            "1",
            "1"
        ],
        "judge_parsing": "1",
        "judge_metric": {
            "CSR": 0.57,
            "ISR": 0,
            "PSR": 0,
            "custom_score": 6.0
        }
    },
    "score_difference": 1.0
}

【输出格式: jsonl】
{
    "messages": [
        {
          "role": "system",
          "content": "你是一个智能聊天伙伴，你擅长根据用户的性格和偏好..."
        },
        {
            "role": "user",
            "content": "最近工作上的压力让我有些喘不过气来，真的需要一些建议。"
        },
        {
            "role": "assistant",
            "content": "听起来你确实面临着一些挑战，工作压力有时候真的很难处理。首先，我想说能够意识到自己需要帮助是非常好的第一步。以下是一些可能对你有帮助的建议：\n\n1. **优先排序**：试着列出你的任务清单，区分哪些是紧急且重要的，优先处理这些任务。这样可以帮助你更好地安排时间，减少压力。\n\n2. **合理休息**：确保自己有足够的休息时间，哪怕是短暂的休息也能让你恢复精力。尝试每天留出一段时间做自己喜欢的事情，无论是读书、运动还是简单的冥想。\n\n3. **与人交流**：有时候和同事或朋友分享你的感受可以减轻负担，他们可能也会给你一些有用的建议或者只是提供一个倾诉的空间。\n\n4. **设定界限**：学会说“不”，避免承担过多的任务。确保你的工作量是合理的，避免过度加班，以免影响身心健康。\n\n5. **保持健康的生活方式**：均衡饮食、定期运动和充足睡眠对减轻压力都很有帮助。\n\n记住，压力是生活的一部分，但我们可以学习如何更好地应对。希望这些建议能对你有所帮助，如果需要更多的支持，请随时告诉我。"
        }
    ],
    "rejected_response": "我很抱歉你感到无助..."
}
"""