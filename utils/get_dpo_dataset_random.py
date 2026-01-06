import json
from collections import Counter
import argparse
import os
import random

# 设置随机种子确保可复现
random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str, default="dataset/train_dpo_with_checklist_final.jsonl")
parser.add_argument("--output_file", type=str, default="dataset/train_dpo_random.jsonl")
# 添加可选参数控制是否只从有效数据中随机抽取（排除error数据）
parser.add_argument("--only_valid", action="store_true", default=True, 
                    help="是否只从非错误数据中随机抽取（默认True）")
args = parser.parse_args()

input_file = args.input_file
output_file = args.output_file

# 读取原始数据
with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

# 首先统计原始数据中的各类数据数量（和原代码逻辑一致）
perfect_cnt, good_cnt, not_bad_cnt, bad_cnt, error_cnt = 0, 0, 0, 0, 0
valid_data = []  # 存储非错误数据
error_data = []  # 存储错误数据

for item in data:
    # 标记是否为错误数据
    is_error = False
    if len(item["judge_result"]) != len(item["criteria"]):
        is_error = True
    else:
        counter = Counter(item["judge_result"])
        invalid_numbers = [num for num in counter if num not in {0, 1, 2}]
        if invalid_numbers:
            is_error = True
    
    if is_error:
        error_cnt += 1
        error_data.append(item)
        continue
    
    # 非错误数据，统计各类数量
    valid_data.append(item)
    counter = Counter(item["judge_result"])
    if counter[2]==0 and counter[0]==0 and counter[1]>0:
        perfect_cnt += 1
    elif counter[2]==0 and counter[1]>=4:
        good_cnt += 1
    elif counter[2]==0 and counter[1]>0:
        not_bad_cnt += 1
    else:
        bad_cnt += 1

# 计算需要随机抽取的样本数量（完美+好数据的总数）
target_sample_num = perfect_cnt + good_cnt
print(f"原始数据统计 - 完美数据：{perfect_cnt}，好数据：{good_cnt}，不坏不优数据：{not_bad_cnt}，坏数据：{bad_cnt}，错误数据：{error_cnt}")
print(f"需要随机抽取的样本数量：{target_sample_num}")

# 执行随机抽取
if args.only_valid:
    # 只从有效数据中随机抽取（排除error数据）
    if len(valid_data) < target_sample_num:
        raise ValueError(f"有效数据数量({len(valid_data)})小于目标抽取数量({target_sample_num})")
    random_data = random.sample(valid_data, target_sample_num)
    print(f"从{len(valid_data)}条有效数据中随机抽取了{target_sample_num}条")
else:
    # 从所有数据（包括error）中随机抽取
    if len(data) < target_sample_num:
        raise ValueError(f"总数据数量({len(data)})小于目标抽取数量({target_sample_num})")
    random_data = random.sample(data, target_sample_num)
    print(f"从{len(data)}条总数据中随机抽取了{target_sample_num}条")

# 处理随机抽取的数据，格式和原代码保持一致
new_data = []
for item in random_data:
    item["dialog_history"].append({"role": "assistant", "content": item["chosen"]})
    new_data.append({
        "messages": item["dialog_history"],
        "rejected_response": item["rejected"],
    })

# 确保输出目录存在并写入文件
output_dir = os.path.dirname(output_file)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

with open(output_file, "w", encoding="utf-8") as f:
    for item in new_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"随机DPO数据集构建完成！已将{len(new_data)}条随机数据写入{output_file}")

# 额外统计随机数据的分布（用于分析）
random_perfect, random_good, random_not_bad, random_bad, random_error = 0, 0, 0, 0, 0
for item in random_data:
    is_error = False
    if len(item["judge_result"]) != len(item["criteria"]):
        is_error = True
    else:
        counter = Counter(item["judge_result"])
        invalid_numbers = [num for num in counter if num not in {0, 1, 2}]
        if invalid_numbers:
            is_error = True
    
    if is_error:
        random_error += 1
        continue
    
    counter = Counter(item["judge_result"])
    if counter[2]==0 and counter[0]==0 and counter[1]>0:
        random_perfect += 1
    elif counter[2]==0 and counter[1]>=4:
        random_good += 1
    elif counter[2]==0 and counter[1]>0:
        random_not_bad += 1
    else:
        random_bad += 1

print(f"随机抽取数据的分布 - 完美：{random_perfect}，好：{random_good}，不坏不优：{random_not_bad}，坏：{random_bad}，错误：{random_error}")