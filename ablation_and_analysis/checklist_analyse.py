import json
from collections import Counter

################################################################
# 如果要换输入文件，只需要改这个
input_file = "./checklist_output/train_1to90/judge/gpt41_pairwise_eval.jsonl"
################################################################

with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
    f.close()
print(len(data))

def check_ok(item):
    if len(item["criteria"]) != len(item["judge_result"]):
        return False
    for cri in item["criteria"]:
        if len(cri) != 3 or cri[2][:2] not in ["性格", "偏好", "规避"]:
            return False
    return True

def result_init():
    result = {
        "性格":{"win": 0,"lose": 0,"tie": 0},
        "偏好":{"win": 0,"lose": 0,"tie": 0},
        "规避":{"win": 0,"lose": 0,"tie": 0},
    }
    return result

def grade(info):
    n = info["win"] + info["lose"] + info["tie"]
    if n==0:
        return 0
    else:
        return 1.0*(info["win"]-info["lose"])/n

def calculate_car(item_data):
    p_positive = item_data["pos_pass"] / item_data["pos"] if item_data["pos"] != 0 else 0 # 计算 P(通过 | gc>0)
    p_negative = item_data["neg_pass"] / item_data["neg"] if item_data["neg"] != 0 else 0 # 计算 P(通过 | gc<=0)
    return p_positive - p_negative

# 1.WinRate统计
global_result = result_init()
# pass_cnt = 0
# fail_cnt = 0
error_cnt = 0
car_result = {
    "性格":{"pos":0,"pos_pass":0,"neg":0,"neg_pass":0},
    "偏好":{"pos":0,"pos_pass":0,"neg":0,"neg_pass":0},
    "规避":{"pos":0,"pos_pass":0,"neg":0,"neg_pass":0}
}

for item in data:
    if not check_ok(item):
        error_cnt += 1
        continue

    counter = Counter(item["judge_result"])
    invalid_numbers = [num for num in counter if num not in {0, 1, 2}]
    if invalid_numbers:
        error_cnt += 1
        continue

    tmp_result = result_init()
    for cri, judge in zip(item["criteria"], item["judge_result"]):
        cri_type = cri[2][:2]
        if cri_type in ["性格", "偏好", "规避"]:
            # print(cri_type)
            if judge==1:
                global_result[cri_type]["win"] += 1
                tmp_result[cri_type]["win"] += 1
            elif judge==2:
                global_result[cri_type]["lose"] += 1
                tmp_result[cri_type]["lose"] += 1
            elif judge==0:
                global_result[cri_type]["tie"] += 1
                tmp_result[cri_type]["tie"] += 1
    
    for cri_type in ["性格", "偏好", "规避"]:
        g_c = grade(tmp_result[cri_type])
        if g_c>0:
            car_result[cri_type]["pos"] += 1
            if (counter[2]==0 and counter[0]==0 and counter[1]>0) or (counter[2]==0 and counter[1]>=4):
                car_result[cri_type]["pos_pass"] += 1
        elif g_c<=0:
            car_result[cri_type]["neg"] += 1
            if (counter[2]==0 and counter[0]==0 and counter[1]>0) or (counter[2]==0 and counter[1]>=4):
                car_result[cri_type]["neg_pass"] += 1
        # print(g_c)
    # if (counter[2]==0 and counter[0]==0 and counter[1]>0) or (counter[2]==0 and counter[1]>=4):
    #     car_result["pos"] += 1
    #     pass_cnt += 1
    # else:
    #     fail_cnt += 1

print(error_cnt)
print(global_result)
print(car_result)

for cri_type in ["性格", "偏好", "规避"]:
    result = global_result[cri_type]
    win_rate = 100.0 * result["win"] / sum(result.values())  # 求和简化分母计算
    print(f"WinRate({cri_type})={win_rate:.2f}%")

for cri_type in ["性格", "偏好", "规避"]:
    car = calculate_car(car_result[cri_type])
    print(f"CAR({cri_type})={car}")

# 2.CAR统计
