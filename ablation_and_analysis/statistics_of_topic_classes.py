import sys
import os
import json
import unicodedata
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
args = parser.parse_args()
input_file = args.input_file

################################################################
# 如果要换输入文件，只需要改这个
# input_file = "./inference_output/llama31_hybird_dpo_30.jsonl"
################################################################

# ===== 计算中英文混排显示宽度的工具函数 =====
def display_len(s: str) -> int:
    """按终端显示宽度计算字符串长度：中文全角算2，英文半角算1"""
    length = 0
    for ch in str(s):
        if unicodedata.east_asian_width(ch) in ("F", "W"):  # 全角 / 宽字符
            length += 2
        else:
            length += 1
    return length

def pad(s, width, align="left") -> str:
    """
    按显示宽度补空格对齐
    align: 'left' | 'right' | 'center'
    """
    s = str(s)
    real_len = display_len(s)
    if real_len >= width:
        return s  # 超出就不截断了，直接返回

    pad_len = width - real_len
    if align == "left":
        return s + " " * pad_len
    elif align == "right":
        return " " * pad_len + s
    elif align == "center":
        left = pad_len // 2
        right = pad_len - left
        return " " * left + s + " " * right
    else:
        return s + " " * pad_len

# ===== 你的原始逻辑 =====

with open(input_file, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

stats = {}
for item in data:
    topic = item.get("topic_main_class", "未知主题")
    emo_point = item.get("emo_point")

    if emo_point is None:
        print(f"⚠️  跳过缺失emo_point的记录（主题：{topic}）")
        continue
    if not isinstance(emo_point, (int, float)):
        print(f"⚠️  跳过无效emo_point（主题：{topic}，分数：{emo_point}，类型：{type(emo_point)}）")
        continue

    if topic not in stats:
        stats[topic] = {"sum": 0.0, "cnt": 0}

    stats[topic]["sum"] += emo_point
    stats[topic]["cnt"] += 1

# ===== 表格打印部分（已按中英文混排对齐） =====

# 每列的“显示宽度”
COL_TOPIC = 18
COL_CNT = 10
COL_AVG = 15
TOTAL_WIDTH = COL_TOPIC + 1 + COL_CNT + 1 + COL_AVG

print("=" * TOTAL_WIDTH)
print(
    pad("主题类别", COL_TOPIC, "left"),
    pad("样本数量", COL_CNT, "right"),
    pad("平均emo_point", COL_AVG, "right"),
)
print("-" * TOTAL_WIDTH)

for topic in sorted(stats.keys()):
    total_score = stats[topic]["sum"]
    sample_count = stats[topic]["cnt"]
    average_score = total_score / sample_count

    print(
        pad(topic, COL_TOPIC, "left"),
        pad(sample_count, COL_CNT, "right"),
        pad(f"{average_score:.4f}", COL_AVG, "right"),
    )

print("-" * TOTAL_WIDTH)
total_all_samples = sum(stats[topic]["cnt"] for topic in stats)
total_all_score = sum(stats[topic]["sum"] for topic in stats)
overall_average = total_all_score / total_all_samples if total_all_samples > 0 else 0.0

print(
    pad("总计", COL_TOPIC, "left"),
    pad(total_all_samples, COL_CNT, "right"),
    pad(f"{overall_average:.4f}", COL_AVG, "right"),
)
print("=" * TOTAL_WIDTH)
