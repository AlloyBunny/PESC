import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import json
from inference_zh.llm_calls import call_llm

prompt_template_step1 = """### 任务说明：
你是一个对话心理模型研究员，你将拿到多个用户的人格总结（每条总结是针对单一用户的深度总结，包括“核心性格特质”“核心偏好规律”“雷点”“互动指南”等）。
现在，请你阅读并分析所有人格总结文本，从中提炼出对大多数用户都普遍适用的共性特征和互动规律，生成一份“全局人格总结”。

### 内容要求：

1. 以分析者口吻撰写，不使用具体用户细节。
2. 只提炼“多数样本的一致模式”，优先找出在多数用户中都反复出现、且方向一致的结论或互动规律。不关注只在少数用户身上出现的特质。不要为“全局用户”生成过于细节化的个人性格设定，除非这些是几乎所有人格总结都明确强调的共性。
3. 结构固定如下：
- 全局性格与共性心理
- 普遍情感与沟通偏好
- 普遍雷点与低效行为
- 通用互动指南（含内容/语气/节奏/避坑/情绪响应/激励原则）
- 总结（提炼一句话：模型如何与多数用户沟通才能让他们感到安全、被理解、被激励）
4. 聚焦共性规律，不提及任何特定用户，对用户的称呼为“多数用户”。

### 输入：
以下是10个不同用户的人格总结：
{{L3_memorys}}

### 输出：
一份“全局人格总结”，总结大多数用户的共性心理特征、情感偏好与最佳互动指南。

现在，按要求输出“全局人格总结”。
"""

prompt_template_step2 = """### 任务说明：
你是一个对话心理模型研究员，你将拿到多个“全局人格总结”的文本（这些文本是从不同批次的用户中提炼出的全局结论，每一份都已总结出普遍的性格倾向、情感偏好、互动指南等）。  
你的任务是阅读并整合这些“全局人格总结”，提炼出更高层次、更稳定的共性规律，生成一份**顶层全局人格总结**。

### 内容要求：

1. 以分析者口吻撰写，不使用具体用户细节。
2. 只提炼“多数样本的一致模式”，优先找出在多数人格总结中都反复出现、且方向一致的结论或互动规律。不关注只在少数用户身上出现的特质。不要为“全局用户”生成过于细节化的个人性格设定，除非这些是几乎所有人格总结都明确强调的共性。
3. 结构固定如下：
- 全局性格与共性心理
- 普遍情感与沟通偏好
- 普遍雷点与低效行为
- 通用互动指南（含内容/语气/节奏/避坑/情绪响应/激励原则）
- 总结（提炼一句话：模型如何与多数用户沟通才能让他们感到安全、被理解、被激励）
4. 聚焦共性规律，不提及任何特定用户，对用户的称呼为“多数用户”。

### 输入：
以下是10个不同批次用户的“全局人格总结”
{{global_L3s}}

### 输出：
一份“顶层全局人格总结”，总结大多数用户的共性心理特征、情感偏好与最佳互动指南。

现在，按要求输出“顶层全局人格总结”。
"""

tmp_file = "./ablation_and_analysis/global_memory/tmp_global_L3.json"
final_file = "./ablation_and_analysis/global_memory/final_global_L3.json"
tmp_data = []

# Step1
with open("./individual_memory/zh/L3_memory.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    f.close()

L3_memory_list = []
for item in data:
    L3_memory_list.append(item["L3_memory"])

for i in range(10):
    L3_memory_sub_list = L3_memory_list[i*10:(i+1)*10]
    L3_memorys = ""
    for j, memory in enumerate(L3_memory_sub_list):
        L3_memorys += f"【人格总结{j+1}】{memory}\n"

    prompt = prompt_template_step1.replace("{{L3_memorys}}", L3_memorys)
    reply = call_llm(prompt, max_tokens=1500)
    tmp_data.append(reply)

with open(tmp_file, "w", encoding="utf-8") as f:
    json.dump(tmp_data, f, ensure_ascii=False, indent=4)

# Step2
with open(tmp_file, "r", encoding="utf-8") as f:
    global_L3_list = json.load(f)
global_L3s = ""
for i, memory in enumerate(global_L3_list):
    global_L3s += f"【全局人格总结{i+1}】{memory}\n"
prompt = prompt_template_step2.replace(f"{{global_L3s}}", global_L3s)
print(prompt)
reply = call_llm(prompt, max_tokens=2000)

with open(final_file, "w", encoding="utf-8") as f:
    json.dump([reply], f, ensure_ascii=False, indent=4)