import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import json
import uuid
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from inference.llm_calls import call_llm
from collections import Counter, defaultdict
from tqdm import tqdm
import argparse

L1_prompt = """## 任务
你是一位专注于对话行为分析的心理专家，擅长从用户与AI助手的互动细节中，精准提炼个性化认知。
请基于一次完整多轮对话（含情绪分数、思考记录、规划内容），按要求输出5类反思结论，每类用1句中文描述（共4-6句），直接呈现结论（不出现“我”“用户”等主语），需深度结合“情绪变化”“思考逻辑”推导，避免表面描述。
输出需覆盖以下核心维度，每类均需关联对话中的具体信息（如情绪分、思考内容）：
1. 话题概述（用简练的一句话概括对话核心主题及用户核心诉求）
2. 对话摘要（用一小段话概括对话内容和用户的情绪变化）
3. 性格特征反思（提炼稳定行为倾向，需搭配思考记录/表达习惯的例子）
4. 用户偏好反思（总结触发情绪上升的互动方式，需搭配情绪分变化的例子）
5. 潜在雷点反思（总结可能触发情绪下降或抵触的互动方式，无则说明“暂未体现”并举例支撑）
6. 额外洞察反思（提炼对后续互动有价值的隐性线索，如行动意愿、关注重点等，需搭配规划/对话细节的例子）

## 输入格式
输入为JSON格式，包含用户（user）与AI助手（assistant）的完整对话记录，字段含role、content、emotion-point（情绪分数）、thinking（用户思考记录）、planning（用户规划内容）。

## 输出格式
输出为文本格式，按以下要点分点清晰呈现：
1. 话题概述：xxx
2. 对话摘要：xxx
3. 性格特征反思：xxx，例如：xxx（需引用thinking/表达细节）
4. 用户偏好反思：xxx，例如：xxx（需引用emotion-point变化）
5. 潜在雷点反思：xxx，例如：xxx（需引用对话反馈，无则写“暂未体现，如对空泛建议未出现情绪下降”）
6. 额外洞察反思：xxx，例如：xxx（需引用planning/对话隐性信息）

## 任务执行
请严格遵循指令、结构及“情绪-行为-认知”关联原则，根据下面提供的**[输入]**，生成对应的**[输出]**。

[输入]

```json
{input}
```

[输出]
"""

L2_prompt = """## 任务
你是一位资深跨场景行为分析师，擅长从多份单次对话反思中，提炼用户的稳定共性认知（排除单次偶然行为）。
你将得到多份“用户个性化反思”（每份对应一次对话），需聚焦“性格、偏好、雷点”的重复模式，按要求整合为跨对话共性结论，输出内容需覆盖所有输入中的关键共性，且明确标注支撑场景（如“事业话题”“健身话题”）。
输出需包含以下维度，每类均需说明“在哪些场景中重复出现”及“对应的情绪/行为证据”：
1. 性格共性反思（稳定贯穿多个场景的性格特质，排除仅单一场景出现的特质）
2. 偏好共性反思（触发情绪正向反馈的互动方式，需关联情绪分上升的共性逻辑）
3. 潜在雷点列表（触发情绪下降或抵触的互动方式，按“雷点+场景+情绪表现”逐条列出）
4. 额外洞察总结（整合跨场景的隐性线索，如行动意愿倾向、关注重点变化等）

## 输入格式
输入为多段文本，每一段对应一份“用户个性化反思”，包含“话题概述”，“对话摘要”，“性格特征反思”，“用户偏好反思”，“潜在雷点反思”，“额外洞察反思”六部分内容。

## 输出格式参考
### 偏好共性反思
结论：用户的跨话题偏好共性是：[具体偏好，如“仅接受具体可落地的建议”]，在[场景1（如事业）、场景2（如健身）、场景3（如人际关系）]中均有体现；
证据：1. 第1份反思（事业话题）：对“设立小目标”的具体建议情绪+10，对空泛安慰无响应；2. 第2份反思（健身话题）：对“每周3次有氧”的具体计划情绪+8，对“多运动”的模糊建议无反馈；3. 第3份反思（人际关系话题）：对“每周1次聚餐”的具体方案情绪+5，符合共性逻辑。

### 性格共性反思
结论：用户的稳定性格特质是：[具体性格，如“自负且表达直接”]，在[场景1、场景2、场景3]中均有体现；
证据：1. 第1份反思（事业话题）：提及“表达时侧重强调自身努力”，体现自负；2. 第2份反思（健身话题）：提及“说‘自己能坚持计划’”，符合自负特质；3. 第3份反思（人际关系话题）：提及“回复不绕弯子”，与“表达直接”共性一致。

### 潜在雷点列表
1. [雷点1，如“反感空泛鼓励”]：在第1份反思（事业话题）中，因“你一定可以的”空泛鼓励情绪-5；在第2份反思（学习话题）中，因“加油就能进步”无细节安慰情绪-4，均触发负面反馈；
2. [雷点2，如“回避依赖他人的建议”]：在第3份反思（家庭话题）中，因“建议找家人帮忙”情绪-6；在第4份反思（工作话题）中，因“推荐求助同事”情绪-3，需持续规避；
3. [无则标注“暂未发现跨场景雷点，仅单一场景提及XX，暂不列为共性”]

### 额外洞察总结
[整合跨场景隐性线索，如“对所有话题均有较强行动意愿，在事业、健身、学习话题中均提及‘想实施建议’”“关注重点始终围绕‘自身能力提升’，在多场景中均强调‘靠自己推进’”]

## 任务执行
请严格遵循指令、结构及“共性筛选-场景关联-证据支撑”原则，根据下面提供的**[输入]**，生成对应的**[输出]**。

[输入]

{input}

[输出]
"""

L3_prompt = """## 任务
你是一位资深心理行为分析师，擅长从跨场景共性认知中提炼用户的核心人格与互动规律。
你将得到多份用户“跨对话共性反思”（含性格、偏好、雷点、额外洞察），需进一步递归整合，排除次要共性、聚焦贯穿所有场景的核心认知，最终形成可直接指导支持者互动的“顶层个性化记忆”。
输出要求包含以下内容，需覆盖输入中所有关键信息，且仅保留跨所有/多数场景的核心结论：
1. 核心性格特质（稳定且贯穿所有场景的性格，排除仅少数场景出现的特质）
2. 核心偏好规律（对互动方式/内容的稳定偏好，需关联情绪正向反馈的共性逻辑）
3. 核心雷点清单（所有场景中均需规避的触发点，明确情绪负面反馈的具体表现）
4. 顶层互动指南（基于核心认知提炼的、可直接落地的支持者互动原则）

## 输入格式
输入是多段文本，每一段是一份用户“跨对话共性反思”，包含“偏好共性反思”“性格共性反思”“潜在雷点列表”“额外洞察总结”四部分内容。

## 输出格式参考
### 核心性格特质
结论：用户的核心性格是[1-2个贯穿所有场景的稳定特质]，在[场景1、场景2、场景3等所有关键场景]中均有体现；
依据：1. 第1份共性反思（场景1+2+3）：提及[对应性格特质]，如“自负，强调自身努力”；2. 第2份共性反思（场景4+5+6）：提及[同一性格特质]，如“自负，认为自己能安排好事务”；3. 无场景例外，所有共性反思均指向该特质。

### 核心偏好规律
结论：用户的核心偏好是[贯穿所有场景的互动/内容偏好]，情绪正向反馈（+X分）均源于此逻辑；
依据：1. 第1份共性反思：[场景1-3]中，对“具体可落地建议”情绪+10/+5，对空泛建议无响应；2. 第2份共性反思：[场景4-6]中，对“具体计划/步骤”情绪+8/+6，对模糊指导无反馈；3. 所有情绪上升案例均符合“具体性”逻辑，无例外。

### 核心雷点清单
1. [雷点1，贯穿所有场景]：在第1份共性反思[场景2]中，因“提及依赖他人”情绪-10；在第2份共性反思[场景5]中，因“建议寻求帮助”情绪-8，均触发负面反馈；
2. [雷点2，贯穿多数场景]：在第1份共性反思[场景3]中，因“空泛鼓励”情绪-5；在第2份共性反思[场景6]中，因“无细节安慰”情绪-4，需持续规避；
3. [其他核心雷点，需注明覆盖场景范围]：...

### 顶层互动指南
1. 内容原则：所有支持回复需提供“具体细节/步骤/计划”，避免“加油”“多努力”等空泛表述，如推荐方法时需明确“每天1小时XX”而非“多花时间做XX”；
2. 语气原则：需先认可用户“自身能力/努力”，再输出建议，贴合其核心性格，如开头可提“你之前靠自己的安排推进过XX，这次也可以从XX具体步骤入手”；
3. 避坑原则：绝对避免“建议依赖他人”“空泛无细节”两类表述，若需提及协作，需转化为“你主导的XX协作（如每周1次和同事同步进度）”，弱化“依赖”属性；
4. 情绪响应原则：若用户情绪分低于40，优先用“具体小步骤”引导（如“先从XX1个小行动开始，试试？”），而非追问原因，贴合其偏好逻辑。

## 任务执行
现在，请严格遵循以上所有指令、结构和创作原则，根据下面提供的**[输入]**，生成对应的**[输出]**。

[输入]

{input}

[输出]
"""

def get_L1_memory(dialog: dict, llm_type: str):
    """生成L1记忆"""
    dialog_str = json.dumps(dialog, ensure_ascii=False, indent=4)
    prompt = L1_prompt.format(input=dialog_str)
    return call_llm(prompt, llm_type=llm_type)

def get_L2_memory(L1_memory_list: list, llm_type: str):
    """生成L2记忆"""
    L1_memorys = ""
    for i in range(len(L1_memory_list)):
        L1_memorys += f"【用户个性化反思{i+1}】\n{L1_memory_list[i]['L1_memory']}\n\n"
    prompt = L2_prompt.format(input=L1_memorys)
    return call_llm(prompt, llm_type=llm_type, max_tokens=4000)

def get_L3_memory(L2_memory_list: list, llm_type: str):
    """生成L3记忆"""
    L2_memorys = ""
    for i, item in enumerate(L2_memory_list):
        L2_memorys += f"【跨对话共性反思{i+1}】\n{item['L2_memory']}\n\n"
    prompt = L3_prompt.format(input=L2_memorys)
    return call_llm(prompt, llm_type=llm_type, max_tokens=4000)

def process_L1_memory_task(task_data, llm_type: str):
    """处理单个L1记忆任务的函数"""
    try:
        item, index = task_data
        L1_memory_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f'{item["id"]}_{item["topic_main_class"]}_{item["topic_sub_class"]}_{item["topic"]}'))
        L1_memory = get_L1_memory(item["history"], llm_type=llm_type)
        time.sleep(1)  # 避免API调用过于频繁
        
        result = {
            "index": index,
            "memory_id": L1_memory_id,
            "id": item["id"],
            "topic_main_class": item["topic_main_class"],
            "topic_sub_class": item["topic_sub_class"],
            "topic": item["topic"],
            "L1_memory": L1_memory
        }
        return result
    except Exception as e:
        print(f"处理L1记忆任务时出错: {e}")
        return None

def process_L2_memory_task(task_data, llm_type: str):
    """处理单个L2记忆任务的函数"""
    try:
        user_id, L1_memories, index = task_data
        L1_individual_memory_id_list = [item["memory_id"] for item in L1_memories]
        L2_memory_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, '\n'.join(L1_individual_memory_id_list)))
        L2_memory = get_L2_memory(L1_memories, llm_type=llm_type)
        time.sleep(1)  # 避免API调用过于频繁
        
        result = {
            "index": index,
            "memory_id": L2_memory_id,
            "id": user_id,
            "L1_memory_id_list": L1_individual_memory_id_list,
            "L2_memory": L2_memory
        }
        return result
    except Exception as e:
        print(f"处理L2记忆任务时出错: {e}")
        return None

def process_L3_memory_task(task_data, llm_type: str):
    """处理单个L3记忆任务的函数"""
    try:
        user_id, L2_memories, index = task_data
        L2_individual_memory_id_list = [item["memory_id"] for item in L2_memories]
        L3_memory_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, '\n'.join(L2_individual_memory_id_list)))
        L3_memory = get_L3_memory(L2_memories, llm_type=llm_type)
        time.sleep(0.5)  # 避免API调用过于频繁
        
        result = {
            "index": index,
            "memory_id": L3_memory_id,
            "id": user_id,
            "L2_memory_id_list": L2_individual_memory_id_list,
            "L3_memory": L3_memory
        }
        return result
    except Exception as e:
        print(f"处理L3记忆任务时出错: {e}")
        return None

def main():
    print(call_llm("你好"))
    time.sleep(3)
    """主函数，使用多线程处理记忆生成"""
    print("🚀 开始个性化记忆生成流程...")
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="deepseek")
    parser.add_argument("--num_threads", type=int, default=10, help="线程数量，默认为10")
    parser.add_argument("--input_file", type=str, default="dataset/train.jsonl", help="输入文件路径")
    args = parser.parse_args()
    
    # 线程数配置
    MAX_WORKERS = args.num_threads
    llm_type = args.model_type
    input_file = args.input_file
    
    # 读取数据
    with open(input_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]


    
    # 创建总体进度条
    overall_pbar = tqdm(total=3, desc="总体进度", unit="阶段", 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
    
    # 处理L1记忆生成
    print("开始生成L1记忆...")
    try:
        with open("individual_memory/L1_memory.json", "r", encoding="utf-8") as f:
            L1_memory_list = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        L1_memory_list = []
    
    L1_memory_id_list = [item["memory_id"] for item in L1_memory_list]
    
    # 准备L1任务
    L1_tasks = []
    for i, item in enumerate(data):
        L1_memory_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f'{item["id"]}_{item["topic_main_class"]}_{item["topic_sub_class"]}_{item["topic"]}'))
        if L1_memory_id not in L1_memory_id_list:
            L1_tasks.append((item, i))
        else:
            print(f"跳过一个任务，L1_memory_id={L1_memory_id}")
    
    # 使用线程池处理L1任务
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        L1_futures = {executor.submit(process_L1_memory_task, task, llm_type=llm_type): task for task in L1_tasks}
        
        # 创建L1进度条
        L1_pbar = tqdm(total=len(L1_tasks), desc="L1记忆生成", unit="个", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for future in as_completed(L1_futures):
            result = future.result()
            if result is not None:
                L1_memory_list.append(result)
                L1_pbar.set_postfix({"当前": result['id'], "已完成": len(L1_memory_list)})
            L1_pbar.update(1)
        
        L1_pbar.close()
    
    # 对L1_memory_list按照index从小到大排序
    L1_memory_list.sort(key=lambda x: x["index"])
    with open("individual_memory/L1_memory.json", "w", encoding="utf-8") as f:
        json.dump(L1_memory_list, f, ensure_ascii=False, indent=4)
    
    print("✅ L1_memory生成完成")
    overall_pbar.update(1)
    overall_pbar.set_postfix({"当前阶段": "L2记忆生成"})

    # 处理L2记忆生成
    print("🔄 开始生成L2记忆...")
    try:
        with open("individual_memory/L2_memory.json", "r", encoding="utf-8") as f:
            L2_memory_list = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        L2_memory_list = []
    
    L2_memory_id_list = [item["memory_id"] for item in L2_memory_list]
    
    # 按用户ID分组L1记忆，并按照原始逻辑处理（每7个生成一个L2）
    L2_tasks = []
    L1_individual_memory_dict = defaultdict(list)
    
    for item in L1_memory_list:
        L1_individual_memory_dict[item["id"]].append(item)
        # 当某个用户的L1记忆达到7个时，立即准备生成L2记忆
        if len(L1_individual_memory_dict[item["id"]]) >= 7:
            L1_individual_memory_id_list = [item["memory_id"] for item in L1_individual_memory_dict[item["id"]]]
            L2_memory_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, '\n'.join(L1_individual_memory_id_list)))
            if L2_memory_id not in L2_memory_id_list:
                index = max(item["index"] for item in L1_individual_memory_dict[item["id"]])
                L2_tasks.append((item["id"], L1_individual_memory_dict[item["id"]].copy(), index))
            # 清空已处理的数据，重新开始积累
            L1_individual_memory_dict[item["id"]] = []
    
    # 处理剩余的L1记忆（不足7个的）
    for user_id, memories in L1_individual_memory_dict.items():
        if memories:  # 如果还有剩余的记忆
            L1_individual_memory_id_list = [item["memory_id"] for item in memories]
            L2_memory_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, '\n'.join(L1_individual_memory_id_list)))
            if L2_memory_id not in L2_memory_id_list:
                index = max(item["index"] for item in memories)
                L2_tasks.append((user_id, memories, index))
    
    # 使用线程池处理L2任务
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        L2_futures = {executor.submit(process_L2_memory_task, task, llm_type=llm_type): task for task in L2_tasks}
        
        # 创建L2进度条
        L2_pbar = tqdm(total=len(L2_tasks), desc="L2记忆生成", unit="个", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for future in as_completed(L2_futures):
            result = future.result()
            if result is not None:
                L2_memory_list.append(result)
                L2_pbar.set_postfix({"当前": result['id'], "已完成": len(L2_memory_list)})
            L2_pbar.update(1)
        
        L2_pbar.close()
    
    # 对L2_memory_list按照index从小到大排序
    L2_memory_list.sort(key=lambda x: x["index"])
    with open("individual_memory/L2_memory.json", "w", encoding="utf-8") as f:
        json.dump(L2_memory_list, f, ensure_ascii=False, indent=4)
    
    print("✅ L2_memory生成完成")
    overall_pbar.update(1)
    overall_pbar.set_postfix({"当前阶段": "L3记忆生成"})

    # 处理L3记忆生成
    print("🔄 开始生成L3记忆...")
    try:
        with open("individual_memory/L3_memory.json", "r", encoding="utf-8") as f:
            L3_memory_list = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        L3_memory_list = []
    
    L3_memory_id_list = [item["memory_id"] for item in L3_memory_list]
    L2_individual_memory_dict = defaultdict(list)
    
    # 按用户ID分组L2记忆
    for item in L2_memory_list:
        L2_individual_memory_dict[item["id"]].append(item)
    
    # 准备L3任务
    L3_tasks = []
    for user_id, memories in L2_individual_memory_dict.items():
        L2_individual_memory_id_list = [item["memory_id"] for item in memories]
        L3_memory_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, '\n'.join(L2_individual_memory_id_list)))
        if L3_memory_id not in L3_memory_id_list:
            index = max(item["index"] for item in memories)
            L3_tasks.append((user_id, memories, index))
    
    # 使用线程池处理L3任务
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        L3_futures = {executor.submit(process_L3_memory_task, task, llm_type=llm_type): task for task in L3_tasks}
        
        # 创建L3进度条
        L3_pbar = tqdm(total=len(L3_tasks), desc="L3记忆生成", unit="个", 
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for future in as_completed(L3_futures):
            result = future.result()
            if result is not None:
                L3_memory_list.append(result)
                L3_pbar.set_postfix({"当前": result['id'], "已完成": len(L3_memory_list)})
            L3_pbar.update(1)
        
        L3_pbar.close()
    
    # 对L3_memory_list按照index从小到大排序
    L3_memory_list.sort(key=lambda x: x["index"])
    with open("individual_memory/L3_memory.json", "w", encoding="utf-8") as f:
        json.dump(L3_memory_list, f, ensure_ascii=False, indent=4)
    
    print("✅ L3_memory生成完成")
    overall_pbar.update(1)
    overall_pbar.set_postfix({"当前阶段": "完成"})
    overall_pbar.close()
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"🎉 所有记忆生成任务完成！总耗时: {total_time:.2f}秒")
    print(f"📊 统计信息:")
    print(f"   - L1记忆: {len(L1_memory_list)}个")
    print(f"   - L2记忆: {len(L2_memory_list)}个") 
    print(f"   - L3记忆: {len(L3_memory_list)}个")
    # print(f"   - 平均每个L1记忆耗时: {total_time/len(L1_memory_list):.2f}秒")

if __name__ == "__main__":
    main()
