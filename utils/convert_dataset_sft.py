import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import json
from individual_memory.memory_query import query_memories
from utils.tools import history_to_text

# --- 配置输入和输出文件路径 ---
input_file_path = 'dataset/val_sft_index.jsonl'       # 你的原始数据集.json文件
output_file_path = f'dataset/val_sft.jsonl' # 转换后用于训练的.jsonl文件
# ---------------------------------

def convert_dataset_sft(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    result_list = []
    for item in data:
        index = item['index']
        history = item['history']
        history_text = history_to_text(history)
        
        L3_memory = query_memories(query_text=history_text, memory_type="L3", user_id=item["id"])[0]["memory"]
        try: # L2_memory可能不存在
            L2_memory = query_memories(query_text=history_text, memory_type="L2", index_before=index, user_id=item["id"])[0]["memory"]
        except (IndexError, KeyError, TypeError):
            L2_memory = "暂无"
        try: # L1_memory可能不存在
            L1_memory = query_memories(query_text=history_text, memory_type="L1", index_before=index, user_id=item["id"])[0]["memory"]
        except (IndexError, KeyError, TypeError):
            L1_memory = "暂无"
        
        system_prompt = ("你是一个智能聊天伙伴，你擅长根据用户的性格和偏好，高情商地和用户聊天，让用户感到舒适、愉快或得到需要的帮助。"
                            "你需要参考以下的个性化记忆来调整你的回答，以更好地满足用户的需求。\n"
                            "<顶层记忆>\n"
                            "{L3_memory}\n"
                            "</顶层记忆>\n"
                            "<多会话记忆>\n"
                            "{L2_memory}\n"
                            "</多会话记忆>\n"
                            "<会话级记忆>\n"
                            "{L1_memory}\n"
                            "</会话级记忆>"
                        )
        system_prompt = system_prompt.format(L3_memory=L3_memory, L2_memory=L2_memory, L1_memory=L1_memory)
        result = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]
        for i, turn in enumerate(history):
            new_item = {
                "role": turn['role'],
                "content": turn['content'],
            }
            if turn['role'] == 'assistant':
                if i+1<len(history) and int(history[i+1].get("planning",{}).get("change", None)) > 0:
                    new_item['loss'] = True
                else:
                    new_item['loss'] = False
            result.append(new_item)
        
        result = {
            "messages": result
        }
        result_list.append(result)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for result in result_list:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

if __name__=="__main__":
    convert_dataset_sft(input_file_path, output_file_path)