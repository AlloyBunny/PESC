import sys
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
from simulator_response import chat_player, player_init, multidim_rating
from npc_response import call_npc
# from inference_zh.llm_calls import print_token_usage_summary
from utils.llm_call import print_token_usage_summary
from dotenv import load_dotenv
import argparse
from utils.tools import cal_scores, trajectory_metrics
import random
import time
import uuid
import json
import requests

def get_valid_prt_levels(prt_levels_str: str) -> list[str]:
    """读取并校验 PRT_LEVELS，非法配置自动返回默认值 [L3, L2, L1]"""
    raw_levels = [l.strip() for l in prt_levels_str.split(",") if l.strip()]
    valid = [l for l in raw_levels if l in ("L3", "L2", "L1")]
    return valid if valid else ["L3", "L2", "L1"]

dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)

file_lock = threading.Lock()

'''
#player structure
player_data = {
    "id": str,
    "emo_point": int,
    "emo_state": str,
    "player": str,
    "scene": str,
    "character": str,
    "topic": str,
    "task": str,
    "history": list
}

#history structure - assistant
history = {
    "role": "assistant",
    "content": str,
    "think": srt#(option)
}

#history structure - user
history = {
    "role": "user",
    "content": str,
    "thinking": str,
    "emotion-point" int,
    "planning":{
          "activity": str,
          "TargetCompletion": str,
          "content": str,
          "analyse": str,
          "change": int
      }
}
'''

#get the current dir load simulator profile with first talk
# current_dir = os.path.dirname(__file__)
# dataset_name = os.getenv("DATASET_NAME")
# profile_path = os.path.join(current_dir, 'profile', dataset_name)
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str) # 必须显式指定
parser.add_argument("--store_file", type=str) # 必须显式指定
parser.add_argument("--prt_type", type=str, default="personalized", choices=["personalized", "global", "none"], help="PRT_TYPE可选值：personalized, global, none（分别为：使用该用户自身的PRT、使用全局用户画像、不启用PRT）")
parser.add_argument("--prt_levels", type=str, default="L3,L2,L1", help="PRT_LEVELS：仅在PRT_TYPE=personalized的情况下生效。支持 L3、L2、L1 三个选项的任意组合（用逗号分隔），顺序不影响结果")
parser.add_argument("--user_llm_type", type=str, default="deepseek", help="USER_LLM_TYPE：用户（玩家）使用的LLM类型，可选值：gpt, deepseek, gemini, local")
parser.add_argument("--assistant_llm_type", type=str, default="gpt", help="ASSISTANT_LLM_TYPE：助手（NPC）使用的LLM类型，可选值：gpt, deepseek, gemini, local, api")
parser.add_argument("--base_url", type=str, default="none", help="LOCAL_MODEL_BASE_URL：本地模型服务的URL")
parser.add_argument("--num_threads", type=int, default=10, help="工作线程数（并发处理的simulator数量），默认10")

parser.add_argument("--HOST", type=str, default=None, help="HOST：API服务的HOST")
parser.add_argument("--user_model_market", type=str, default=None, help="USER_MODEL_MARKET：API模型")
parser.add_argument("--user_USER_NAME", type=str, default=None, help="USER_USER_NAME：API服务的USER_NAME")
parser.add_argument("--user_USER_TOKEN", type=str, default=None, help="USER_USER_TOKEN：API服务的USER_TOKEN")
parser.add_argument("--assistant_model_market", type=str, default=None, help="ASSISTANT_MODEL_MARKET：API模型")
parser.add_argument("--assistant_USER_NAME", type=str, default=None, help="ASSISTANT_USER_NAME：API服务的USER_NAME")
parser.add_argument("--assistant_USER_TOKEN", type=str, default=None, help="ASSISTANT_USER_TOKEN：API服务的USER_TOKEN")

args = parser.parse_args()
dataset_path = args.dataset_path
store_file = args.store_file
prt_type = args.prt_type
prt_levels = get_valid_prt_levels(args.prt_levels)
user_llm_type = args.user_llm_type
assistant_llm_type = args.assistant_llm_type
base_url = args.base_url
num_threads = args.num_threads

host = args.HOST
user_model_market = args.user_model_market
user_USER_NAME = args.user_USER_NAME
user_USER_TOKEN = args.user_USER_TOKEN
assistant_model_market = args.assistant_model_market
assistant_USER_NAME = args.assistant_USER_NAME
assistant_USER_TOKEN = args.assistant_USER_TOKEN

user_api_args = {
    "host": host,
    "model_market": user_model_market,
    "USER_NAME": user_USER_NAME,
    "USER_TOKEN": user_USER_TOKEN,
}
assistant_api_args = {
    "host": host,
    "model_market": assistant_model_market,
    "USER_NAME": assistant_USER_NAME,
    "USER_TOKEN": assistant_USER_TOKEN,
}

print(f"dataset_path: {dataset_path}")
print(f"store_file: {store_file}")
print(f"prt_type: {prt_type}")
print(f"prt_levels: {prt_levels}")
print(f"user_llm_type: {user_llm_type}")
print(f"assistant_llm_type: {assistant_llm_type}")
print(f"base_url: {base_url}")
print(f"num_threads: {num_threads}")
print(f"user_api_args: {user_api_args}")
print(f"assistant_api_args: {assistant_api_args}")

# dataset_path = os.getenv("DATASET_PATH")
with open(dataset_path,"r",encoding="utf-8") as f:
    data = [json.loads(line) for line in f]
    f.close()

os.makedirs(os.path.dirname(store_file), exist_ok=True)
if not os.path.exists(store_file):
    with open(store_file,"w",encoding="utf-8") as f:
        f.close()

#if there exists data in store file, then avoid running with the same simulator profile
existing_keys = set()
with open(store_file,"r",encoding="utf-8") as f:
    for line in f:
        try:
            result = json.loads(line)
            # 创建唯一键：id + topic_main_class + topic_sub_class
            key = (result.get("id", ""), 
                   result.get("topic_main_class", ""), 
                   result.get("topic_sub_class", ""))
            existing_keys.add(key)
        except:
            continue

#talk with human
# testmode = "human"

def process_simulator(simulator: dict):
    """单个simulator的完整对话流程（用于在线程池中并发执行）"""
    #talk with llm
    testmode = "npc"

    #ignore the already talked simulator profile
    # 检查是否已存在相同id、topic_main_class、topic_sub_class的结果
    key = (
        simulator.get("id", ""),
           simulator.get("topic_main_class", ""), 
        simulator.get("topic_sub_class", ""),
    )
    if key in existing_keys:
        # print(f"跳过已处理的数据: ID={simulator.get('id', '')}, topic_main_class={simulator.get('topic_main_class', '')}, topic_sub_class={simulator.get('topic_sub_class', '')}")
        return

    #initialize player with simulator profile
    player = player_init(
        id=simulator["id"],
        topic_main_class=simulator["topic_main_class"],
        topic_sub_class=simulator["topic_sub_class"],
        dataset_path=dataset_path,
    )

    turns = 0
    #talk with llm
    while testmode != "human":
        turns += 1
        #the max conversation turn is 10 round
        if turns > 10:
            break
        
        # (1) 用户发言
        #if it is the first talk, then load the presetting first talk of simulator
        if turns == 1:
            player["history"].append(
                {
                    "role": "user",
                    "content": simulator["first_talk"],
                    "emotion-point": player["emo_point"],
                }
            )
        else:
            #call the simulator response
            player = chat_player(player, user_llm_type=user_llm_type, user_api_args=user_api_args)

            #if simulator says goodble or the emotion-point is smaller than 10 or larger than 100, stop the conversation
            if "再见" in player["history"][-1]["content"] or "拜拜" in player["history"][-1]["content"]:
                break
            
            if (
                player["history"][-1]["emotion-point"] >= 100
                or player["history"][-1]["emotion-point"] < 10
            ):
                break
        
        # (2) NPC发言
        # call the npc response and update history
        # query =  call_npc(player["history"], have_memory=have_memory, user_id=player["id"])
        query = call_npc(
            player["history"],
            prt_type=prt_type,
            prt_levels=prt_levels,
            user_id=player["id"],
            assistant_llm_type=assistant_llm_type,
            base_url=base_url,
            assistant_api_args=assistant_api_args,
        )
        new_state = {"role": "assistant", "content": query}
        player["history"].append(new_state)
    
    #talk with human
    while testmode == "human":
        turns += 1
            #the max conversation turn is 10 round
        if turns > 10:
            break
        #if it is the first talk, then load the presetting first talk of simulator
        if turns == 1:
            player["history"].append(
                {
                    "role": "user",
                    "content": simulator["first_talk"],
                    "emotion-point": player["emo_point"],
                }
            )
        else:
            #call the simulator response
            player = chat_player(player, user_llm_type=user_llm_type, user_args=user_api_args)

            #if simulator says goodble or the emotion-point is smaller than 10 or larger than 100, stop the conversation
            if "再见" in player["history"][-1]["content"] or "拜拜" in player["history"][-1]["content"]:
                break
            
            if (
                player["history"][-1]["emotion-point"] >= 100
                or player["history"][-1]["emotion-point"] < 10
            ):
                break

        query = input(f"User:")
        new_state = {"role": "assistant", "content": query}
        player["history"].append(new_state)

    player = multidim_rating(player, user_llm_type=user_llm_type, user_api_args=user_api_args)
    
    player = trajectory_metrics(player)

    # store the result
    # 确保结果包含topic_main_class和topic_sub_class字段
    result_data = player.copy()
    result_data["topic"] = simulator.get("topic", "")
    result_data["topic_main_class"] = simulator.get("topic_main_class", "")
    result_data["topic_sub_class"] = simulator.get("topic_sub_class", "")

    final_result_data = {
        "index": simulator.get("index", ""),
        **result_data,
    }
    
    # 使用文件锁避免多线程写文件时出现行交错
    with file_lock:
        with open(store_file, "a", encoding="utf-8") as file:
            file.write(json.dumps(final_result_data, ensure_ascii=False) + "\n")
            file.flush()  # 手动刷新缓冲区
    
    print(
        f"完成数据: ID={simulator.get('id', '')}, "
        f"topic_main_class={simulator.get('topic_main_class', '')}, "
        f"topic_sub_class={simulator.get('topic_sub_class', '')}"
    )


def main():
    # 使用线程池并发处理每个simulator
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_sim = {
            executor.submit(process_simulator, simulator): simulator for simulator in data
        }

        for future in as_completed(future_to_sim):
            simulator = future_to_sim[future]
            try:
                future.result()
            except Exception as e:
                print(
                    f"处理数据出错: ID={simulator.get('id', '')}, "
                    f"topic_main_class={simulator.get('topic_main_class', '')}, "
                    f"topic_sub_class={simulator.get('topic_sub_class', '')}, "
                    f"错误: {e}"
                )

    print(f"dataset_path: {dataset_path}")
    print(f"store_file: {store_file}")
    print(f"prt_type: {prt_type}")
    print(f"prt_levels: {prt_levels}")
    print(f"user_llm_type: {user_llm_type}")
    print(f"assistant_llm_type: {assistant_llm_type}")
    print(f"base_url: {base_url}")
    print(f"num_threads: {num_threads}")

    # 计算平均分数
    avg_scores = cal_scores(store_file)
    print("Average Scores:")
    print(json.dumps(avg_scores, indent=2, ensure_ascii=False, sort_keys=False))
    # 程序结束时打印token使用统计
    print_token_usage_summary()


if __name__ == "__main__":
    main()
