import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import random
import time
import uuid
import requests
import json
import time
import re
from utils.llm_call import call_llm
from individual_memory.memory_query import query_memories
from utils.tools import history_to_text

strategy_prompt = '''你的回复必须是"({策略}){回复}"的格式，这里的"策略"是指情感支持策略，可以选择以下八种中的一种
1. Question: 提出问题，引导用户继续分享，以获取更多信息
2. Restatement or Paraphrasing: 用自己的话复述或概括用户的表达，以确认理解并让用户感到被倾听。
3. Reflection of Feelings: 关注用户的情绪，指出并回应他们可能正在经历的感受，帮助其更好地觉察自身情绪。
4. Self-disclosure: 适度分享自己的类似经验或感受，以拉近距离，让用户感到不孤单。
5. Affirmation and Reassurance: 给予肯定、认可或安抚，强化用户的价值感，并减轻其紧张、不安情绪。
6. Providing Suggestions: 在理解用户处境后，提供温和、不强迫的建议或可行方向，帮助其找到下一步行动。
7. Information: 提供与用户问题相关的客观资讯、解释或知识，帮助其更清楚地理解现状或做决策。
8. Others: 除上述7种外的其他策略

注意，你的回复必须体现所选策略，比如使用了Question策略，就需要对用户进行提问。
以下是一个例子，User说的话是输入示例，Assistant的回复是输出示例
User: 最近工作上遇到些麻烦，我感到很苦恼。
Assistant: (Question) 听起来你最近承受了不少压力，让你感到苦恼的事情应该已经影响到你的情绪了。我在这里陪你一起梳理、倾听，也可以帮你分析状况、找出下一步的方向。你愿意多说一点吗，到底是哪方面的工作麻烦呢？'''

# def call_npc(player_history, llm_type=None, have_memory:bool=False, user_id:str=None):
def call_npc(player_history, prt_type:str="none", prt_levels:list=['L3','L2','L1'], user_id:str=None, assistant_llm_type=None, base_url: str=None, assistant_api_args: dict=None, retry: int=5):
    history = []
    for mes in player_history:
        if mes["role"]=="user":
            history.append({"role": "user", "content": mes["content"]})
        else:
            history.append({"role": "assistant", "content": mes["content"]})
    # assert history[-1]["role"] == "user", "历史记录中最后一个消息必须是user"
    # history[-1]["content"] += "[指令]你回复的回复必须遵循({策略}){回复}的格式，其中{策略}是情感支持策略，只能选以下八种中的一种：Question、Restatement or Paraphrasing、Reflection of Feelings、Self-disclosure、Affirmation and Reassurance、Providing Suggestions、Information、Others。{回复}是具体的回复内容。比如你的回复可以为：'(Question) 听起来你最近承受了不少压力，让你感到苦恼的事情应该已经影响到你的情绪了。你愿意多说一点吗，到底是哪方面的工作麻烦呢？'。务必记住，回复必须遵循({策略}){回复}的格式，注意策略必须被包含在括号中，不能输出多余内容。"
    
    if prt_type=="personalized":
        assert user_id is not None, "如果需要个性化记忆，则user_id必须填"
        print(f"【提示】启用个性化PRT，启用的PRT层级: {prt_levels}")
        history_text = history_to_text(history)
        memory_dict = {}
        for level in ['L3', 'L2', 'L1']:
            if level in prt_levels:
                memory_dict[level] = query_memories(query_text=history_text, memory_type=level, user_id=user_id)[0]["memory"]
        
        system_prompt = (
            "你是一个智能聊天伙伴，你擅长根据用户的性格和偏好，高情商地和用户聊天，"
            "让用户感到舒适、愉快或得到需要的帮助。你需要参考以下的个性化记忆来调整你的回答，"
            "以更好地满足用户的需求。"
        )
        if memory_dict.get('L3'):
            system_prompt += f"\n<顶层记忆>\n{memory_dict['L3']}\n</顶层记忆>"
        if memory_dict.get('L2'):
            system_prompt += f"\n<多会话记忆>\n{memory_dict['L2']}\n</多会话记忆>"
        if memory_dict.get('L1'):
            system_prompt += f"\n<会话级记忆>\n{memory_dict['L1']}\n</会话级记忆>"
    elif prt_type=="global":
        print("【提示】启用全局PRT")
        global_L3_memory = query_memories(query_text="全局记忆", memory_type="L3", user_id="global")[0]["memory"] # 这里query_text填啥无所谓，user_id填"global"就行
        system_prompt = ("你是一个智能聊天伙伴，你擅长根据用户的性格和偏好，高情商地和用户聊天，让用户感到舒适、愉快或得到需要的帮助。"
                "你需要参考以下的“全局人格总结”来调整你的回答，以更好地满足用户的需求（该全局人格总结大多数用户的共性心理特征、情感偏好与最佳互动指南）。\n"
                "<全局人格总结>\n"
                "{global_L3_memory}\n"
                "</全局人格总结>"
            )
        system_prompt = system_prompt.format(global_L3_memory=global_L3_memory)
        # print(f"system_prompt:\n{system_prompt}")
    elif prt_type=="none":
        print("【提示】未启用PRT")
        system_prompt = '''你是一个智能聊天伙伴，你擅长高情商地和用户聊天，让用户感到舒适、愉快或得到需要的帮助。''' 
    
    # system_prompt += "\n"+strategy_prompt
    
    history = [{"role": "system", "content": system_prompt}] + history
    # history = history_to_text(history) #【ZYX】

    # 正则匹配格式：({策略}){回复}
    # 括号必须是英文括号，策略必须是 strategy_prompt 中列出的 8 种之一
    strategy_pattern = re.compile(
        r'^\s*\('
        r'(Question|Restatement or Paraphrasing|Reflection of Feelings|Self-disclosure|Affirmation and Reassurance|Providing Suggestions|Information|Others)'
        r'\)\s*.+',
        re.DOTALL
    )

    ret = ""
    for attempt in range(1, retry + 1):
        reply = call_llm(history, assistant_llm_type, role="assistant", base_url=base_url, api_args=assistant_api_args)
        ret = reply
        return reply
        if isinstance(reply, str) and strategy_pattern.match(reply.strip()):
            return reply
        else:
            print(f"【警告】NPC回复未匹配预期格式 '({{策略}}){{回复}}'，当前为第 {attempt}/{retry} 次重试。")
            print(f"NPC回复: {reply}")
            if attempt == retry:
                print("【警告】已达到最大重试次数，将返回最后一次回复结果（可能不符合预期格式）。")

    return ret