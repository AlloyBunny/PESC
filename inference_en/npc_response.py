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

strategy_prompt = '''Your reply must follow the format "({strategy}){response}", where "strategy" refers to an emotional support strategy. You may choose one from the following eight types:
1. Question: Ask questions to guide the user to continue sharing and obtain more information.
2. Restatement or Paraphrasing: Rephrase or summarize the user’s expression in your own words to confirm understanding and make the user feel heard.
3. Reflection of Feelings: Focus on the user’s emotions, identify and respond to the feelings they may be experiencing, helping them become more aware of their own emotions.
4. Self-disclosure: Appropriately share your own similar experience or feelings to build closeness and help the user feel less alone.
5. Affirmation and Reassurance: Provide affirmation, recognition, or comfort to reinforce the user’s sense of value and reduce their tension or anxiety.
6. Providing Suggestions: After understanding the user’s situation, offer gentle, non-forceful suggestions or possible directions to help them find their next step.
7. Information: Provide objective information, explanations, or knowledge related to the user’s issue to help them better understand the situation or make decisions.
8. Others: Strategies other than the above seven.

Note: Your response must reflect the chosen strategy. For example, if the chosen strategy is Question, then you need to ask the user a question.
Below is an example. The User’s words are the input sample, and the Assistant’s reply is the output sample.
User: I’ve been having some trouble at work lately, and I feel really distressed.
Assistant: (Question) It sounds like you've been under quite a lot of pressure lately, and the things troubling you at work seem to be affecting your emotions. I’m here to help you sort things out, listen to you, and analyze the situation together to find the next direction. Would you like to share more about what kind of work difficulties you’ve been facing?'''

# def call_npc(player_history, llm_type=None, have_memory:bool=False, user_id:str=None):
def call_npc(player_history, prt_type:str="none", prt_levels:list=['L3','L2','L1'], user_id:str=None, assistant_llm_type=None, base_url: str=None, assistant_api_args: dict=None, retry: int=5):
    history = []
    for mes in player_history:
        if mes["role"]=="user":
            history.append({"role": "user", "content": mes["content"]})
        else:
            history.append({"role": "assistant", "content": mes["content"]})
    # assert history[-1]["role"] == "user", "历史记录中最后一个消息必须是user"
    # history[-1]["content"] += "[Instruction] Your reply must follow the format ({Strategy}){Response}, where {Strategy} is an emotional support strategy, and you may choose only one from the following eight: Question, Restatement or Paraphrasing, Reflection of Feelings, Self-disclosure, Affirmation and Reassurance, Providing Suggestions, Information, Others. {Response} is the actual reply content. For example, your reply could be: '(Question) It sounds like you've been under a lot of pressure lately, and the things troubling you have probably affected your emotions. Would you like to talk more about it? What part of your work has been difficult?' Remember, the reply must follow the format ({Strategy}){Response}, and the strategy must be enclosed in parentheses. Do not output any additional content."

    
    if prt_type=="personalized":
        assert user_id is not None, "如果需要个性化记忆，则user_id必须填"
        print(f"【提示】启用个性化PRT，启用的PRT层级: {prt_levels}")
        history_text = history_to_text(history)
        memory_dict = {}
        for level in ['L3', 'L2', 'L1']:
            if level in prt_levels:
                memory_dict[level] = query_memories(query_text=history_text, memory_type=level, user_id=user_id, lang="en")[0]["memory"]
                
        system_prompt = (
            "You are an intelligent conversational companion, skilled at chatting with users in a way that aligns with their personality and preferences."
            "making them feel comfortable, pleasant, or obtaining the help they need. "
            "You need to refer to the following personalized memories to adjust your responses, so as to better meet the user's needs."
        )
        if memory_dict.get('L3'):
            system_prompt += f"\n<Top-level Memory>\n{memory_dict['L3']}\n</Top-level Memory>"
        if memory_dict.get('L2'):
            system_prompt += f"\n<Multi-session Memory>\n{memory_dict['L2']}\n</Multi-session Memory>"
        if memory_dict.get('L1'):
            system_prompt += f"\n<Session-level Memory>\n{memory_dict['L1']}\n</Session-level Memory>"
    elif prt_type=="global":
        print("【提示】启用全局PRT")
        global_L3_memory = query_memories(query_text="global memory", memory_type="L3", user_id="global", lang="en")[0]["memory"] # 这里query_text填啥无所谓，user_id填"global"就行
        system_prompt = ("You are an intelligent chat companion who excels at chatting with users with high emotional intelligence based on their personality and preferences, making them feel comfortable, happy, or getting the help they need."
                        "You need to refer to the following 'Global Personality Summary' to adjust your responses in order to better meet the user's needs (this global personality summary reflects the common psychological traits, emotional preferences, and best interaction guidelines of most users).\n"
                        "<Global Personality Summary>\n"
                        "{global_L3_memory}\n"
                        "</Global Personality Summary>"
            )
        system_prompt = system_prompt.format(global_L3_memory=global_L3_memory)
        # print(f"system_prompt:\n{system_prompt}")
    elif prt_type=="none":
        print("【提示】未启用PRT")
        system_prompt = '''You are an intelligent chat partner who excels at conversing with users with high emotional intelligence, making them feel comfortable, pleasant, or obtaining the help they need.''' 
    
    # system_prompt += "\n"+strategy_prompt

    history = [{"role": "system", "content": system_prompt}] + history
    history = history_to_text(history) #【ZYX】

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