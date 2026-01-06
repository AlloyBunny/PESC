import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import json
# from individual_memory.memory_query import query_memories
from utils.tools import history_to_text
import argparse

# --- 配置路径和选项 ---
parser = argparse.ArgumentParser()
parser.add_argument("--input_file_path", type=str, default='dataset/train.jsonl')
parser.add_argument("--output_file_path", type=str, default='dataset/train_sft.jsonl')
# parser.add_argument("--character_id_start", type=int, default=1)
# parser.add_argument("--character_id_end", type=int, default=10)
parser.add_argument("--only_emo_up", action='store_true', default=False) # 是否只选用令sage分数提高的utterance
parser.add_argument("--language", type=str, default="zh") # 语言
args = parser.parse_args()

input_file_path = args.input_file_path
output_file_path = args.output_file_path
only_emo_up = args.only_emo_up
language = args.language
prompt_zh = '''你是一个智能聊天伙伴，你擅长高情商地和用户聊天，让用户感到舒适、愉快或得到需要的帮助。
你的回复必须是"({策略}){回复}"的格式，这里的"策略"是指情感支持策略，可以选择以下八种中的一种
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

prompt_en = '''You are an intelligent chat partner who excels at conversing with users with high emotional intelligence, making them feel comfortable, pleasant, or obtaining the help they need.
Your reply must follow the format "({strategy}){response}", where "strategy" refers to an emotional support strategy. You may choose one from the following eight types:
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



def convert_dataset_sft(input_file_path, output_file_path):
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    
    result_list = []
    for item in data:
        if item.get("emo_point", 0) < 70: # 只要高sage分数的数据
            continue
        index = item['index']
        history = item['history']
        history_text = history_to_text(history)

        system_prompt = prompt_zh if language == "zh" else prompt_en
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
            if turn['role'] == 'assistant' and only_emo_up: # 如果只选用令sage分数提高的utterance，则需要加loss标签
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