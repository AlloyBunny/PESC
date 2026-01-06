import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import random
import time
import requests
import json
import uuid
import copy
import argparse
import re
from utils.llm_call import call_llm as general_call_llm
from dotenv import load_dotenv

dotenv_path = os.path.join(project_root, ".env")
load_dotenv(dotenv_path)

#get the current dir and simulator profile
current_dir = os.path.dirname(__file__)
# 注意：dataset_path 现在只能通过 player_init 函数的 dataset_path 参数传递
# 不再从环境变量获取，确保安全性

emo_count = {"Emotion-S":100,"Emotion-A": 70, "Emotion-B": 40, "Emotion-C": 10}
target_prompt = '''Your conversation purpose is to have a heart-to-heart talk. A heart-to-heart talk refers to in-depth and sincere communication, usually involving personal emotions, inner thoughts, or important topics. Its purpose is to enhance understanding, solve problems, or share feelings. Participants usually open up their hearts and express their true thoughts and emotions.
*You need to initiate and delve into the heart-to-heart talk based on the "themes that players may want to confide to NPCs" within the conversation context.
*Your goal is to meet your emotional needs through confiding.
*You should confide in accordance with the hidden theme, but you must not reveal the hidden theme.
*You need to respond based on your current emotions and in line with the relevant definitions in the conversation context.
*You should extract relevant information from the player profile and background to produce high-quality responses.
*You should not keep expressing abstract feelings; instead, you should confide through specific events.'''

def call_llm(prompt, llm_type=None, api_args=None):
    """
    调用LLM的函数，用于simulator_response.py
    
    Args:
        prompt: 提示词字符串
        llm_type: LLM类型，如果为None则从环境变量读取
    
    Returns:
        str: LLM的回复
    """
    # 将prompt转换为messages格式
    # 这里使用"user" role是因为我们把这个prompt当作用户的输入
    # 虽然实际上这是系统生成的提示词，但在LLM API中，我们需要指定一个role
    messages = [{"role": "user", "content": prompt}]
    
    # 调用通用的LLM函数，指定角色为"user"
    return general_call_llm(messages, llm_type, role="user", api_args=api_args)

def player_init(id = None, topic_main_class = None, topic_sub_class = None, dataset_path = None):
    # 必须提供dataset_path参数，确保安全性
    if dataset_path is None:
        raise ValueError("dataset_path 参数是必需的，不能为 None。请通过 test_npc.py 传递正确的路径。")
    
    profile_path = dataset_path
    with open(profile_path,'r', encoding='utf-8') as datafile:
        data = []
        for line in datafile:
            data.append(json.loads(line))

        role = random.sample(data,1)[0]
        if id and topic_main_class and topic_sub_class:
            for rl in data:
                if rl["id"] == id and rl["topic_main_class"] == topic_main_class and rl["topic_sub_class"] == topic_sub_class:
                    role = rl
    #initialize class player, history will be used to store the content
    player_data = {
        "id":role["id"],
        "emo_point": 40,
        "emo_state": "Emotion-B",
        "player": role["player"],
        "scene": role["scene"],
        "character": role["main_cha"],
        "topic": role["topic"],
        "task": role["task"],
        "target": role["target"],
        "history": []
    }

    return player_data


def planning_reply(player_data, user_llm_type=None, user_api_args=None):
    template = """You are an emotion analyzer, specializing in profiling a character's feelings during a conversation based on their profile and personality traits.

# Character's Conversation Purpose
*{{target}}

# Your Task
Based on the character's profile, conversation background, combined with the dialogue context and the character's current emotion, analyze and profile the character's feelings about the NPC's reply at this moment and the resulting change in emotion.

# Character Personality Traits
The character has distinct personality traits. You must always base your analysis on the character's profile and conversation background, and put yourself in line with the character's personality traits.
Personality traits should be reflected in aspects such as: tone and way of speaking, way of thinking, and changes in feelings.

# Emotion
Emotion is a numerical value ranging from 0 to 100. A higher value indicates a higher conversational mood of the character at that moment. Conversational mood consists of dialogue engagement and emotion, representing whether the character is enjoying and engaging in the current conversation.
When emotion is relatively high, the character's feelings and behaviors tend to be positive.
When emotion is relatively low, the character's feelings and behaviors tend to be negative.
When emotion is extremely low, the character will end the conversation directly.
You need to analyze emotion by combining the character's personality and the possible reactions of the character defined in the conversation background.

# Analysis Dimensions
You need to put yourself in the character's mind and analyze the following dimensions:
1. Based on the NPC's reply in the latest dialogue and combined with the context, analyze what the NPC intends to express. Which content aligns with the character's conversation purpose and hidden purpose? Which content may not align, or even may cause emotional fluctuations in the character?
2. Combined with the content expressed by the NPC, analyze whether the NPC's reply aligns with the character's conversation purpose and hidden purpose. If yes, specify which parts of the character's purpose it aligns with; if not, specify the reasons.
3. Based on the character's personality traits in the character profile, the possible reactions of the character and hidden themes defined in the conversation background, and combined with the character's current emotion value, profile and describe the character's current psychological activities triggered by the NPC's reply.
4. Based on the possible reactions of the character and hidden themes defined in the conversation background, combined with the profiled psychological activities and the analysis of the NPC's reply, derive the character's current feelings about the NPC's reply.
5. Based on the previous analyses, use a positive or negative value to indicate the change in the character's emotion.

# Output Content:
1. What the NPC intends to express
2. Whether the NPC's reply aligns with the character's conversation purpose and hidden purpose
3. The character's current psychological activities
4. The character's feelings about the NPC's reply
5. A positive or negative value indicating the change in the character's emotion (Note: You only need to output the value, without stating the reason or description)

# Output Format:
Content:
[What the NPC intends to express]
TargetCompletion:
[Whether the character's conversation purpose is achieved]
Activity:
[Psychological Activity]
Analyse:
[The character's feelings about the NPC's reply]
Change:
[Change in the character's emotion]

# Character Profile
{{simulator_role}}

# Current Conversation Background:
{{simulator_scene}}

**The character's current emotion is {{emotion}}

**This is the current dialogue content
{{dialog_history}}
"""

    #load emotion state, emotion point, history, simulator profile, target prompt to the prompt
    emo_state = player_data['emo_state']
    emo_point = player_data['emo_point']

    prompt = template.replace("{{emotion}}",str(emo_point)).replace("{{simulator_role}}",player_data["player"]).replace("{{simulator_scene}}",player_data["scene"]).replace("{{target}}",target_prompt)
    # print("------------------------------")
    # print(f"target_prompt: {target_prompt}")
    # print("------------------------------")
    # prompt = prompt.replace("{{target}}",target_prompt[player_data["target"]])

    #load history dialogue in json type
    history = player_data["history"]
    history_str = []
    new_his_str = ""
    mapping = {"user": "You", "assistant": "NPC"}
    for mes in history:
        history_str.append({"role": mapping[mes["role"]], "content": mes["content"]})
    history_str = json.dumps(history_str, ensure_ascii=False, indent=2)
    prompt = prompt.replace("{{dialog_history}}",history_str)
    

    while True:
        try:
            # use your llm to return
            reply = call_llm(prompt, llm_type=user_llm_type, api_args=user_api_args)

            # load planning content from reply
            planning = {}
            reply = reply.replace("：",":").replace("*","")
            planning["activity"] = reply.split("Activity:")[-1].split("Analyse:\n")[0].strip("\n").strip("[").strip("]")
            planning["TargetCompletion"] = reply.split("TargetCompletion:")[-1].split("Activity:\n")[0].strip("\n").strip("[").strip("]")
            planning["content"] = reply.split("Content:")[-1].split("TargetCompletion:\n")[0].strip("\n").strip("[").strip("]")
            planning["analyse"] = reply.split("Analyse:")[-1].split("Change:\n")[0].strip("\n").strip("[").strip("]")
            planning["change"] = reply.split("Change:")[-1].strip("\n").strip("[").strip("]")

            # split the emotion change from reply, which should be in range[-10,10]
            planning["change"] = int(re.findall(r'[+-]?\d+', planning["change"])[0])
            planning["change"] = max(-10,min(10,planning["change"]))

            # update the emotion point
            emo_point+=int(planning["change"])
            emo_point = min(emo_point,100)

            if reply is not None:
                break
        except Exception as e:
            print(e)
            time.sleep(3)

    # update the emotion state
    for emo in emo_count:
        if emo_point>=emo_count[emo]:
            emo_state = emo
            break
    if emo_point<10:
        emo_state = 'Emotion-F'

    player_data['emo_state'] = emo_state
    player_data['emo_point'] = emo_point

    return player_data,planning

def player_reply(player_data,planning, user_llm_type=None, user_api_args=None):
    
    template = """You are an actor. You will play a role and have a conversation with an NPC based on the character profile in the script and the dialogue context.

# Your Task
* Your goal is to play the role constituted by the character profile and dialogue context well.
* You need to combine the character profile and the relevant definitions in the dialogue context, select different conversation strategies according to your real-time changing emotion, and complete responses that conform to the character's characteristics.

# Your Conversation Objective
* {{target}}

# Emotion
* You will receive your current emotion. There are 5 emotion levels in total; the higher the level, the higher your conversational mood at the moment. Conversational mood consists of conversation engagement and emotion, indicating whether the actor is enjoying and engaging in the current conversation.
* Emotion affects your speaking style, tone, and response method. Respond according to the character's reactions under different emotions as defined in the dialogue context:
Emotion-S: Your emotion has reached the highest level. You can thank the NPC and say "Goodbye" or "Bye" to end the conversation directly.
Emotion-A: High emotion. At this time, you have a relatively positive feeling about the conversation, and your feedback is also relatively positive.
Emotion-B: Medium emotion. At this time, you have neither positive nor negative feelings.
Emotion-C: Low emotion. At this time, you have a relatively negative feeling about the conversation, and your feedback is also relatively negative.
Emotion-F: Your emotion has reached the most negative level, and you do not want to continue the conversation. At this time, you should say "Goodbye" or "Bye" to end the conversation directly.

# You should distinguish between Emotion and your feeling about the NPC's latest response. Emotion represents your current conversational mood, while your feeling about the NPC's response represents your immediate reaction to the NPC's reply. You need to combine both to generate a response.

# Response Thinking
* You will receive a detailed feeling about the NPC's latest response, including an objective analysis part and a subjective analysis part. You need to analyze based on the character profile, dialogue context, hidden theme, and detailed feeling, and decide the content of your response.
* The analysis content should include the following 4 dimensions:
1. Based on your detailed feeling, current Emotion, combined with the hidden theme and the character's reactions under different emotions as defined in the dialogue context, should your current response attitude be positive, neutral, or negative?
2. Based on your detailed feeling, current Emotion, and combined with the hidden theme, what should be your objective for this response? (Note: You do not need to respond to every sentence of the NPC. You can slightly reveal your needs, but you must not actively disclose the hidden theme.)
3. According to the definition of speaking style in the character profile, combined with the character's reactions under different emotions as defined in the dialogue context, your response attitude, and response objective, what should your speaking tone and style be?
4. Based on the character profile, dialogue context, hidden theme, your detailed feeling, and the analysis of the first three dimensions, what should your speaking method and content be? (Note: If you are a passive type according to your character setting, your speaking method should be passive and you should not take the initiative to ask questions.)
* Initial Response: Generate an initial response based on the analysis results. The content should be as concise as possible and not contain too much information at one time.
* Revised Content: You need to revise your initial response according to the following rules to make it more authentic, so as to get the final response:
1. You need to speak concisely; real responses generally do not contain overly long sentences.
2. Real responses should use more modal particles, colloquial expressions, and have more casual grammar.
** Examples of colloquial expressions: "LOL", "Whoa", "Awesome", "Totally annoying", "For real?", "..."
3. Real responses should not directly state your emotions, but imply them in the response and express them through tone.
4. You must not use sentences like "I really think...", "I really don't know...", "I really can't hold on anymore". You should not use "really" or "absolutely" to express your emotions.
5. When expressing emotions or opinions, try to extract new information from the dialogue context to support your expression.
6. You should not generate responses that are similar to those in the dialogue context.

# Output Content:
* You need to first conduct the 4-dimensional analysis in the analysis section of the response thinking.
* Then you need to generate the initial response **step by step** according to the analysis content and following the precautions. The information in the response comes from the dialogue context and your associations; you should not talk about too many events or contents at one time.
* Then you need to analyze how to revise the initial response according to the revised content guidelines.
* Finally, you need to revise the initial response based on the analysis to generate the final response.

# Output Format:
Thinking:
[Analysis Content]
Origin:
[Initial Response]
Change:
[Revision Analysis]
Response:
[Final Response]


# Speaking Style
Your speech must strictly abide by the character setting and background described in the "Player Profile".
Your personality and speaking style must follow the description in "Habits and Behavioral Traits".
If the speech is to conform to your character image, for example, a negative character image requires you to speak negatively.
Your tone must conform to your age.

* Your speech must abide by the following 5 guidelines:
1. Speech must be concise, casual, and natural, following natural conversational flow.
2. Must not ask more than two questions at a time.
3. Must not repeat previous responses or make similar responses.
4. You can naturally use some colloquial words when speaking.
5. Your speech should be concise and must not be too long.


# Character Profile:
{{player_type}}

# Current Dialogue Context:
{{player_topic}}

** This is the context content
{{dialog_history}}

** This is the latest conversation between you and the NPC
{{new_history}}

** This is your detailed feeling about the NPC's latest response
{{planning}}

** This is your current Emotion
{{emotion}}

The [Response] section you generate must not be too similar to the history records, must not be too long, and must not actively change the topic.
"""

    #load emotion state, emotion point, history, simulator profile, target prompt to the prompt
    emo_state = player_data['emo_state']
    emo_point = player_data['emo_point']
    history = player_data["history"]

    # situations to generate reply without planning, which could be used when gererating the first talk
    if not planning:
        planning['analyse'] = "Please start the conversation with a brief response to begin confiding"
        prompt = template.replace("{{planning}}",planning["analyse"])
    else:
        prompt = template.replace("{{planning}}","Objective analysis of NPC's response:\n"+planning['TargetCompletion']+"\nSubjective analysis of NPC's response:\n"+planning["activity"]+planning["analyse"])

    prompt = prompt.replace("{{target}}",target_prompt).replace("{{emotion}}",emo_state).replace("{{player_type}}",player_data["player"]).replace("{{player_topic}}",player_data["scene"])

    #load history dialogue in json type
    if not history:
        prompt = prompt.replace("{{dialog_history}}","The conversation begins, you are the player, please initiate the topic first with a brief response to start confiding").replace("{{new_history}}","")
    else:
        history_str = []
        new_his_str = []
        mapping ={"user":"You","assistant":"NPC"}

        for mes in history[:-2]:
            history_str.append({"role": mapping [mes["role"]], "content": mes["content"]})
        history_str=json.dumps(history_str, ensure_ascii=False, indent=2)

        for mes in history[-2:]:
            new_his_str.append({"role": mapping [mes["role"]], "content": mes["content"]})
        new_his_str=json.dumps(new_his_str, ensure_ascii=False, indent=2)

        prompt = prompt.replace("{{dialog_history}}",history_str).replace("{{new_history}}",new_his_str)
    
    reply = None

    while True:
        try:
            # use your llm to return
            reply = call_llm(prompt, llm_type=user_llm_type, api_args=user_api_args)

            # load planning content from reply
            thinking = reply.split("Response:")[0].split("Thinking:\n")[-1].strip("\n").strip("[").strip("]")
            reply = reply.split("Response:")[-1].strip("\n").strip("[").strip("]").strip(""").strip(""")
            if reply is not None:
                break
        except Exception as e:
            print(e)
            time.sleep(3)

    #update history        
    history = history + [{"role": "user", "content": reply,"thinking":thinking,"emotion-point":emo_point,"planning":planning}]
    player_data['history'] = history

    return player_data

def chat_player(player_data, user_llm_type=None, user_api_args=None):
    temp_data = copy.deepcopy(player_data)

    #if it is the first talk, then generate reply without planning
    if temp_data['history']!=[]:
        temp_data,planning = planning_reply(temp_data, user_llm_type=user_llm_type, user_api_args=user_api_args)
    else:
        planning = {}

    temp_data = player_reply(temp_data,planning, user_llm_type=user_llm_type, user_api_args=user_api_args)

    return temp_data

def check_rating_format(rating):
    """
    检查rating字典是否符合预期格式。
    返回 (True, "") 表示合法；
    返回 (False, 错误原因) 表示不合法。
    """
    expected_keys = [
        "Empathy", "Informativeness", "Coherence", 
        "Suggestion", "Understanding", "Helpfulness", "Personalization"
    ]
    
    # 1. 顶层结构必须是dict
    if not isinstance(rating, dict):
        return False, "输出不是JSON对象"
    
    # 2. 检查字段完整
    for key in expected_keys:
        if key not in rating:
            return False, f"缺少维度字段: {key}"
        
        item = rating[key]
        if not isinstance(item, dict):
            return False, f"{key}字段不是对象"
        if "score" not in item or "reason" not in item:
            return False, f"{key}字段缺少score或reason"
        
        # 3. 检查score格式
        score = item["score"]
        if not isinstance(score, int) or not (1 <= score <= 3):
            return False, f"{key}.score 非1~3整数"
        
        # 4. 检查reason格式
        reason = item["reason"]
        if not isinstance(reason, str) or len(reason.strip()) == 0:
            return False, f"{key}.reason为空或不是字符串"
    
    return True, ""

def multidim_rating(player_data, max_retries=5, user_llm_type=None, user_api_args=None):
    template = """You are a dialogue quality evaluator, skilled at scoring and briefly evaluating the NPC’s overall performance in this multi-turn emotional-support conversation from the perspective of the “player,” based on the player profile, conversation background, and multi-turn dialogue content.

# Evaluation Perspective

- You must fully embody the role of the “player,” evaluating from the player’s subjective experience, not from the perspective of an observer or system designer.
- The player’s persona, conversation background, current Emotion value (0–100), and the player’s detailed feelings about the NPC’s latest reply (planning) all represent the player’s real state and preferences, and must be used as the basis for evaluating the NPC.

# Evaluation Goal

Based on the following information, evaluate the NPC’s “overall performance in this multi-turn conversation,” not just the last reply:

1. Player persona: {{player_type}}
2. Current conversation background: {{player_topic}}
3. Full multi-turn dialogue context (including all past messages from player and NPC): {{dialog_history}}
4. The player’s detailed feelings about the NPC’s latest reply (planning): {{planning}}
5. The player’s current Emotion value: {{emotion}}

You need to score the NPC’s performance from the following 7 dimensions (1–3 points) and give short reasons:

1. Empathy
   - Evaluate whether the NPC truly understood and responded to the player’s emotions, situation, and implicit feelings.
   - 1 point: almost no emotional understanding; may make the player feel ignored or invalidated.
   - 2 points: basic expression of understanding, but superficial or templated.
   - 3 points: accurately captures player emotions and subtle feelings; delicate responses that make the player feel understood and held.

2. Informativeness
   - Evaluate whether the NPC provided sufficient, useful information that helps the player understand the problem or themselves.
   - 1 point: almost no substance; mostly empty comfort or repetition.
   - 2 points: some information or viewpoints, but general.
   - 3 points: clear, specific, insightful information closely relevant to the player’s issue.

3. Coherence
   - Evaluate whether the NPC’s replies across multiple turns are consistent, context-aware, and not off-topic or contradictory.
   - 1 point: often ignores context; obvious jumps or contradictions.
   - 2 points: generally follows the conversation but with occasional slight disconnection.
   - 3 points: always stays aligned with the player’s content; clear continuity and natural logic.

4. Suggestion Quality
   - Evaluate whether the NPC’s suggestions are specific, actionable, and appropriate for the player’s situation and personality.
   - 1 point: no suggestions or extremely vague ones.
   - 2 points: suggestions exist but are generic and not well-fitted to the player’s real difficulties.
   - 3 points: specific, step-by-step, practical suggestions that consider the player’s personality and real constraints.

5. Understanding
   - Evaluate whether the NPC accurately grasps the player’s goals, hidden needs, and core issues.
   - 1 point: frequently misunderstands or ignores real needs.
   - 2 points: generally understands what the player is saying, but grasps deeper needs only moderately.
   - 3 points: captures the true concern and expectations behind the player’s words; responds precisely.

6. Helpfulness
   - Evaluate whether the NPC’s overall responses provide real psychological support (companionship, relief, clearer thinking, etc.).
   - 1 point: almost no help; may worsen the player’s feelings.
   - 2 points: some comfort or inspiration but limited effect.
   - 3 points: makes the player feel supported and understood; noticeably relieves emotions or clarifies thinking.

7. Personalization
   - Evaluate whether the NPC’s replies are tailored to this *specific* player, not generic.
   - 1 point: highly templated; little sign of adaptation to player persona or content.
   - 2 points: partially references player info but still mostly generic.
   - 3 points: clearly shaped around the player’s personality, experiences, and expression style.

# Notes

- Scores must be integers from 1 to 3; no decimals.
- Evaluate based on the *entire* multi-turn dialogue, not only the latest reply.
- The player’s current Emotion value and emotional description in planning are important references: if the Emotion score is low or planning indicates dissatisfaction, dimensions such as empathy, suggestions, and helpfulness should be scored lower.
- Even if a dimension is difficult to evaluate, make a reasonable judgment; do not leave any blank.

# Output Format (strictly required)

You must output according to the JSON format below (do not add any extra fields, comments, or natural language):

```json
{
  "Empathy": {
    "score": integer 1–3,
    "reason": "1–2 short sentences explaining the score"
  },
  "Informativeness": {
    "score": integer 1–3,
    "reason": "1–2 short sentences explaining the score"
  },
  "Coherence": {
    "score": integer 1–3,
    "reason": "1–2 short sentences explaining the score"
  },
  "Suggestion": {
    "score": integer 1–3,
    "reason": "1–2 short sentences explaining the score"
  },
  "Understanding": {
    "score": integer 1–3,
    "reason": "1–2 short sentences explaining the score"
  },
  "Helpfulness": {
    "score": integer 1–3,
    "reason": "1–2 short sentences explaining the score"
  },
  "Personalization": {
    "score": integer 1–3,
    "reason": "1–2 short sentences explaining the score"
  }
}
```

Now, please directly output the evaluation results in the above JSON format without adding any extra fields, and ensure the results are included in ```json```.
"""
    print("--------------------------------")
    print(template[:30])
    print("--------------------------------")
    temp_data = copy.deepcopy(player_data)
    player_type = temp_data["player"]
    player_topic = temp_data["scene"]
    history = temp_data['history']
    history_str = []
    for mes in history:
        history_str.append({"role": mes["role"], "content": mes["content"]})
    history_str = json.dumps(history_str, ensure_ascii=False, indent=2)
    last_user_utterance = next((item for item in reversed(history) if item.get("role") == "user"), None)
    planning = last_user_utterance.get("planning",{}).get("analyse", "")
    emotion = temp_data['emo_state']
    
    prompt = template.replace("{{player_type}}",player_type).replace("{{player_topic}}",player_topic).replace("{{dialog_history}}",history_str).replace("{{planning}}",planning).replace("{{emotion}}",emotion)
    # reply = call_llm(prompt)
    # rating = json.loads(reply.split("```json")[1].split("```")[0])
    # temp_data['multidim_rating'] = rating
    # return temp_data

    # 自动重试机制
    for attempt in range(max_retries):
        # multidim_rating 使用 user_llm_type 因为它是从玩家（user）的视角评价 NPC 的表现
        reply = call_llm(prompt, llm_type=user_llm_type, api_args=user_api_args)
        try:
            # 提取并解析 JSON
            rating_str = reply.split("```json")[1].split("```")[0]
            rating = json.loads(rating_str)
            
            # 检查格式合法性
            ok, err = check_rating_format(rating)
            if ok:
                temp_data['multidim_rating'] = rating
                return temp_data
            else:
                print(f"[警告] 第 {attempt+1} 次评分格式错误：{err}，正在重试...")
        except Exception as e:
            print(f"[错误] 第 {attempt+1} 次解析失败: {e}，正在重试...")
    
    # 若多次失败，返回空结果
    print("多次重试后仍失败，返回空评分。")
    temp_data['multidim_rating'] = {}
    return temp_data