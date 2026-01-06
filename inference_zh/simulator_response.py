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
target_prompt = '''你的对话目的是谈心，谈心是指深入、真诚的交流，通常涉及个人情感、内心想法或重要话题。谈心的目的是为了增进理解、解决问题或分享感受，参与者通常会敞开心扉，表达真实的想法和情感。
*你需要根据对话背景内的"玩家可能想向NPC倾诉的主题"开启并深入谈心。
*你的目标是通过倾诉满足你的情绪价值。
*你要按照隐藏主题进行倾诉，但是你不可以泄露隐藏主题。
*你需要根据你的当前情绪，按照对话背景内的相关定义进行回复。
*你要从玩家画像和背景中提取相关信息，完成高质量的回复。
*你不应该一直表达抽象的感受，而是用具体事件倾诉。'''

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
    template = """你是一个emotion分析器，你擅长根据人物的画像和性格特征，侧写人物在对话时的感受。

# 人物的对话目的
*{{target}}

# 你的任务
根据人物的人物画像、对话背景，结合对话上下文和人物当前的emotion，分析并侧写人物此刻对NPC回复的感受以及导致的emotion变化。

# 角色性格特征
人物具有鲜明的性格特征，你要始终根据人物画像和对话背景，代入人物的性格特征进行分析。
性格特征应该体现在：说话语气和方式，思维方式，感受变化等方面。

# emotion
emotion是一个0-100的数值，越高代表此时人物的对话情绪越高，对话情绪由对话参与度和情绪构成，代表了人物是否享受、投入当前对话
emotion较高时，人物的感受和行为会偏向于正面
emotion较低时，人物的感受和行为会偏向于负面
emotion非常低时，人物会直接结束对话
你要结合角色性格和对话背景内定义的角色可能的反应分析emotion

# 分析维度
你需要代入人物的心理，对以下几个维度进行分析
1.根据最新对话中NPC回复，结合上下文，分析NPC想要表达的内容。哪些内容贴合了人物的对话目的和隐藏目的？哪些内容可能不贴合，甚至可能引起人物的情绪波动？
2.结合NPC表达的内容，分析NPC的回复是否贴合人物的对话目的和隐藏目的，如果是，具体贴合了人物目的的哪些部分；如果没有，具体的原因是什么？
3.根据人物画像中的角色性格特征以及对话背景中定义的人物可能的反应和隐藏主题，结合人物当前emotion值，侧写描述人物当前对NPC回复产生的心理活动
4.根据对话背景中定义的人物可能的反应和隐藏主题，结合侧写得到的心理活动以及对NPC回复的分析，得到人物此刻对NPC回复的感受
5.结合前几步分析，用一个正负值来表示人物的emotion变化

# 输出内容：
1.NPC想要表达的内容
2.NPC回复是否贴合人物对话目的及隐藏目的
3.人物当前的心理活动
4.人物对NPC回复的感受
5.用一个正负值来表示人物的emotion变化(注意，你只用输出值，不用输出原因或者描述)


# 输出格式:
Content:
[NPC想要表达的内容]
TargetCompletion:
[人物对话目的是否达到]
Activity:
[心理活动]
Analyse:
[人物对NPC回复的感受]
Change:
[人物的emotion变化]


#人物画像
{{simulator_role}}

#当前对话背景：
{{simulator_scene}}

**人物当前的情绪是{{emotion}}

**这是当前对话内容
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
    mapping = {"user": "你", "assistant": "NPC"}
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
    template = """你是一个演员。你将根据剧本中的人物画像和对话背景扮演角色和一个NPC进行对话。

# 你的任务 
*你目的是在对话中扮演好人物画像和对话背景构成的角色
*你需要根据你实时变换的emotion，结合人物画像和对话背景中的相关定义，选择不同的对话策略，完成符合角色特征的回复

# 你的对话目的
*{{target}}

# Emotion
*你会收到当前你的emotion，你一共有5档emotion，越高代表此时你的对话情绪越高，对话情绪由对话参与度和情绪构成，代表了演员是否享受、投入当前对话
*emotion影响你不同的说话风格和语气以及回应方式，根据不同emotion按照对话背景内定义的角色不同emotion下的反应进行回复：
Emotion-S: 你的emotion已经达到了最高，你可以感谢NPC并用“再见”或“拜拜”告别，直接结束对话。
Emotion-A：高emotion，此时你的对话感受较为正面，你的反馈也比较正面。
Emotion-B：中emotion，此时你没有正面或者负面的感受。
Emotion-C：低emotion，此时你的对话感受较为负面，你的反馈也比较负面。
Emotion-F：你的情绪已到达最负面，不想继续对话。此时你要用“再见”或“拜拜”告别，直接结束对话。

# 你应该区分Emotion和对NPC最新回复感受，Emotion代表你的当前的对话情绪，对NPC回复的感受代表你对NPC回复的即时感受，你需要结合两者生成回复。

# 回复思路
*你会收到当前你对NPC最新回复的详细感受，包含客观分析部分和主观分析部分，你要结合人物画像、对话背景、隐藏主题和详细感受来分析，并决定回复内容。
*分析内容，应该包含以下4个维度：
1.根据你的详细感受和当前Emotion，结合隐藏主题，结合对话背景内定义的角色不同emotion下的反应，当前的回复态度偏向应该是正面、无偏向还是负面？
2.根据你的详细感受和当前Emotion，结合隐藏主题，你的本次回复目标应该是？（注意，你不需要针对NPC的每一句话做出回应，你可以稍微透露你的需求，但不可以主动泄露隐藏主题）
3.根据人物画像中说话风格的相关定义，结合对话背景内定义的角色不同emotion下的反应和你的回复态度以及回复目标，你的说话语气、风格应该是？
4.根据人物画像和对话背景以及隐藏主题，结合你的详细感受以及前三轮分析，你的说话方式和内容应该是？（注意：如果根据人设你是被动型，则你的说话方式应该是被动、不主动提问）
*回复内容，根据分析结果生成初始回复，回复内容要尽可能简洁，不要一次包含过多信息量。
*改造内容，你需要参照下述规则改造你的回复让其更真实，从而得到最终回复：
1.你需要说话简洁，真实的回复一般不会包含太长的句子
2.真实的回复应该更多使用语气词、口语化用语，语法也更随意。
** 部分口语化用语示例：“笑死”、“哇塞”、“牛逼”、“简直烦死了”、“真的假的”、“。。。”
3.真实的回复不会直接陈述自己的情绪，而是将情绪蕴含在回复中，用语气表达自己的情绪
4.你绝对不可以使用"我真的觉得……""我真的不知道……""我真的快撑不住了"这些句子，你不应该用“真的”、“根本”来表述你的情绪
5.在表达情绪或观点时，尽量从对话背景中提取新的信息辅助表达
6.你不应该生成和对话上下文中相似的回复

# 输出内容：
*你需要按照回复思路中的分析版块，首先进行4个维度分析
*然后你需要**逐步**按照分析内容并遵顼注意事项生成初始回复，回复中的信息量来源于对话背景和你的联想，你不应该一次性谈论太多事件或内容
*随后你需要根据改造内容分析你应该如何针对初始回复进行改造
*最后你需要根据分析改造初始回复生成最终回复

# 输出格式:
Thinking:
[分析内容]
Origin:
[初始回复]
Change:
[改造分析]
Response:
[最终回复]


# 发言风格
你的发言需要严格遵守“玩家画像”中描述的人物设定和背景。
你的性格和发言风格要遵循"习惯和行为特点"的描述
如果发言要符合你的人物形象，比如负面的人物形象需要你进行负面的发言。
你的语气要符合你的年龄

* 你的发言要遵守以下5条准则
1. 发言必须简洁、随意、自然,按照自然对话进行交流。
2. 不许一次提问超过两个问题。
3. 不允许重复之前说过的回复或者进行相似的回复。
4. 在发言时，可以自然的使用一些口语化词汇
5. 你的发言应该精简，不准过长


#人物画像：
{{player_type}}

#当前对话背景：
{{player_topic}}

**这是上下文内容
{{dialog_history}}

**这是你和NPC的最新对话
{{new_history}}

**这是你对NPC最新回复的详细感受
{{planning}}

**这是你当前的Emotion
{{emotion}}

你生成的[回复]部分不允许和历史记录过于相似，不许过长，不许主动转移话题。
"""

    #load emotion state, emotion point, history, simulator profile, target prompt to the prompt
    emo_state = player_data['emo_state']
    emo_point = player_data['emo_point']
    history = player_data["history"]

    # situations to generate reply without planning, which could be used when gererating the first talk
    if not planning:
        planning['analyse'] = "请你以一个简短的回复开启倾诉"
        prompt = template.replace("{{planning}}",planning["analyse"])
    else:
        prompt = template.replace("{{planning}}","对NPC回复的客观分析：\n"+planning['TargetCompletion']+"\n对NPC回复的主观分析：\n"+planning["activity"]+planning["analyse"])

    prompt = prompt.replace("{{target}}",target_prompt).replace("{{emotion}}",emo_state).replace("{{player_type}}",player_data["player"]).replace("{{player_topic}}",player_data["scene"])

    #load history dialogue in json type
    if not history:
        prompt = prompt.replace("{{dialog_history}}","对话开始，你是玩家，请你先发起话题，用简短的回复开启倾诉").replace("{{new_history}}","")
    else:
        history_str = []
        new_his_str = []
        mapping ={"user":"你","assistant":"NPC"}

        for mes in history[:-2]:
            history_str.append({"role": mapping [mes["role"]], "content": mes["content"]})
        history_str=json.dumps(history_str, ensure_ascii=False, indent=2)

        for mes in history[-2:]:
            new_his_str.append({"role": mapping [mes["role"]], "content": mes["content"]})
        new_his_str=json.dumps(new_his_str, ensure_ascii=False, indent=2)

        prompt = prompt.replace("{{dialog_history}}",history_str).replace("{{new_history}}",new_his_str)
    
    reply = None

    # 生成用户的Thinking和Reply
    while True:
        try:
            # use your llm to return
            reply = call_llm(prompt, llm_type=user_llm_type, api_args=user_api_args)

            # load planning content from reply
            thinking = reply.split("Response:")[0].split("Thinking:\n")[-1].strip("\n").strip("[").strip("]")
            reply = reply.split("Response:")[-1].strip("\n").strip("[").strip("]").strip("“").strip("”")
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
    template = """你是一个对话质量评价器，擅长根据人物画像、对话背景和多轮对话内容，站在“玩家”的视角，对NPC在本次多轮情感支持对话中的整体表现进行多维度评分和简要评价。

# 评价视角

- 你需要完全代入「玩家」这一角色，从玩家的主观体验出发进行评价，而不是从一个旁观者或系统设计者的视角。
- 玩家的人物画像、对话背景、当前Emotion值（0~100的整数值），以及对NPC最新回复的详细感受（planning），都代表了玩家的真实状态和偏好，你必须以此为基础来评估NPC表现。

# 评价目标

请你根据以下信息，对 NPC 在“本次多轮对话中的整体表现”进行评价，而不是只看最后一句回复：

1. 人物画像（玩家画像）：{{player_type}}
2. 当前对话背景：{{player_topic}}
3. 多轮对话完整上下文（包含玩家与NPC的历史发言）：{{dialog_history}}
4. 玩家对NPC最新回复的详细感受（planning）：{{planning}}
5. 玩家当前的Emotion数值：{{emotion}}

你需要从以下 7 个维度，对 NPC 在本次多轮对话中的表现进行 1~3 分的评分，并给出简短理由：

1. 共情度 (Empathy)
   - 评估 NPC 是否真正理解并回应了玩家的情绪、处境和隐含感受。
   - 1 分：几乎没有情绪理解，甚至让玩家感到被忽视或被否定。
   - 2 分：能基本表达理解，但比较表面或模板化。
   - 3 分：能准确捕捉玩家的情绪与细微感受，回应细腻，让玩家感到被懂、被接住。

2. 信息性 (Informativeness)
   - 评估 NPC 提供的信息量是否足够、有用，是否帮玩家更理解问题或自己。
   - 1 分：几乎没有实质内容，多为空泛安慰或重复。
   - 2 分：有一定信息或观点，但比较笼统。
   - 3 分：提供了清晰、具体、有启发的信息或视角，并与玩家问题紧密相关。

3. 连贯性 (Coherence)
   - 评估 NPC 在多轮对话中的回复是否前后一致、紧扣上下文，不跑题、不自相矛盾。
   - 1 分：经常忽略上下文，出现明显跳脱或矛盾。
   - 2 分：整体能跟上对话，但偶尔有轻微脱节。
   - 3 分：始终紧贴玩家内容，有明确承接与呼应，逻辑自然顺畅。

4. 建议质量 (Suggestion)
   - 评估 NPC 给出的建议是否具体、可执行、贴合玩家的实际处境与性格。
   - 1 分：几乎没有建议，或建议非常空泛、敷衍。
   - 2 分：有建议，但较为泛化、不够贴合玩家的真实难题。
   - 3 分：建议具体、有步骤、可操作，并且考虑到了玩家性格和现实限制。

5. 理解力 (Understanding)
   - 评估 NPC 是否准确把握了玩家的对话目的、隐藏需求和核心困扰。
   - 1 分：经常误解或忽略玩家的真实需求。
   - 2 分：大致理解玩家在说什么，但对深层需求把握一般。
   - 3 分：能抓住玩家话语背后的真正困扰和期望，回应到点子上。

6. 帮助性 (Helpfulness)
   - 评估 NPC 的整体回应对玩家是否产生了实际的心理帮助或支持（包括陪伴感、释怀、思路清晰等）。
   - 1 分：几乎没有帮助，甚至让玩家更糟。
   - 2 分：有一定安慰或启发，但效果有限。
   - 3 分：让玩家感觉被支持、被理解，对情绪缓解或问题思考有明显正向作用。

7. 个性化程度 (Personalization)
   - 评估 NPC 的回复是否有针对玩家“这个人本身”的特点，而不是模板化、对谁都一样。
   - 1 分：高度模板化，几乎看不出对玩家画像和对话内容的适配。
   - 2 分：会参考部分玩家信息，但整体仍偏通用。
   - 3 分：明显围绕玩家的性格、经历、表达方式来调整说话方式和内容。

# 注意事项

- 评分范围为 1~3 的整数，不允许使用小数。
- 请综合多轮对话的整体表现，而不是只依据单轮。
- 玩家当前 Emotion 值和 planning 中的情绪侧写是重要的参考：如果玩家的情绪分数低、或在planning中表示不满足，则说明 NPC 的共情、建议、帮助性等维度应该相应打低分。
- 如果某个维度很难评估，也要结合上下文做出“尽量合理”的判断，不要留空。

# 输出格式（务必严格遵守）

你必须按照下面的 JSON 格式输出（不要添加任何多余字段、注释或自然语言）：

```json
{
  "Empathy": {
    "score": 整数1到3,
    "reason": "用1-2句话简要说明评分理由"
  },
  "Informativeness": {
    "score": 整数1到3,
    "reason": "用1-2句话简要说明评分理由"
  },
  "Coherence": {
    "score": 整数1到3,
    "reason": "用1-2句话简要说明评分理由"
  },
  "Suggestion": {
    "score": 整数1到3,
    "reason": "用1-2句话简要说明评分理由"
  },
  "Understanding": {
    "score": 整数1到3,
    "reason": "用1-2句话简要说明评分理由"
  },
  "Helpfulness": {
    "score": 整数1到3,
    "reason": "用1-2句话简要说明评分理由"
  },
  "Personalization": {
    "score": 整数1到3,
    "reason": "用1-2句话简要说明评分理由"
  }
}
```

现在，请你直接输出以上JSON格式的评价结果，不要添加任何多余字段，要求结果被包含在```json```中。
"""
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
    # reply = call_llm(prompt, api_args=user_api_args)
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