import hashlib
import json
import re
import numpy as np

def history_to_text(history):
    return "\n".join([f"{turn['role']}: {turn['content']}" for turn in history])

def cal_hash(input_data):
    # Case 1: string
    if isinstance(input_data, str):
        processed_data = input_data.strip()
        # 编码为UTF-8字节流
        data_bytes = processed_data.encode('utf-8')
    # Case 2: list
    elif isinstance(input_data, list):
        json_str = json.dumps(
            input_data, 
            sort_keys=False, 
            separators=(',', ':'), 
            ensure_ascii=False
        )
        data_bytes = json_str.encode('utf-8')
    else:
        # Other types are not supported
        raise TypeError("输入必须是列表或字符串")
    
    # 计算并返回MD5哈希值
    return hashlib.md5(data_bytes).hexdigest()

def cal_scores(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
        f.close()
    single_key = "emo_point"
    multi_keys = ["Empathy", "Informativeness", "Coherence", "Suggestion", 
                  "Understanding", "Helpfulness", "Personalization"]
    trajectory_metrics_keys = ["BEL", "ETV", "ECP_Cx", "ECP_Cy"]
    
    # 初始化总分字典
    total_scores = {key: 0 for key in [single_key] + multi_keys + trajectory_metrics_keys}
    n = len(data)
    if n == 0:
        return total_scores  # 避免除以0
    
    # 累加总分
    for item in data:
        total_scores[single_key] += item[single_key]
        for key in multi_keys:
            total_scores[key] += item["multidim_rating"][key]["score"]
        for key in trajectory_metrics_keys:
            total_scores[key] += item["trajectory_metrics"][key]
    
    # 计算平均值
    return {key: total / n for key, total in total_scores.items()}

def parse_npc_reply(reply: str):
    """
    解析形如 '(Strategy)Response' 的字符串。
    返回：strategy, response
    若解析失败：strategy=None, response=ret
    """
    if not isinstance(reply, str):
        return None, reply

    pattern = re.compile(
        r'^\s*\('
        r'(Question|Restatement or Paraphrasing|Reflection of Feelings|Self-disclosure|Affirmation and Reassurance|Providing Suggestions|Information|Others)'
        r'\)\s*(.*)$',
        re.DOTALL
    )

    match = pattern.match(reply.strip())
    if match:
        strategy = match.group(1)
        response = match.group(2).strip()
        return strategy, response
    else:
        return None, reply

def trajectory_metrics(player_data, num_states=5):
    """
    参考 https://arxiv.org/abs/2511.09003 
    计算轨迹级 Sentient Score 的 BEL、ETV、ECP
    emo_point_list: 原始情绪分数，0~100，可含小数
    num_states: 离散化状态数量（论文中的 N）
    """
    # 0. 从player_data中提取emo_point_list
    def _get_emo_point_list(player_data):
        result = []
        for turn in player_data["history"]:
            if turn["role"] == "user":
                result.append(turn["emotion-point"])
        return result
    
    emo_point_list = _get_emo_point_list(player_data)
    
    # 1. 连续值归一化到 [0,1]
    emo_arr = np.array(emo_point_list, dtype=float)
    emo_norm = emo_arr / 100.0

    # 2. 分箱：离散化成 num_states 个情绪状态
    # 区间如 [0-0.2), [0.2-0.4) ... 最后一个包含1.0
    bins = np.linspace(0, 1, num_states + 1)
    state_seq = np.digitize(emo_norm, bins, right=False)
    state_seq[state_seq > num_states] = num_states  # 修正 digitize 可能越界
    # state_seq ∈ {1,2,...,num_states}

    # =========================== #
    # 下面统一在 state_seq 上计算指标
    # =========================== #

    # ===== BEL =====
    def _cal_BEL_metric(state_seq):
        # 与论文一致，使用状态值（1~num_states）的平均
        return np.mean(state_seq)

    # ===== Transition Matrix =====
    def _cal_transition_matrix(state_seq, num_states):
        M = np.zeros((num_states, num_states))
        for i in range(len(state_seq) - 1):
            s_from = state_seq[i] - 1
            s_to = state_seq[i + 1] - 1
            M[s_from][s_to] += 1
        # 归一化（避免除0）
        for i in range(num_states):
            row_sum = M[i].sum()
            if row_sum > 0:
                M[i] /= row_sum
        return M

    M = _cal_transition_matrix(state_seq, num_states)

    # ===== ETV =====
    def _cal_ETV_metric(state_seq, M, num_states):
        # 权重 ω(e_i)：情绪越低权重越高
        # 论文没具体定义，这里用简单线性：从高到低递减
        # e1 权重大，eN 权重小
        state_weights = np.linspace(1.0, 0.5, num_states)  # 可自定义
        
        etv = 0.0
        for i in range(num_states):
            for j in range(i + 1, num_states):  # j>i，上行
                upward = M[i, j]
                downward = M[j, i]
                delta = upward - downward
                span = (j + 1) - (i + 1)  # 状态差距 ej - ei
                etv += state_weights[i] * span * delta
        return etv

    # ===== ECP =====
    def _cal_ECP_metric(state_seq, M, num_states):
        # 初始状态分布 P(s_{t-1})
        # 用 state_seq 的前 T 个状态统计
        prev_states = state_seq[:-1]
        unique, counts = np.unique(prev_states, return_counts=True)
        P_prev = np.zeros(num_states)
        P_prev[unique - 1] = counts / counts.sum()

        # 状态取值（1~num_states）
        state_values = np.arange(1, num_states + 1)

        # Cx = E[s_{t-1}]
        Cx = np.sum(state_values * P_prev)

        # Cy = E[s_t] = sum_i sum_j e_j * m_ij * P(s_{t-1}=e_i)
        Cy = 0.0
        for i in range(num_states):
            for j in range(num_states):
                Cy += state_values[j] * M[i, j] * P_prev[i]

        return (Cx, Cy)

    BEL = _cal_BEL_metric(state_seq)
    ETV = _cal_ETV_metric(state_seq, M, num_states)
    ECP_Cx, ECP_Cy = _cal_ECP_metric(state_seq, M, num_states)

    # return BEL, ETV, ECP
    player_data["trajectory_metrics"]={
        "BEL": BEL,
        "ETV": ETV,
        "ECP_Cx": ECP_Cx,
        "ECP_Cy": ECP_Cy
    }
    return player_data
