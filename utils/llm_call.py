import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import requests
import json
import time
import threading
from typing import List, Dict, Union
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 全局变量用于记录token使用情况
total_prompt_tokens = 0
total_completion_tokens = 0
total_prompt_cost = 0.0
total_completion_cost = 0.0
gpt_total_prompt_tokens = 0
gpt_total_completion_tokens = 0
deepseek_total_prompt_tokens = 0
deepseek_total_completion_tokens = 0

# 全局锁，保证多线程环境下统计变量更新和读取的原子性
_token_stats_lock = threading.Lock()

# GPT-4.1价格设置
GPT_PROMPT_TOKEN_PRICE = 2 / 1000000  # $2.5 per 1M tokens = $0.0000025 per token
GPT_COMPLETION_TOKEN_PRICE = 4 / 1000000  # $10 per 1M tokens = $0.00001 per token

# Gemini价格设置
GEMINI_PROMPT_TOKEN_PRICE = 1.25 / 1000000  # $1.25 per 1M tokens = $0.00000126 per token
GEMINI_COMPLETION_TOKEN_PRICE = 5 / 1000000  # $5.00 per 1M tokens = $0.00000504 per token

# DeepSeek价格设置
DEEPSEEK_PROMPT_TOKEN_PRICE = 2.0 / 1000000  # $2.0 per 1M tokens = $0.000002 per token
DEEPSEEK_COMPLETION_TOKEN_PRICE = 8.0 / 1000000  # $8.0 per 1M tokens = $0.000008 per token

def call_GPT(messages: Union[List[Dict[str, str]], str], model_name: str = None, max_tokens: int = 1000, max_retries: int = 20) -> str:
    """
    调用OpenAI GPT模型的函数
    
    Args:
        messages: 消息列表，格式为[{"role": "user/assistant/system", "content": "消息内容"}]
        model_name: 模型名称，如果为None则从环境变量读取
        max_retries: 最大重试次数，默认3次
    
    Returns:
        str: 模型回复的内容
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if model_name is None:
        model_name = os.getenv("GPT_MODEL_NAME", "gpt-4.1")
    
    # model_name="deepseek-r1"
    model_name="Moonshot-Kimi-K2-Instruct"
    
    print(f"[call_GPT], model_name: {model_name}, messages: [{messages[-1]['role']}] {messages[-1]['content'][:100]}") 
    global total_prompt_tokens, total_completion_tokens, total_prompt_cost, total_completion_cost
    global gpt_total_prompt_tokens, gpt_total_completion_tokens
    
    
    base_url = os.getenv("GPT_BASE_URL", "https://api.openai.com/v1")
    api_key = os.getenv("GPT_API_KEY")
    
    if not api_key:
        raise ValueError("GPT_API_KEY not found in environment variables")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            # 提取token使用情况
            if "usage" in result:
                with _token_stats_lock:
                    prompt_tokens = result["usage"].get("prompt_tokens", 0)
                    completion_tokens = result["usage"].get("completion_tokens", 0)
                    
                    # 更新全局变量
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    
                    # 更新GPT全局变量
                    gpt_total_prompt_tokens += prompt_tokens
                    gpt_total_completion_tokens += completion_tokens
                    
                    # 计算费用
                    prompt_cost = prompt_tokens * GPT_PROMPT_TOKEN_PRICE
                    completion_cost = completion_tokens * GPT_COMPLETION_TOKEN_PRICE
                    total_prompt_cost += prompt_cost
                    total_completion_cost += completion_cost
                    
                    print(f"本次调用 - Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
                    print(f"本次费用 - Prompt: ${prompt_cost:.4f}, Completion: ${completion_cost:.4f}")
            # print(result)
            return result["choices"][0]["message"]["content"]
        
        except (requests.exceptions.RequestException, KeyError) as e:
            print(f"GPT API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:  # 不是最后一次尝试
                print(f"等待2秒后重试...")
                time.sleep(2)
            else:  # 最后一次尝试失败
                print(f"GPT API连续{max_retries}次调用失败，返回错误信息")
                return "抱歉，我现在无法回复，请稍后再试。"

def call_Deepseek(messages: Union[List[Dict[str, str]], str], model_name: str = None, max_tokens: int = 1000, max_retries: int = 10) -> str:
    """
    调用DeepSeek模型的函数
    
    Args:
        messages: 消息列表，格式为[{"role": "user/assistant/system", "content": "消息内容"}]
        model_name: 模型名称，如果为None则从环境变量读取
        max_retries: 最大重试次数，默认10次
    
    Returns:
        str: 模型回复的内容
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if model_name is None:
        model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-v3")

    print(f"[call_Deepseek], model_name: {model_name}, messages: [{messages[-1]['role']}] {messages[-1]['content'][:30]}") 
    global total_prompt_tokens, total_completion_tokens, total_prompt_cost, total_completion_cost
    global deepseek_total_prompt_tokens, deepseek_total_completion_tokens
    
    base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            # 提取token使用情况
            if "usage" in result:
                with _token_stats_lock:
                    prompt_tokens = result["usage"].get("prompt_tokens", 0)
                    completion_tokens = result["usage"].get("completion_tokens", 0)
                    
                    # 更新全局变量
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    
                    # 更新DeepSeek全局变量
                    deepseek_total_prompt_tokens += prompt_tokens
                    deepseek_total_completion_tokens += completion_tokens
                    
                    # 计算费用
                    prompt_cost = prompt_tokens * DEEPSEEK_PROMPT_TOKEN_PRICE
                    completion_cost = completion_tokens * DEEPSEEK_COMPLETION_TOKEN_PRICE
                    total_prompt_cost += prompt_cost
                    total_completion_cost += completion_cost
                    
                    print(f"本次调用 - Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
                    print(f"本次费用 - Prompt: ${prompt_cost:.4f}, Completion: ${completion_cost:.4f}")
            
            return result["choices"][0]["message"]["content"]
        
        except (requests.exceptions.RequestException, KeyError) as e:
            print(f"DeepSeek API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:  # 不是最后一次尝试
                print(f"等待2秒后重试...")
                time.sleep(2)
            else:  # 最后一次尝试失败
                print(f"DeepSeek API连续{max_retries}次调用失败，返回错误信息")
                return "抱歉，我现在无法回复，请稍后再试。"

def call_Gemini(messages: Union[List[Dict[str, str]], str], model_name: str = None, max_tokens: int = 1000, max_retries: int = 10) -> str:
    """
    调用Google Gemini模型的函数（使用OpenAI兼容接口）
    
    Args:
        messages: 消息列表，格式为[{"role": "user/assistant/system", "content": "消息内容"}]
        model_name: 模型名称，如果为None则从环境变量读取
        max_retries: 最大重试次数，默认10次
    
    Returns:
        str: 模型回复的内容
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
        
    global total_prompt_tokens, total_completion_tokens, total_prompt_cost, total_completion_cost
    
    if model_name is None:
        model_name = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro")
    
    # 使用OpenAI兼容的接口
    base_url = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens
    }
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            # 提取token使用情况（与GPT格式相同）
            if "usage" in result:
                with _token_stats_lock:
                    prompt_tokens = result["usage"].get("prompt_tokens", 0)
                    completion_tokens = result["usage"].get("completion_tokens", 0)
                    
                    # 更新全局变量
                    total_prompt_tokens += prompt_tokens
                    total_completion_tokens += completion_tokens
                    
                    # 计算费用
                    prompt_cost = prompt_tokens * GEMINI_PROMPT_TOKEN_PRICE
                    completion_cost = completion_tokens * GEMINI_COMPLETION_TOKEN_PRICE
                    total_prompt_cost += prompt_cost
                    total_completion_cost += completion_cost
                    
                    print(f"本次调用 - Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}")
                    print(f"本次费用 - Prompt: ${prompt_cost:.4f}, Completion: ${completion_cost:.4f}")
            
            return result["choices"][0]["message"]["content"]
        
        except (requests.exceptions.RequestException, KeyError) as e:
            print(f"Gemini API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:  # 不是最后一次尝试
                print(f"等待2秒后重试...")
                time.sleep(2)
            else:  # 最后一次尝试失败
                print(f"Gemini API连续{max_retries}次调用失败，返回错误信息")
                return "抱歉，我现在无法回复，请稍后再试。"

def call_local_model(messages: Union[List[Dict[str, str]], str], base_url: str, model_name: str = None,
                    lora_path: str = None, temperature: float = 0.7, top_p: float = 0.85, top_k: int = 80, 
                    max_tokens: int = 1000, max_retries: int = 10) -> str:
    """
    调用本地部署模型的函数（支持qwen2.5-7B-Instruct, qwen3-8B, llama3-8B等）
    支持加载ms-swift微调的LoRA模型
    
    Args:
        messages: 消息列表，格式为[{"role": "user/assistant/system", "content": "消息内容"}]
        model_name: 模型名称或路径，如果为None则从环境变量读取
        lora_path: LoRA模型路径，如果为None则从环境变量读取
        temperature: 温度参数，控制输出的随机性 (0.0-2.0)，默认0.7
        top_p: 核采样参数，控制词汇选择的多样性 (0.0-1.0)，默认0.85
        top_k: 选择前k个最可能的token (-1表示不限制)，默认80
        max_tokens: 最大生成token数，默认1000
        max_retries: 最大重试次数，默认3次
        【补充】
        1. 通用场景推荐temperature=0.8, top_p=0.85, top_k=80；发散性场景推荐temperature=1.1, top_p=0.95, top_k=-1；如果不填默认是temperature=1.0, top_p=1.0, top_k=-1
        2. model_name和lora_path实际上在.env都没填，所以都是None
    
    Returns:
        str: 模型回复的内容
    """
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    # print(f"[call_local_model], model_name: {model_name}, messages: [{messages[-1]['role']}] {messages[-1]['content'][:30]}") 
    # 简化配置：只需要端口，其他从vLLM服务自动获取
    # base_url = os.getenv("LOCAL_MODEL_BASE_URL", "http://localhost:8000")
    
    # 如果没指定模型名，尝试从环境变量获取，否则让vLLM自动选择
    if model_name is None:
        model_name = os.getenv("LOCAL_MODEL_NAME", None)
    
    # LoRA路径（可选）
    if lora_path is None:
        lora_path = os.getenv("LOCAL_LORA_PATH", None)
    
    headers = {
        "Content-Type": "application/json"
    }
    
    data = {
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "max_tokens": max_tokens
    }
    
    # 只有在指定了模型名时才添加到请求中
    if model_name:
        data["model"] = model_name
    
    # 如果提供了LoRA路径，添加到请求中
    if lora_path and lora_path.lower() != "none":
        data["lora_path"] = lora_path
        print(f"使用LoRA模型: {lora_path}")
    
    # 重试机制
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=60  # 本地模型可能需要更长时间
            )
            response.raise_for_status()
            
            result = response.json()
            # print(result)
            return result["choices"][0]["message"]["content"]
        
        except (requests.exceptions.RequestException, KeyError) as e:
            print(f"本地模型API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:  # 不是最后一次尝试
                print(f"等待2秒后重试...")
                time.sleep(2)
            else:  # 最后一次尝试失败
                print(f"本地模型API连续{max_retries}次调用失败，返回错误信息")
                return "抱歉，本地模型现在无法回复，请检查模型是否正在运行。"

def call_llm(messages: Union[List[Dict[str, str]], str], llm_type: str = None, role: str = None, max_tokens: int = 1000, base_url: str = None, api_args: dict = None) -> str:
    """
    通用的LLM调用函数
    
    Args:
        messages: 消息列表，格式为[{"role": "user/assistant/system", "content": "消息内容"}]，也支持字符串类型（会转换成标准格式）
        llm_type: LLM类型，如果为None则根据role从环境变量读取（兼容旧代码，但推荐显式传入）
        role: 角色类型（"user"或"assistant"），当llm_type为None时用于自动选择对应的模型
    
    Returns:
        str: 模型回复的内容
    """
    # 处理字符串类型的messages，转换为标准格式
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    
    # 如果llm_type为None，则从环境变量读取（兼容旧代码，但推荐显式传入）
    if llm_type is None:
        print("【警告】llm_type为None，将根据role从环境变量读取，这不是期望行为！请显式传入llm_type！")
        if role == "user":
            llm_type = os.getenv("USER_LLM_TYPE", "gpt")
        elif role == "assistant":
            llm_type = os.getenv("ASSISTANT_LLM_TYPE", "gpt")
        else:
            llm_type = os.getenv("DEFAULT_LLM_TYPE", "gpt")
    
    if llm_type.lower() == "gpt":
        return call_GPT(messages, max_tokens=max_tokens)
    elif llm_type.lower() == "deepseek":
        return call_Deepseek(messages, max_tokens=max_tokens)
    elif llm_type.lower() == "gemini":
        return call_Gemini(messages, max_tokens=max_tokens)
    # elif llm_type.lower() in ["local", "qwen", "llama", "qwen2.5", "qwen3", "llama3"]:
    elif llm_type.lower() == "local":
        assert base_url is not None and base_url.lower() != "none", "如果使用本地模型，则base_url必须填"
        return call_local_model(messages, base_url, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}. Supported types: gpt, deepseek, gemini, local, qwen, llama, qwen2.5, qwen3, llama3")

def print_token_usage_summary():
    """
    打印token使用情况和费用统计
    """
    global total_prompt_tokens, total_completion_tokens, total_prompt_cost, total_completion_cost
    with _token_stats_lock:
        prompt_tokens = total_prompt_tokens
        completion_tokens = total_completion_tokens
        prompt_cost = total_prompt_cost
        completion_cost = total_completion_cost

    print("\n" + "="*50)
    print("TOKEN使用情况和费用统计")
    print("="*50)
    print(f"总Prompt Tokens: {prompt_tokens:,}")
    print(f"总Completion Tokens: {completion_tokens:,}")
    print(f"总Token数: {prompt_tokens + completion_tokens:,}")
    print(f"总Prompt费用: ${prompt_cost:.4f}")
    print(f"总Completion费用: ${completion_cost:.4f}")
    print(f"总费用: ${prompt_cost + completion_cost:.4f}")
    print("="*50)
    
    print(f"GPT总Prompt Tokens: {gpt_total_prompt_tokens:,}")
    print(f"GPT总Completion Tokens: {gpt_total_completion_tokens:,}")
    print("="*50)
    
    print(f"DeepSeek总Prompt Tokens: {deepseek_total_prompt_tokens:,}")
    print(f"DeepSeek总Completion Tokens: {deepseek_total_completion_tokens:,}")
    print("="*50)

if __name__ == "__main__":
    result = call_llm(messages=[{"role": "user", "content": "你好，请问1+1等于几？"}], llm_type="deepseek")
    print(result)