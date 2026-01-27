import sys
import json 
import random 
from openai import OpenAI

class gpt4o():
    def __init__(self, model_name="gpt-4o", temperature=None) -> None:
        # gpt-4o-2024-05-13
        self.api_key = ""
        self.base_url = "https://pro.xiaoai.plus/v1"
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.model_name = model_name
        self.temperature = temperature
        
        # 价格设置 (每百万token)
        self.input_price_per_million = 2.5  # 美元
        self.output_price_per_million = 10.0  # 美元
        
        # 统计信息
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0
        
        print(f"model_name: {self.model_name}; temperature:{self.temperature}")
        print(f"价格设置: 输入 {self.input_price_per_million}$/M tokens, 输出 {self.output_price_per_million}$/M tokens")

    
    def __call__(self, message, maxtry=10):
        # 支持两种输入格式：str 或 list of dict (messages)
        if isinstance(message, str):
            messages = [{"role":"user", "content": message}]
        elif isinstance(message, list):
            # 验证 messages 格式
            assert all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in message), \
                'Each message should be a dict with "role" and "content" keys.'
            messages = message
        else:
            raise ValueError('The input should be either a string or a list of message dictionaries.')
        
        i = 0
        while i < maxtry:
            try:
                if self.temperature is None:
                    response = self.client.chat.completions.create(
                        model = self.model_name,
                        messages=messages
                    )
                else:
                    response = self.client.chat.completions.create(
                        model = self.model_name,
                        messages=messages,
                        temperature=self.temperature
                    )
                
                # 统计token使用量
                usage = response.usage
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens
                
                # 计算本次调用成本
                input_cost = (input_tokens / 1_000_000) * self.input_price_per_million
                output_cost = (output_tokens / 1_000_000) * self.output_price_per_million
                call_cost = input_cost + output_cost
                
                # 更新总统计
                self.total_input_tokens += input_tokens
                self.total_output_tokens += output_tokens
                self.total_cost += call_cost
                
                print(f"Token使用: 输入 {input_tokens}, 输出 {output_tokens}, 本次成本 ${call_cost:.4f}")
                
                response = response.choices[0].message.content
                return response
            except Exception as e:
                print(f"Try {i}/{maxtry}\t messages:{messages} \tError:{e}", flush=True)
                i += 1
                continue
        return response
    
    def get_usage_stats(self):
        """获取使用统计信息"""
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost": round(self.total_cost, 4),
            "input_price_per_million": self.input_price_per_million,
            "output_price_per_million": self.output_price_per_million
        }
    
    def reset_stats(self):
        """重置统计信息"""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0.0

if __name__ == "__main__":
    model = gpt4o()
    
    # 测试字符串输入
    print("测试字符串输入:", model("1+1"))
    
    # 测试messages输入
    messages = [
        {"role": "system", "content": "你是一个数学助手"},
        {"role": "user", "content": "1+1等于多少？"}
    ]
    print("测试messages输入:", model(messages))
    
    
