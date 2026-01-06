import sys
import json 
import random 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import os

class Llama2_7b_chat_hf():
    def __init__(self, model_name="Llama-2-7b-chat-hf", temperature=None) -> None:
        # 模型路径
        self.model_path = "../models/Llama-2-7b-chat-hf"
        self.model_name = model_name
        self.temperature = temperature if temperature is not None else 0.7
        
        # 检查CUDA是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 加载tokenizer和模型
        print(f"Loading model from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 设置生成配置
        self.generation_config = GenerationConfig(
            temperature=self.temperature,
            do_sample=True if self.temperature > 0 else False,
            max_new_tokens=512,  # 保持原来的输出长度
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1
        )
        
        # 设置最大输入长度
        self.max_length = 4096
        
        print(f"model_name: {self.model_name}; temperature: {self.temperature}")

    def _format_chat_prompt(self, messages):
        """
        将messages格式转换为Llama2 chat格式的prompt
        """
        try:
            # 使用tokenizer的chat_template来格式化
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return formatted_prompt
        except Exception as e:
            print(f"Error in chat template formatting: {e}")
            # 如果chat_template失败，使用手动格式化
            return self._manual_format_chat(messages)
    
    def _manual_format_chat(self, messages):
        """
        手动格式化Llama2 chat prompt，作为备用方案
        """
        formatted_parts = []
        system_message = None
        
        # 提取system消息
        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
                break
        
        # 格式化对话
        conversation_messages = [msg for msg in messages if msg["role"] != "system"]
        
        for i, message in enumerate(conversation_messages):
            if message["role"] == "user":
                if i == 0 and system_message:
                    # 第一个user消息，包含system指令
                    content = f"<<SYS>>\n{system_message}\n<</SYS>>\n\n{message['content']}"
                else:
                    content = message["content"]
                formatted_parts.append(f"<s>[INST] {content} [/INST]")
            elif message["role"] == "assistant":
                formatted_parts.append(f" {message['content']} </s>")
        
        # 如果最后一个消息是user消息，我们需要为assistant的回复留出空间
        if conversation_messages and conversation_messages[-1]["role"] == "user":
            return "".join(formatted_parts) + " "
        else:
            return "".join(formatted_parts)
    
    def __call__(self, message, maxtry=3):
        # 支持两种输入格式：str 或 list of dict (messages)
        if isinstance(message, str):
            messages = [{"role": "user", "content": message}]
        elif isinstance(message, list):
            # 验证 messages 格式
            assert all(isinstance(msg, dict) and "role" in msg and "content" in msg for msg in message), \
                'Each message should be a dict with "role" and "content" keys.'
            messages = message
        else:
            raise ValueError('The input should be either a string or a list of message dictionaries.')
        
        for attempt in range(maxtry):
            try:
                # 格式化prompt
                if isinstance(message, str):
                    # 对于简单字符串，直接使用
                    prompt = message
                    inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
                else:
                    # 对于messages格式，使用chat template
                    prompt = self._format_chat_prompt(messages)
                    inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
                
                # 检查是否发生了截断
                input_length = inputs['input_ids'].shape[1]
                if input_length == self.max_length:
                    print(f"Warning: Input truncated to {self.max_length} tokens")
                
                # 移动到正确的设备
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # 生成回复
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        generation_config=self.generation_config,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                
                # 解码回复（只获取新生成的部分）
                input_length = inputs['input_ids'].shape[1]
                generated_tokens = outputs[0][input_length:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                
                # 清理回复
                response = response.strip()
                
                return response
                
            except Exception as e:
                print(f"Try {attempt+1}/{maxtry}\t messages:{messages} \tError:{e}", flush=True)
                if attempt == maxtry - 1:
                    return "抱歉，我现在无法回复，请稍后再试。"
                continue

if __name__ == "__main__":
    model = Llama2_7b_chat_hf()
    
    # 测试字符串输入
    print("测试字符串输入:")
    print(model("你好，请介绍一下自己"))
    print("\n" + "="*50 + "\n")
    
    # 测试messages输入
    print("测试messages输入:")
    messages = [
        {"role": "system", "content": "你是一个友善的AI助手"},
        {"role": "user", "content": "你好，请介绍一下自己"}
    ]
    print(model(messages))
