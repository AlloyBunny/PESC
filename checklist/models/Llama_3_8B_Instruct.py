import sys
import json 
import random 
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel, PeftConfig
import os

class Llama_3_8B_Instruct():
    def __init__(self, model_name="Llama-3-8B-Instruct", temperature=None, lora_path=None) -> None:
        # 模型路径
        self.model_path = "models/Meta-Llama-3-8B-Instruct"
        self.model_name = model_name
        self.temperature = temperature if temperature is not None else 0.7
        self.lora_path = lora_path
        
        # 检查CUDA是否可用
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 加载tokenizer和模型
        print(f"Loading model from: {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        
        # 设置pad_token (Llama通常使用eos_token作为pad_token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 如果指定了LoRA路径，加载LoRA权重
        if self.lora_path:
            print(f"Loading LoRA weights from: {self.lora_path}")
            try:
                # 首先检查LoRA配置
                if os.path.exists(os.path.join(self.lora_path, "adapter_config.json")):
                    peft_config = PeftConfig.from_pretrained(self.lora_path)
                    print(f"LoRA配置: r={peft_config.r}, lora_alpha={peft_config.lora_alpha}")
                    print(f"目标模块: {peft_config.target_modules}")
                
                # 加载LoRA权重
                self.model = PeftModel.from_pretrained(
                    self.model,
                    self.lora_path,
                    torch_dtype=torch.float16,
                    is_trainable=False  # 推理时设置为False
                )
                
                # 确保模型处于评估模式
                self.model.eval()
                
                print("LoRA weights loaded successfully!")
                
                # 验证LoRA是否正确加载
                self._verify_lora_loading()
                
            except Exception as e:
                print(f"Error loading LoRA weights: {e}")
                print("Continuing without LoRA...")
                self.lora_path = None
        
        # 设置生成配置
        self.generation_config = GenerationConfig(
            temperature=self.temperature,
            do_sample=True if self.temperature > 0 else False,
            max_new_tokens=1024,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=1.1,
            top_p=0.9,  # Llama模型通常使用top_p采样
            top_k=50
        )
        
        # 设置最大输入长度 (Llama-3支持8K上下文)
        self.max_length = 8192
        
        print(f"model_name: {self.model_name}; temperature: {self.temperature}; lora_path: {self.lora_path}")

    def _verify_lora_loading(self):
        """验证LoRA是否正确加载"""
        try:
            # 检查是否有LoRA层
            lora_layers = 0
            for name, module in self.model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_layers += 1
            
            if lora_layers > 0:
                print(f"✓ 成功加载了 {lora_layers} 个LoRA层")
            else:
                print("⚠ 警告: 未检测到LoRA层，可能加载失败")
                
        except Exception as e:
            print(f"验证LoRA加载时出错: {e}")

    def _format_chat_prompt(self, messages):
        """
        将messages格式转换为Llama-3 chat格式的prompt
        Llama-3使用特殊的chat格式:
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        
        {system_content}<|eot_id|><|start_header_id|>user<|end_header_id|>
        
        {user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """
        try:
            # 首先尝试使用tokenizer的chat_template
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
        手动格式化Llama-3 chat prompt，作为备用方案
        """
        formatted_parts = ["<|begin_of_text|>"]
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role == "user":
                formatted_parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>")
            elif role == "assistant":
                formatted_parts.append(f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>")
        
        # 为assistant的回复添加开始标记
        formatted_parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        
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
                    # 对于简单字符串，使用基本的user格式
                    messages_for_prompt = [{"role": "user", "content": message}]
                    prompt = self._format_chat_prompt(messages_for_prompt)
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
                
                # 清理回复 - 移除可能的特殊标记
                response = response.strip()
                # 移除可能残留的Llama-3特殊标记
                response = response.replace("<|eot_id|>", "").strip()
                response = response.replace("<|end_header_id|>", "").strip()
                
                return response
                
            except Exception as e:
                print(f"Try {attempt+1}/{maxtry}\t messages:{messages} \tError:{e}", flush=True)
                if attempt == maxtry - 1:
                    return "抱歉，我现在无法回复，请稍后再试。"
                continue

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Llama-3-8B-Instruct模型测试")
    parser.add_argument("--lora_path", type=str, help="LoRA权重路径")
    parser.add_argument("--temperature", type=float, default=0.7, help="生成温度")
    args = parser.parse_args()
    
    model = Llama_3_8B_Instruct(temperature=args.temperature, lora_path=args.lora_path)
    
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
    
    # 如果使用了LoRA，测试SAGE场景
    if args.lora_path:
        print("\n" + "="*50 + "\n")
        print("测试SAGE场景（DPO训练后）:")
        sage_messages = [
            {"role": "system", "content": "你是一个智能聊天伙伴，你擅长高情商地和用户聊天，让用户感到舒适、愉快或得到需要的帮助。"},
            {"role": "user", "content": "最近有些事搞得我头大，张浩那个态度，真让我摸不着头脑。"}
        ]
        print(model(sage_messages))
