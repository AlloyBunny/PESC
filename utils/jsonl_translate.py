import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
import json
import re
from inference.llm_calls import call_llm

def translate_jsonl(input_file, output_file):
    """
    读取JSONL文件，翻译每个JSON对象，并写入新的JSONL文件
    
    参数:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # 解析JSON行
                data = json.loads(line.strip())
                
                # 将JSON对象转换为字符串，准备翻译
                json_str = json.dumps(data, ensure_ascii=False)
                
                # 调用LLM API进行翻译，结果包含在```json ```中
                translated_with_wrapper = call_llm(json_str+'\n帮我把上面这段json翻译为英文，格式不要改变，放在```json ```里输出，方便我复制。不要换行')
                print(translated_with_wrapper)
                
                # 提取```json ```包裹的内容
                # 使用正则表达式匹配代码块中的JSON内容
                match = re.search(r'```json\s*(.*?)\s*```', translated_with_wrapper, re.DOTALL)
                if not match:
                    raise ValueError("翻译结果中未找到有效的JSON代码块")
                
                translated_json_str = match.group(1)
                
                # 验证提取的内容是否为有效的JSON
                translated_data = json.loads(translated_json_str)
                
                # 写入到输出文件，保持JSONL格式
                outfile.write(json.dumps(translated_data, ensure_ascii=False) + '\n')
                
                print(f"已处理第 {line_num} 行")
                
            except json.JSONDecodeError as e:
                print(f"第 {line_num} 行JSON解析错误: {str(e)}")
            except Exception as e:
                print(f"第 {line_num} 行处理错误: {str(e)}")

# 示例用法
if __name__ == "__main__":
    # 输入和输出文件路径
    input_jsonl = "profile/test_random30.jsonl"
    output_jsonl = "profile/test_random30_en.jsonl"
    
    # 调用翻译函数
    translate_jsonl(input_jsonl, output_jsonl)
    print("翻译完成！")
