#!/usr/bin/env python
#coding:utf-8

import json, os, glob, random, sys, argparse, time
import concurrent.futures
import traceback
import threading 
import importlib
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from models import *

class SAGEPairwiseInference():
    def __init__(self, infer_model, input_path, output_path, para_num, temperature=0.9, num_iterations=5, batch_size=1000):
        self.infer_model = infer_model
        self.output_path = output_path
        self.input_path = input_path
        self.example_num = 0
        self.para_num = para_num
        self.temperature = temperature
        self.num_iterations = num_iterations
        self.batch_size = batch_size  # 批量保存的大小
        self.model = self._get_model(self.infer_model)
        
        # 添加线程锁用于文件写入
        self.file_lock = threading.Lock()
        self.saved_count = 0
        self.batch_buffer = []  # 批量保存缓冲区
        
        # SAGE系统指令
        # self.system_instruction = "你是一个智能聊天伙伴，你擅长高情商地和用户聊天，让用户感到舒适、愉快或得到需要的帮助。"
    
    def _get_model(self, model_name):
        try:
            module = importlib.import_module(f"models.{model_name}")
            model_class = getattr(module, model_name)
            print(f"module:{module}, model_class:{model_class}")
            return model_class(temperature=self.temperature)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"model_name:'{model_name}' is not defined: {e}")
        except Exception as e:
            print(f'error:{e}')
    
    def _load_examples(self, in_path):
        try:
            data = [json.loads(line) for line in open(self.input_path, "r", encoding="utf-8") if line.strip()]
            self.example_num = len(data)
            return data
        except:
            raise ValueError(f"Dataset error, please check data or in_path")
    
    def _infer_one(self, task_with_iteration):
        task, iteration = task_with_iteration
        try:
            # 从dialog_history构建messages格式
            dialog_history = task["dialog_history"]
            assert dialog_history[0]["role"] == "system", "dialog_history的第一项的role必须是system"
            assert dialog_history[-1]["role"] == "assistant", "dialog_history的最后一项的role必须是assistant"
            if iteration == 0: # 期望chosen回复
                system_instruction = dialog_history[0]["content"]
                # 找到最后一条用户消息的位置
                last_user_idx = -1
                for i, msg in enumerate(dialog_history):
                    if msg["role"] == "user":
                        last_user_idx = i
                
                if last_user_idx == -1:
                    raise ValueError("对话历史中没有用户消息")
                
                # 构建messages格式，包含system指令和多轮对话
                messages = [{"role": "system", "content": system_instruction}]
                
                # 添加历史对话，只到最后一条用户消息
                for i in range(last_user_idx + 1):
                    msg = dialog_history[i]
                    if msg["role"] in ["user", "assistant"]: # sytem的部分是给打分模型看的，inference的system在上面已经给了
                        messages.append({
                            "role": msg["role"],
                            "content": msg["content"]
                        })
            
                # 调用模型生成回复
                response = self.model(messages)

            elif iteration == 1: # 期望rejected回复
                response = dialog_history[-1]["content"]
            else:
                raise ValueError("iteration必须为0或1")
            
            # 创建新的任务副本，添加iteration信息
            task_copy = task.copy()
            task_copy['dialog_history'] = dialog_history[:-1] # 去掉最后的assistant回复
            task_copy['response'] = response
            task_copy["infer_model"] = self.infer_model
            task_copy["iteration"] = iteration
            task_copy["temperature"] = self.temperature
            
            return task_copy
            
        except Exception as e:
            print(f"Error in _infer_one for task {task.get('idx', 'unknown')} iteration {iteration}: {e}")
            task_copy = task.copy()
            task_copy['response'] = "抱歉，我现在无法回复，请稍后再试。"
            task_copy["infer_model"] = self.infer_model
            task_copy["iteration"] = iteration
            task_copy["temperature"] = self.temperature
            return task_copy
    
    def _infer_parallel(self, tasks, para_num):
        """并行推理并批量保存结果"""
        # 为每个任务创建多个iteration
        tasks_with_iterations = []
        for task in tasks:
            for iteration in range(self.num_iterations):
                tasks_with_iterations.append((task, iteration))
        
        # 用于收集成对的结果
        task_results = {}  # {task_idx: [result0, result1]}
        
        def process_and_batch_save(entry):
            """处理单个结果并尝试批量保存成对数据"""
            task_idx = entry["idx"]
            
            # 线程安全地添加到结果字典
            with self.file_lock:
                if task_idx not in task_results:
                    task_results[task_idx] = [None, None]
                task_results[task_idx][entry["iteration"]] = entry
            
            # 检查是否收集到一对完整的结果
            if all(task_results[task_idx]):
                # 格式化并添加到批量缓冲区
                formatted_item = self._format_single_pair(task_results[task_idx])
                self.batch_buffer.append(formatted_item)
                
                # 检查是否达到批量保存阈值
                if len(self.batch_buffer) >= self.batch_size:
                    self._save_batch()
                
                # 清理已保存的数据以节省内存
                del task_results[task_idx]
            
            return entry
        
        results = []
        with ThreadPoolExecutor(para_num) as executor:
            for entry in tqdm(executor.map(self._infer_one, tasks_with_iterations), 
                             total=len(tasks_with_iterations),
                             desc=f'{self.infer_model} pairwise inference (temp={self.temperature}, iter={self.num_iterations}):'):
                process_and_batch_save(entry)
                results.append(entry)
        
        # 处理剩余未配对的数据（如果有的话）
        for task_idx, pair_results in task_results.items():
            if all(pair_results):
                formatted_item = self._format_single_pair(pair_results)
                self.batch_buffer.append(formatted_item)
        
        # 保存剩余的批量数据
        if self.batch_buffer:
            self._save_batch()
        
        print(f"批量保存完成，总共保存了 {self.saved_count} 条数据")
        return results

    def _format_single_pair(self, pair_results):
        """格式化单个配对的结果"""
        item0, item1 = pair_results
        assert item0["iteration"] == 0 and item1["iteration"] == 1
        assert item0["idx"] == item1["idx"] and item0["id"] == item1["id"]
        
        new_item = {
            "idx": item0["idx"],
            "id": item0["id"],
            "dialog_history": item0["dialog_history"],
            "chosen": item0["response"],
            "rejected": item1["response"],
            "gold": "",
            "split": "hard",
            "L3_memory": item0["L3_memory"],
            "L2_memory": item0["L2_memory"],
            "L1_memory": item0["L1_memory"],
            "infer_model": item0["infer_model"],
            "temperature": item0["temperature"]
        }
        return new_item

    def _save_batch(self):
        """批量保存数据到JSONL文件"""
        if not self.batch_buffer:
            return
            
        try:
            with self.file_lock:
                # 确保输出目录存在
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                
                # 以追加模式写入JSONL格式
                with open(self.output_path, 'a', encoding='utf-8') as f:
                    for item in self.batch_buffer:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                self.saved_count += len(self.batch_buffer)
                print(f"批量保存了 {len(self.batch_buffer)} 条数据，累计保存 {self.saved_count} 条")
                
                # 清空缓冲区
                self.batch_buffer = []
                    
        except Exception as e:
            print(f"批量保存错误: {e}")
    
    def _save_result(self, result):
        """保留原有的批量保存方法作为备用"""
        try:
            if not os.path.exists(self.output_path):
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            json.dump(result, open(self.output_path,'w',encoding='utf-8'),ensure_ascii=False,indent=4)
            print(f"Results saved to: {self.output_path}")
        except Exception as e:
            print(f"save result error, {e}")
            
    def _result_format(self, result):
        assert len(result)%2 == 0, "result的length必须是偶数"
        result_format = []
        for i in range(0, len(result), 2):
            item0 = result[i]
            item1 = result[i+1]
            assert item0["iteration"] == 0 and item1["iteration"] == 1
            assert item0["idx"] == item1["idx"] and item0["id"] == item1["id"]
            new_item = {
                "idx": item0["idx"],
                "id": item0["id"],
                "dialog_history": item0["dialog_history"],
                "chosen": item0["response"],
                "rejected": item1["response"],
                # "criteria": item0["criteria"],
                "gold": "",
                "split": "hard",
                "L3_memory": item0["L3_memory"],
                "L2_memory": item0["L2_memory"],
                "L1_memory": item0["L1_memory"],
                "infer_model": item0["infer_model"],
                "temperature": item0["temperature"]
            }
            result_format.append(new_item)
        return result_format

    def __call__(self):
        start_time = time.time()
        
        # 重置模型统计信息
        self.model.reset_stats()
        
        # 清空输出文件（如果存在）
        if os.path.exists(self.output_path):
            os.remove(self.output_path)
            print(f"已清空输出文件: {self.output_path}")
        
        datas = self._load_examples(self.input_path)
        result = self._infer_parallel(datas, self.para_num)
        
        # 注意：这里不再需要_result_format和_save_result，因为已经在_infer_parallel中批量处理了
        
        # 获取token使用统计
        usage_stats = self.model.get_usage_stats()
        print(f"\n=== Token使用统计 ===")
        print(f"总输入tokens: {usage_stats['total_input_tokens']:,}")
        print(f"总输出tokens: {usage_stats['total_output_tokens']:,}")
        print(f"总tokens: {usage_stats['total_tokens']:,}")
        print(f"总成本: ${usage_stats['total_cost']:.4f}")
        print(f"输入价格: ${usage_stats['input_price_per_million']}/M tokens")
        print(f"输出价格: ${usage_stats['output_price_per_million']}/M tokens")
        
        end_time = time.time()
        print(f"**** Pairwise Inference Done, Total Time {end_time-start_time:.2f} s, API Cost ${usage_stats['total_cost']:.4f} ****")
        print(f"**** 结果已批量保存到: {self.output_path} ****")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_model", type=str, default="moonshot")
    parser.add_argument("--input_path", type=str, default="../data/sage_data_1.json")
    parser.add_argument("--output_path", type=str, default="../output/response")
    parser.add_argument("--max_threads", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for model inference")
    parser.add_argument("--num_iterations", type=int, default=2, help="Number of iterations per task")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for saving results")
    args = parser.parse_args()
    
    assert args.num_iterations == 2, "现在要求num_iterations必须为2，因为0对应【有个性化记忆、期望为best】，1对应【无个性化记忆、期望为worst】"
    sage_infer = SAGEPairwiseInference(
        args.infer_model, 
        args.input_path, 
        args.output_path, 
        args.max_threads,
        args.temperature,
        args.num_iterations,
        args.batch_size
    )
    sage_infer()
