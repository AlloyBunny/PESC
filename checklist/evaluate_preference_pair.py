#!/usr/bin/env python
#coding:utf-8

import os, glob, json, re, time
from tqdm import tqdm
import numpy as np
import pandas as pd
import importlib
from concurrent.futures import ThreadPoolExecutor
import arrow
import argparse
from collections import defaultdict

# SAGE数据集专用的评估提示模板 - 新版支持chosen/rejected对比
sage_judge_for_cfbench = lambda dialog_history, chosen_response, rejected_response, criteria: f'''
我想让你扮演一个情感智能评估器的角色。你需要结合[对话历史]、[AI回复1]、[AI回复2]、[评估标准]对AI回复进行评估，对评估标准中的每一项，对比[AI回复1]和[AI回复2]，选出更能满足该项标准的回复。

具体而言：[对话历史]显示了用户与AI的交互过程，[AI回复1]和[AI回复2]都是AI对最后一条用户消息的回应，[评估标准]定义了AI回复应该满足的情感支持和沟通质量标准。

你需要严格按照[评估标准]中的每条标准，选出针对每条[评估标准]的更优的AI回复。

输出格式要求: 
1. 严格按照[评估标准]中标准的编号顺序输出，每行输出一个标准的评估结果
2. 每行格式为："标准X: [标准内容]\t[0/1/2]"，其中X为标准编号，后面填1表示[AI回复1]更优，填2表示[AI回复2]更优，填0表示两者相当（比如，如果[AI回复1]在某个标准上完成得比[AI回复2]好，那么这项填1；如果[AI回复1]和[AI回复2]在某个标准上完成得一样好，或者两个回复都做的不好，那么这一项填0）
3. 请直接输出评估结果，不要输出其他任何内容
4. 不要遗漏任何一个标准

可参考下面样例。

[样例]:
    [对话历史]: User: 我真是想不通，明明做得很不错，结果客户却没了，真是烦透了。\n\nAssistant: 听起来你真的很努力，遇到这样的情况真是让人沮丧。客户流失可能有很多原因，有时候并不一定是你的错。也许下一个机会会更适合你。要不要我们一起来看看有没有什么地方可以调整一下，帮助你在未来的项目中取得更好的结果？或者，如果你需要，我也可以只是陪你聊聊，让你感觉轻松一些。\n\nUser: 哎，客户这要求真是多得让人头大，感觉怎么都满足不了。
    
    [AI回复1]: 我完全理解这样的感受，客户的需求有时候真是让人脑袋要炸了！或许我们可以从几个具体步骤入手来减轻你的负担。比如，明天早上去尝试与客户建立一个明确的、分优先级的需求清单，把需求记录下来，逐步按优先级去解决。同时，设定一些可以达成的小目标，把任务拆分几份，每天完成一点，这样压力可能会小一些。\n\n如果客户的需求确实过于繁杂，考虑是不是可以通过定期的进度汇报，让他们了解项目的进展，而不是一次性满足所有要求。这样可能会帮助你更好地管理项目，并让客户看到你的努力和进展。希望这些小方法能给你带来一些帮助！
    
    [AI回复2]: 听起来客户的要求真的让你感到很头疼。我很理解这种感觉，似乎无论怎么努力都无法满足对方的期望。或许可以尝试主动和客户沟通，明确哪些要求是优先级最高的，这样能帮助你更有针对性地处理问题。\n\n另外，设定一些界限也是很重要的，这样才能保护自己的精力，不至于被过多的要求压垮。记得，你已经很努力了，不必过于苛责自己。或许在适当的时候，寻找同事或朋友的支持，也会让你感觉好一点。你并不孤单，有人愿意倾听并帮助你。
    
    [评估标准]:
    标准1: 必须直接回应并接住用户‘多得让人头大’的夸张表达风格，例如：‘这要求多到简直能让人原地爆炸！我们先别被它压垮，看看有没有什么能快速见效的突破口？
    标准2: 禁止提供任何宏观、抽象的建议（如‘换个角度’、‘视为学习机会’）。因为用户的核心偏好是‘具体且快速见效的建议’。
    标准3: 必须提供至少1个‘3天内可执行’的具体行动步骤，例如：‘针对最紧急的3个要求，今晚先列个优先级清单，明早9点直接打电话跟客户确认重点’
    标准4: 回应的前半部分必须包含对用户‘很拼’的明确肯定（如‘你这种拼劲儿绝对值得个VIP客户！’），再衔接具体建议，以符合其需要情感支持的偏好
    标准5: 在给出建议后，需用‘24小时见效’这类短时效词语强化其快速获得感（如‘明天这时候你就能看到客户态度软化’）
    
    [你的评估]: 
    标准1: 必须直接回应并接住用户‘多得让人头大’的夸张表达风格，例如：‘这要求多到简直能让人原地爆炸！我们先别被它压垮，看看有没有什么能快速见效的突破口？\t1
    标准2: 禁止提供任何宏观、抽象的建议（如‘换个角度’、‘视为学习机会’）。因为用户的核心偏好是‘具体且快速见效的建议’。\t1
    标准3: 必须提供至少1个‘3天内可执行’的具体行动步骤，例如：‘针对最紧急的3个要求，今晚先列个优先级清单，明早9点直接打电话跟客户确认重点’\t1
    标准4: 回应的前半部分必须包含对用户‘很拼’的明确肯定（如‘你这种拼劲儿绝对值得个VIP客户！’），再衔接具体建议，以符合其需要情感支持的偏好\t2
    标准5: 在给出建议后，需用‘24小时见效’这类短时效词语强化其快速获得感（如‘明天这时候你就能看到客户态度软化’）\t0

[对话历史]: {dialog_history}

[AI回复1]: {chosen_response}

[AI回复2]: {rejected_response}

[评估标准]: {criteria}

[你的评估]: 
'''

class SAGEPreferencePairEvaluation():
    def __init__(self, input_path, output_path, para_num, eval_model="gpt4o", temperature=0.01):
        self.eval_model = eval_model
        self.in_path = input_path
        self.out_path = output_path
        self.para_num = para_num
        self.temperature = temperature
        self.eval_model = self._get_model(self.eval_model)
        print(f"in_path:{self.in_path}\t out_path:{self.out_path}\t para_num:{self.para_num}\t temperature:{self.temperature}")

    def _get_model(self, model_name):
        try:
            module = importlib.import_module(f"models.{model_name}")
            model_class = getattr(module, model_name)
            return model_class(temperature=self.temperature)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"{model_name} is not defined: {e}, Please ensure that module_name is equal to model_name and that both have been defined.")
        except Exception as e:
            print(f'error:{e}')

    def _format_dialog_history(self, dialog_history):
        """格式化对话历史为易读格式"""
        formatted = []
        for i, msg in enumerate(dialog_history):
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def _get_judge_results(self, task):
        """解析judge_response，提取0/1/2的评估结果"""
        judge_response = task["judge_response"]
        criteria_count = len(task["criteria"])
        results = []
        try:
            judges = judge_response.strip().split("\n")
            # 过滤空行
            judges = [j for j in judges if j.strip()]
            
            assert len(judges) == criteria_count, f"评估条目数量不匹配: 期望{criteria_count}个，实际{len(judges)}个"
            
            for idx, judge in enumerate(judges):
                parts = judge.split("\t")
                assert len(parts) == 2, f"格式错误，无法用\\t分割: {judge}"
                
                crit_part, result = parts[0].strip(), parts[1].strip()
                
                # 只验证是否以"标准X:"开头，不验证具体内容
                expected_prefix = f"标准{idx+1}:"
                assert crit_part.startswith(expected_prefix), f"标准编号错误: 期望以'{expected_prefix}'开头，实际为'{crit_part[:20]}...'"
                
                # 验证结果
                result_val = int(result)
                assert result_val in [0, 1, 2], f"结果必须为0、1或2，实际为{result_val}"
                
                results.append(result_val)
                
        except Exception as e:
            print(f"Judge parsing error: {e}")
            return None
        return results

    def _get_judge_results_guaranteed(self, task):
        """保证返回结果的解析方法，即使格式有问题也能返回部分结果"""
        judge_response = task["judge_response"]
        results = []
        judges = judge_response.strip().split("\n")
        
        for judge in judges:
            if not judge.strip():  # 跳过空行
                continue
            if len(judge.split("\t")) == 2:
                crit_part, result = judge.split("\t")
                try:
                    # 尝试验证结果是否有效
                    result_val = int(result.strip())
                    if result_val in [0, 1, 2]:
                        results.append(result_val)
                except ValueError:
                    continue
        return results

    def _judge_by_gpt(self, inputs):
        """调用GPT进行偏好对评估"""
        dialog_history, chosen_response, rejected_response, criteria = inputs
        formated_input = sage_judge_for_cfbench(dialog_history, chosen_response, rejected_response, criteria)
        MAX_RETRY_NUM = 10
        for i in range(MAX_RETRY_NUM):
            try:
                response = self.eval_model(formated_input)
                return response
            except Exception as e:
                print(f"Error in gpt judge, retrying...{i}/{MAX_RETRY_NUM}")
                print(e)
                continue
        print(f"Failed after {MAX_RETRY_NUM} retries.")
        return 'Error'

    def _judge_one(self, task):    
        """评估单个偏好对"""
        dialog_history = self._format_dialog_history(task['dialog_history'])
        chosen_response = task['chosen']
        rejected_response = task['rejected']
        # 为每个criteria添加编号，格式为：标准X: [标准内容]
        criteria_list = [f"标准{i+1}: {crit[0]}" for i, crit in enumerate(task['criteria'])]
        criteria = "\n".join(criteria_list)
        inputs = (dialog_history, chosen_response, rejected_response, criteria)
        completion = self._judge_by_gpt(inputs)
        task['judge_response'] = completion
        return task

    def _judged_parallel(self, tasks, para_num):
        """多线程并行评估"""
        results = []
        with ThreadPoolExecutor(para_num) as executor:
            for entry in tqdm(executor.map(self._judge_one, tasks), total=len(tasks), desc=f'eval preference pair'):
                results.append(entry)
                if len(results) % 500 == 0: 
                    print(f"results:{results[-1]}")
        if None in results:
            raise ValueError("Some entries are not annotated due to errors in judge_one, please inspect and retry.")
        return results

    def _judged(self, tasks, para_num):
        """带重试机制的评估"""
        MAX_RETRY_NUM = 5
        tasks_remained, tasks_judged = tasks, []
        for _ in range(MAX_RETRY_NUM):
            tasks_judged_ = self._judged_parallel(tasks_remained, para_num)         
            tasks_remained = []
            for task in tasks_judged_:
                if self._get_judge_results(task):
                    tasks_judged.append(task)
                    task['judge_result'] = self._get_judge_results(task)
                    task['judge_parsing'] = "1"
                else:
                    tasks_remained.append(task)

            if len(tasks_remained) == 0:
                break
            else:
                print(f"try:{_}/{MAX_RETRY_NUM}, Still {len(tasks_remained)} tasks remained to be judged. try...")

        if len(tasks_remained) > 0:
            print(f"Max retries ({MAX_RETRY_NUM}) reached. The model's response may lack a valid answer.")
            for task in tasks_remained:
                task['judge_result'] = self._get_judge_results_guaranteed(task)
                if not task['judge_result']:
                    task['judge_result'] = [0] * len(task["criteria"])  # 默认全部为0
                    task['judge_parsing'] = "-1"
                else:
                    task['judge_parsing'] = "0"
                tasks_judged.append(task)
        
        assert len(tasks_judged) == len(tasks), "The number of judged tasks doesn't match the input tasks."
        return tasks_judged

    def _compute_statistics(self, task):
        """计算统计信息"""
        judge_result = task["judge_result"]
        chosen_wins = sum(1 for result in judge_result if result == 1)  # chosen更优
        rejected_wins = sum(1 for result in judge_result if result == 2)  # rejected更优
        ties = sum(1 for result in judge_result if result == 0)  # 平局
        
        # 判断整体胜负
        if chosen_wins > rejected_wins:
            overall_winner = "chosen"
        elif rejected_wins > chosen_wins:
            overall_winner = "rejected"
        else:
            overall_winner = "tie"
        
        stats = {
            "chosen_wins": chosen_wins,
            "rejected_wins": rejected_wins,
            "ties": ties,
            "overall_winner": overall_winner
        }
        return stats

    def _safe_mean(self, arr):
        """安全地计算均值，当数组为空时返回0"""
        if len(arr) == 0:
            return 0.0
        return np.mean(arr)

    def _scores(self, results):
        """计算整体统计分数"""
        chosen_wins_count, rejected_wins_count, ties_count = 0, 0, 0
        
        for exm in results:
            stats = exm["statistics"]
            # 统计胜负情况
            if stats["overall_winner"] == "chosen":
                chosen_wins_count += 1
            elif stats["overall_winner"] == "rejected":
                rejected_wins_count += 1
            else:
                ties_count += 1
        
        total_samples = len(results)
        chosen_win_rate = round(chosen_wins_count / total_samples, 3) if total_samples > 0 else 0
        rejected_win_rate = round(rejected_wins_count / total_samples, 3) if total_samples > 0 else 0
        tie_rate = round(ties_count / total_samples, 3) if total_samples > 0 else 0

        current_time = arrow.now().to('Asia/Shanghai').format('YYYY-MM-DD-HH:mm:ss')
        final_score = {
            "time": [current_time], 
            "name": ["preference_pair"],
            "chosen_win_rate": [chosen_win_rate], 
            "rejected_win_rate": [rejected_win_rate], 
            "tie_rate": [tie_rate],
            "total_samples": [total_samples]
        }
        print(f"final_score:{final_score}")
        return final_score

    def __call__(self):
        """主执行函数"""
        start_time = time.time()
        
        # 重置模型统计信息
        self.eval_model.reset_stats()
        
        # 读取数据
        data = []
        with open(self.in_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        # 评估
        judge_results = self._judged(data, self.para_num)
        
        # 计算每个结果的统计信息
        results = []
        for task in judge_results:
            task['statistics'] = self._compute_statistics(task)
            results.append(task)
        
        # 保存结果
        if not os.path.exists(os.path.dirname(self.out_path)):
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        
        with open(self.out_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        # 计算统计分数
        final_score = self._scores(results)
        
        # 获取token使用统计
        usage_stats = self.eval_model.get_usage_stats()
        print(f"\n=== Token使用统计 ===")
        print(f"总输入tokens: {usage_stats['total_input_tokens']:,}")
        print(f"总输出tokens: {usage_stats['total_output_tokens']:,}")
        print(f"总tokens: {usage_stats['total_tokens']:,}")
        print(f"总成本: ${usage_stats['total_cost']:.4f}")
        print(f"输入价格: ${usage_stats['input_price_per_million']}/M tokens")
        print(f"输出价格: ${usage_stats['output_price_per_million']}/M tokens")
        
        # 将成本信息添加到最终分数中
        final_score["total_input_tokens"] = [usage_stats['total_input_tokens']]
        final_score["total_output_tokens"] = [usage_stats['total_output_tokens']]
        final_score["total_tokens"] = [usage_stats['total_tokens']]
        final_score["total_cost_usd"] = [usage_stats['total_cost']]
        
        end_time = time.time()
        print(f"**** Preference Pair Evaluation Done, Total Cost {end_time-start_time} s, API Cost ${usage_stats['total_cost']:.4f} ****")


class SAGEPreferencePairEvaluationSingleThread():
    """单线程版本的评估类，用于调试"""
    def __init__(self, input_path, output_path, eval_model="gpt4o", temperature=0.01):
        self.eval_model = eval_model
        self.in_path = os.path.join(input_path)
        self.out_path = os.path.join(output_path)
        self.temperature = temperature
        self.eval_model = self._get_model(self.eval_model)
        print(f"in_path:{self.in_path}\t out_path:{self.out_path}\t temperature:{self.temperature}")

    def _get_model(self, model_name):
        try:
            module = importlib.import_module(f"models.{model_name}")
            model_class = getattr(module, model_name)
            return model_class(temperature=self.temperature)
        except (ImportError, AttributeError) as e:
            raise ValueError(f"{model_name} is not defined: {e}, Please ensure that module_name is equal to model_name and that both have been defined.")
        except Exception as e:
            print(f'error:{e}')

    def _format_dialog_history(self, dialog_history):
        """格式化对话历史为易读格式"""
        formatted = []
        for i, msg in enumerate(dialog_history):
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def _get_judge_results(self, task):
        """解析judge_response，提取0/1/2的评估结果"""
        judge_response = task["judge_response"]
        criteria_count = len(task["criteria"])
        results = []
        try:
            judges = judge_response.strip().split("\n")
            # 过滤空行
            judges = [j for j in judges if j.strip()]
            
            assert len(judges) == criteria_count, f"评估条目数量不匹配: 期望{criteria_count}个，实际{len(judges)}个"
            
            for idx, judge in enumerate(judges):
                parts = judge.split("\t")
                assert len(parts) == 2, f"格式错误，无法用\\t分割: {judge}"
                
                crit_part, result = parts[0].strip(), parts[1].strip()
                
                # 只验证是否以"标准X:"开头，不验证具体内容
                expected_prefix = f"标准{idx+1}:"
                assert crit_part.startswith(expected_prefix), f"标准编号错误: 期望以'{expected_prefix}'开头，实际为'{crit_part[:20]}...'"
                
                # 验证结果
                result_val = int(result)
                assert result_val in [0, 1, 2], f"结果必须为0、1或2，实际为{result_val}"
                
                results.append(result_val)
                
        except Exception as e:
            print(f"Judge parsing error: {e}")
            return None
        return results

    def _judge_by_gpt(self, inputs):
        """调用GPT进行偏好对评估"""
        dialog_history, chosen_response, rejected_response, criteria = inputs
        formated_input = sage_judge_for_cfbench(dialog_history, chosen_response, rejected_response, criteria)
        MAX_RETRY_NUM = 10
        for i in range(MAX_RETRY_NUM):
            try:
                response = self.eval_model(formated_input)
                return response
            except Exception as e:
                print(f"Error in gpt judge, retrying...{i}/{MAX_RETRY_NUM}")
                print(e)
                continue
        print(f"Failed after {MAX_RETRY_NUM} retries.")
        return 'Error'

    def _judge_one(self, task):    
        """评估单个偏好对"""
        dialog_history = self._format_dialog_history(task['dialog_history'])
        chosen_response = task['chosen']
        rejected_response = task['rejected']
        # 为每个criteria添加编号，格式为：标准X: [标准内容]
        criteria_list = [f"标准{i+1}: {crit[0]}" for i, crit in enumerate(task['criteria'])]
        criteria = "\n".join(criteria_list)
        inputs = (dialog_history, chosen_response, rejected_response, criteria)
        completion = self._judge_by_gpt(inputs)
        task['judge_response'] = completion
        return task

    def _compute_statistics(self, task):
        """计算统计信息"""
        judge_result = task["judge_result"]
        chosen_wins = sum(1 for result in judge_result if result == 1)  # chosen更优
        rejected_wins = sum(1 for result in judge_result if result == 2)  # rejected更优
        ties = sum(1 for result in judge_result if result == 0)  # 平局
        
        # 判断整体胜负
        if chosen_wins > rejected_wins:
            overall_winner = "chosen"
        elif rejected_wins > chosen_wins:
            overall_winner = "rejected"
        else:
            overall_winner = "tie"
        
        stats = {
            "chosen_wins": chosen_wins,
            "rejected_wins": rejected_wins,
            "ties": ties,
            "overall_winner": overall_winner
        }
        return stats

    def __call__(self):
        """主执行函数 - 单线程版本"""
        start_time = time.time()
        
        # 重置模型统计信息
        self.eval_model.reset_stats()
        
        # 读取数据
        data = []
        with open(self.in_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        # 单线程评估
        results = []
        for i, task in enumerate(tqdm(data, desc=f'eval preference pair (single thread)')):
            try:
                # 评估
                task = self._judge_one(task)
                
                # 解析结果
                judge_result = self._get_judge_results(task)
                if judge_result is None:
                    print(f"Warning: Failed to parse judge result for task {i}, using default values")
                    judge_result = [0] * len(task["criteria"])
                
                task['judge_result'] = judge_result
                task['statistics'] = self._compute_statistics(task)
                results.append(task)
                
                if (i + 1) % 100 == 0:
                    print(f"Processed {i + 1}/{len(data)} tasks")
                    
            except Exception as e:
                print(f"Error processing task {i}: {e}")
                # 添加默认值
                task['judge_result'] = [0] * len(task["criteria"])
                task['statistics'] = self._compute_statistics(task)
                results.append(task)
        
        # 保存结果
        if not os.path.exists(os.path.dirname(self.out_path)):
            os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        
        with open(self.out_path, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
        
        # 计算统计分数
        chosen_wins_count = sum(1 for r in results if r["statistics"]["overall_winner"] == "chosen")
        rejected_wins_count = sum(1 for r in results if r["statistics"]["overall_winner"] == "rejected")
        ties_count = sum(1 for r in results if r["statistics"]["overall_winner"] == "tie")
        
        total_samples = len(results)
        chosen_win_rate = round(chosen_wins_count / total_samples, 3) if total_samples > 0 else 0
        rejected_win_rate = round(rejected_wins_count / total_samples, 3) if total_samples > 0 else 0
        tie_rate = round(ties_count / total_samples, 3) if total_samples > 0 else 0

        current_time = arrow.now().to('Asia/Shanghai').format('YYYY-MM-DD-HH:mm:ss')
        final_score = {
            "time": [current_time], 
            "name": ["preference_pair_single"],
            "chosen_win_rate": [chosen_win_rate], 
            "rejected_win_rate": [rejected_win_rate], 
            "tie_rate": [tie_rate],
            "total_samples": [total_samples]
        }
        print(f"final_score:{final_score}")
        
        # 获取token使用统计
        usage_stats = self.eval_model.get_usage_stats()
        print(f"\n=== Token使用统计 ===")
        print(f"总输入tokens: {usage_stats['total_input_tokens']:,}")
        print(f"总输出tokens: {usage_stats['total_output_tokens']:,}")
        print(f"总tokens: {usage_stats['total_tokens']:,}")
        print(f"总成本: ${usage_stats['total_cost']:.4f}")
        
        end_time = time.time()
        print(f"**** Single Thread Preference Pair Evaluation Done, Total Cost {end_time-start_time} s, API Cost ${usage_stats['total_cost']:.4f} ****")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--infer_model", type=str, default="deepseek_3")
    parser.add_argument("--input_path", type=str, default="dataset/train_dpo_with_checklist_after1000.jsonl")
    # parser.add_argument("--in_dir", type=str, default="")
    parser.add_argument("--output_path", type=str, default="dataset/train_dpo_with_checklist_after1000_final.jsonl")
    parser.add_argument("--max_threads", type=int, default=10)
    parser.add_argument("--eval_model", type=str, default='deepseek_v3')
    parser.add_argument("--single_thread", action="store_true", help="使用单线程版本进行调试")
    args = parser.parse_args()

    if args.single_thread:
        evaluation = SAGEPreferencePairEvaluationSingleThread(
            args.input_path, args.output_path,  args.eval_model
        )
    else:
        evaluation = SAGEPreferencePairEvaluation(
            args.input_path, args.output_path,  args.max_threads, args.eval_model
        )
    
    evaluation()