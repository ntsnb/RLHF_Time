#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek API调用工具 (并发版本)
提供简单的接口来调用DeepSeek API进行文本处理

主要功能：
1. call_deepseek_jsonl() - 返回JSONL格式的API回复: {"content":..., "role":...}
2. call_deepseek_with_separation_jsonl() - 分离<answer>和<think>标签内容
3. process_directory_batch_concurrent() - 并发批处理生成两个JSONL文件
   - answers_xxx.jsonl: 包含数值回答内容（<answer>标签间的内容）
   - reasoning_xxx.jsonl: 包含推理内容（<think>标签间的内容）

并发处理特点：
- 使用ThreadPoolExecutor实现真正的并发API调用
- 按原始索引保存结果，确保即使某些请求慢也不会影响顺序
- 两个文件中answer和reasoning严格按行一一对应
"""

import requests
import json
import os
import re
import threading
from typing import Optional, Dict, Any, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


def extract_content_from_tags(content: str) -> tuple:
    """
    从内容中提取<answer>标签和<think>标签的内容
    
    Args:
        content: 原始内容
        
    Returns:
        (answer_content, reasoning_content) 元组
    """
    # 提取<answer>标签间的数值回答
    answer_pattern = r'<answer>(.*?)</answer>'
    answer_match = re.search(answer_pattern, content, re.DOTALL)
    answer_content = answer_match.group(1).strip() if answer_match else ""
    
    # 提取<think>标签间的推理内容
    reasoning_pattern = r'<think>(.*?)</think>'
    reasoning_match = re.search(reasoning_pattern, content, re.DOTALL)
    reasoning_content = reasoning_match.group(1).strip() if reasoning_match else ""
    
    return answer_content, reasoning_content


def save_to_jsonl(data: Dict[str, Any], filename: str) -> bool:
    """
    保存数据到JSONL文件（每行一个JSON对象）
    
    Args:
        data: 要保存的数据
        filename: 文件名
        
    Returns:
        是否保存成功
    """
    try:
        with open(filename, 'a', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')
        return True
    except Exception as e:
        print(f"保存文件 {filename} 失败: {e}")
        return False


def extract_question_from_data(data):
    """从数据中提取问题文本"""
    if isinstance(data, dict):
        # 尝试不同的字段名
        question = data.get('question') or data.get('text') or data.get('content') or data.get('prompt')
        if question:
            return str(question).strip()
        
        # 如果是API格式的消息
        if 'messages' in data:
            for msg in data['messages']:
                if msg.get('role') == 'user':
                    return str(msg.get('content', '')).strip()
    elif isinstance(data, str):
        return data.strip()
    
    return None


class DeepSeekAPI:
    """DeepSeek API客户端"""
    
    def __init__(self, api_key: str = None, base_url: str = "https://api.deepseek.com/v1"):
        """
        初始化DeepSeek API客户端
        
        Args:
            api_key: DeepSeek API密钥，如果为None则从环境变量DEEPSEEK_API_KEY读取
            base_url: API基础URL
        """
        self.api_key = api_key or os.getenv('DEEPSEEK_API_KEY')
        self.base_url = base_url.rstrip('/')
        self.chat_completions_url = f"{self.base_url}/chat/completions"
        
        if not self.api_key:
            raise ValueError("API密钥未提供。请设置DEEPSEEK_API_KEY环境变量或直接传入api_key参数")
    
    def _get_headers(self) -> Dict[str, str]:
        """获取API请求头"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion(
        self, 
        message: str, 
        model: str = "deepseek-reasoner",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        system_prompt: str = "你是一个有用的AI助手。"
    ) -> Dict[str, Any]:
        """
        发送聊天请求到DeepSeek API
        
        Args:
            message: 用户消息
            model: 使用的模型名称
            temperature: 温度参数，控制随机性
            max_tokens: 最大输出token数
            system_prompt: 系统提示词
            
        Returns:
            API响应字典
        """
        url = self.chat_completions_url
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user", 
                    "content": message
                }
            ],
            "temperature": temperature
        }
        
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        try:
            response = requests.post(
                url, 
                headers=self._get_headers(),
                json=payload,
                timeout=180  # 增加到3分钟，对于R1模型的推理时间
            )
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {
                "error": True,
                "message": f"API请求失败: {str(e)}",
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }
        except json.JSONDecodeError as e:
            return {
                "error": True,
                "message": f"JSON解析失败: {str(e)}",
                "status_code": getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None
            }


def call_deepseek(text: str, api_key: str = None) -> str:
    """
    便捷函数：调用DeepSeek API并返回结果
    
    Args:
        text: 输入的文本
        api_key: API密钥（可选）
        
    Returns:
        DeepSeek的回复文本，如果出错则返回错误信息
    """
    try:
        client = DeepSeekAPI(api_key=api_key)
        response = client.chat_completion(text)
        
        if response.get("error"):
            return f"错误: {response.get('message', '未知错误')}"
        
        # 提取回复内容
        choices = response.get("choices", [])
        if choices:
            return choices[0]["message"]["content"]
        else:
            return "未能获取到回复内容"
            
    except Exception as e:
        return f"调用失败: {str(e)}"


def call_deepseek_jsonl(text: str, api_key: str = None) -> str:
    """
    便捷函数：调用DeepSeek API并返回JSONL格式结果
    
    Args:
        text: 输入的文本
        api_key: API密钥（可选）
        
    Returns:
        JSONL格式的字符串：{"content":..., "role":...}
    """
    try:
        client = DeepSeekAPI(api_key=api_key)
        response = client.chat_completion(text)
        
        if response.get("error"):
            return json.dumps({"content": f"错误: {response.get('message', '未知错误')}", "role": "assistant"}, ensure_ascii=False)
        
        # 提取回复内容
        choices = response.get("choices", [])
        if choices:
            message = choices[0]["message"]
            content = message.get("content", "")
            
            # 返回JSONL格式
            return json.dumps({"content": content, "role": "assistant"}, ensure_ascii=False)
        else:
            return json.dumps({"content": "未能获取到回复内容", "role": "assistant"}, ensure_ascii=False)
            
    except Exception as e:
        return json.dumps({"content": f"调用失败: {str(e)}", "role": "assistant"}, ensure_ascii=False)


def call_deepseek_with_separation_jsonl(text: str, api_key: str = None) -> Dict[str, str]:
    """
    便捷函数：调用DeepSeek R1 API并分离推理过程和最终答案，返回JSONL格式
    
    Args:
        text: 输入的文本
        api_key: API密钥（可选）
        
    Returns:
        包含answer_content和reasoning_content的字典
    """
    try:
        client = DeepSeekAPI(api_key=api_key)
        response = client.chat_completion(text)
        
        if response.get("error"):
            return {
                "answer_content": f"错误: {response.get('message', '未知错误')}",
                "reasoning_content": ""
            }
        
        # 提取回复内容
        choices = response.get("choices", [])
        if choices:
            message = choices[0]["message"]
            content = message.get("content", "")
            
            # 从完整回复中提取answer和reasoning内容
            answer_content, reasoning_content = extract_content_from_tags(content)
            
            # 如果没有找到标签，尝试使用API的reasoning_content字段
            if not reasoning_content:
                reasoning_content = message.get("reasoning_content", "")
            
            # 如果仍然没有answer内容，使用完整content
            if not answer_content:
                answer_content = content
                
            return {
                "answer_content": answer_content,
                "reasoning_content": reasoning_content
            }
        else:
            return {
                "answer_content": "未能获取到回复内容",
                "reasoning_content": ""
            }
            
    except Exception as e:
        return {
            "answer_content": f"调用失败: {str(e)}",
            "reasoning_content": ""
        }


def process_single_question(question_data: Dict[str, Any], api_key: str) -> tuple:
    """
    处理单个问题，返回(answer_content, reasoning_content)的元组
    
    Args:
        question_data: 包含问题的数据
        api_key: API密钥
        
    Returns:
        (answer_content, reasoning_content) 元组
    """
    try:
        result = call_deepseek_with_separation_jsonl(question_data['question'], api_key)
        return result["answer_content"], result["reasoning_content"]
    except Exception as e:
        return f"处理失败: {str(e)}", ""


def process_directory_batch_concurrent(
    directory_path: str, 
    api_key: str = None, 
    output_dir: str = None,
    max_workers: int = 5
):
    """
    并发批量处理目录中的JSONL文件和JSON文件，生成两个简洁的JSONL文件
    
    核心特点：使用预分配结果数组 + 按原始索引保存，确保即使某些请求慢也不会影响顺序
    
    Args:
        directory_path: 要处理的目录路径
        api_key: API密钥
        output_dir: 输出目录，默认为与输入目录相同
        max_workers: 最大并发线程数
    """
    import glob
    
    if output_dir is None:
        output_dir = directory_path
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有JSONL和JSON文件
    jsonl_files = glob.glob(os.path.join(directory_path, "*.jsonl"))
    json_files = glob.glob(os.path.join(directory_path, "*.json"))
    all_files = jsonl_files + json_files
    
    if not all_files:
        print(f"在目录 {directory_path} 中没有找到JSONL或JSON文件")
        return
    
    print(f"找到 {len(all_files)} 个文件（{len(jsonl_files)} 个JSONL文件，{len(json_files)} 个JSON文件），开始并发批处理...")
    
    # 准备输出文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    answer_file = os.path.join(output_dir, f"answers_{timestamp}.jsonl")
    reasoning_file = os.path.join(output_dir, f"reasoning_{timestamp}.jsonl")
    
    # 清除之前的文件（如果存在）
    for file_path in [answer_file, reasoning_file]:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # 收集所有问题
    all_questions = []
    
    # 处理每个文件
    for i, file_path in enumerate(all_files, 1):
        filename = os.path.basename(file_path)
        print(f"\n[{i}/{len(all_files)}] 读取文件: {filename}")
        
        try:
            if filename.endswith('.jsonl'):
                # 处理JSONL文件 - 每行一个JSON对象
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                            question = extract_question_from_data(data)
                            if question:
                                all_questions.append({
                                    "question": question,
                                    "source": f"{filename}:{line_num}"
                                })
                        except json.JSONDecodeError as e:
                            print(f"    ⚠️  第{line_num}行JSON解析失败，跳过")
                            continue
            else:
                # 处理JSON文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                        if isinstance(data, list):
                            for item in data:
                                question = extract_question_from_data(item)
                                if question:
                                    all_questions.append({
                                        "question": question,
                                        "source": f"{filename}:1"
                                    })
                        else:
                            question = extract_question_from_data(data)
                            if question:
                                all_questions.append({
                                    "question": question,
                                    "source": f"{filename}:1"
                                })
                    except json.JSONDecodeError as e:
                        print(f"    ❌ JSON文件解析失败: {e}")
                        continue
                        
        except Exception as e:
            print(f"    ❌ 处理文件 {file_path} 时出错: {e}")
            continue
    
    if not all_questions:
        print("❌ 没有找到有效问题")
        return
    
    print(f"\n📋 总共找到 {len(all_questions)} 个问题，开始并发处理（最大并发数: {max_workers}）...")
    
    # 关键改进：预分配结果数组，确保按原始顺序保存
    results = [None] * len(all_questions)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，并保存原始索引
        futures = []
        for i, question_data in enumerate(all_questions):
            future = executor.submit(process_single_question, question_data, api_key)
            futures.append((i, future))  # 保存原始索引
        
        completed = 0
        for original_index, future in futures:
            completed += 1
            
            try:
                answer_content, reasoning_content = future.result()
                # 按原始索引保存结果 - 这确保了即使第3个任务比第1个任务先完成，
                # 结果仍然保存在正确的位置
                results[original_index] = (answer_content, reasoning_content)
                
                if completed % 10 == 0 or completed == len(all_questions):
                    print(f"✅ 并发处理进度: {completed}/{len(all_questions)} ({completed/len(all_questions)*100:.1f}%)")
                    
            except Exception as e:
                print(f"❌ 处理问题失败: {str(e)} (索引: {original_index})")
                # 按索引保存失败结果
                results[original_index] = (f"处理失败: {str(e)}", "")
    
    print(f"\n📝 按原始顺序保存结果到文件...")
    
    # 按原始顺序保存到文件 - 这里保证了最终的文件顺序与输入顺序一致
    for i, (answer_content, reasoning_content) in enumerate(results):
        # 保存数值回答
        answer_data = {
            "content": answer_content,
            "role": "assistant"
        }
        save_to_jsonl(answer_data, answer_file)
        
        # 保存推理内容
        reasoning_data = {
            "content": reasoning_content,
            "role": "assistant"
        }
        save_to_jsonl(reasoning_data, reasoning_file)
        
        if (i + 1) % 50 == 0 or (i + 1) == len(results):
            print(f"📁 文件保存进度: {i + 1}/{len(results)} ({(i + 1)/len(results)*100:.1f}%)")
    
    print(f"\n🎉 并发批处理完成！")
    print(f"📁 数值回答文件: {answer_file}")
    print(f"📁 推理内容文件: {reasoning_file}")
    print(f"📊 处理完成: {len(all_questions)} 个样本")
    print(f"✅ 结果已按原始顺序保存，确保answer和reasoning在两个文件中一一对应")
    print(f"🔧 顺序保证机制: 预分配结果数组 + 按原始索引保存，即使某些请求慢也不会影响顺序")


def demo_usage():
    """演示如何使用新功能"""
    api_key = "sk-2d89cfdfa645428ebd5af2a8b5c5df72"
    
    print("=== DeepSeek API 并发版本使用示例 ===\n")
    
    # 示例1: 单次查询JSONL格式
    print("1. 单次查询JSONL格式:")
    question = "请计算 2+3 的结果"
    result = call_deepseek_jsonl(question, api_key)
    print(f"输入问题: {question}")
    print(f"JSONL结果: {result}\n")
    
    # 示例2: 分离内容
    print("2. 分离<answer>和<think>标签内容:")
    result = call_deepseek_with_separation_jsonl(question, api_key)
    print(f"输入问题: {question}")
    print(f"数值回答: {result['answer_content']}")
    print(f"推理内容: {result['reasoning_content']}\n")
    
    # 示例3: 并发批处理（需要准备输入目录）
    print("3. 并发批处理目录:")
    print("   请先在当前目录创建包含JSON/JSONL文件的目录")
    print("   然后调用: process_directory_batch_concurrent('你的目录路径', api_key, max_workers=5)")
    print("   将生成两个文件: answers_时间戳.jsonl 和 reasoning_时间戳.jsonl")
    print("   并发处理提高速度，同时保证两个文件中answer和reasoning按行一一对应")


def main():
    """主函数，提供交互界面和批处理功能"""
    print("=== DeepSeek R1 API 工具 (并发版本) ===")
    print("功能：并发调用DeepSeek API并返回JSONL格式，支持高效批处理\n")
    
    # 选择模式
    print("请选择操作模式:")
    print("1. 单次查询模式 (JSONL格式)")
    print("2. 原始模式 (简单文本回复)")
    print("3. 并发批处理模式（处理目录中的JSON文件，生成两个JSONL文件）")
    print("4. 使用示例")
    
    mode = input("请输入选择（1、2、3或4）: ").strip()
    
    # 检查API密钥
    api_key = "sk-2d89cfdfa645428ebd5af2a8b5c5df72"
    
    try:
        client = DeepSeekAPI(api_key=api_key)
        print("✅ API连接成功！\n")
        
        if mode == "1":
            # 单次查询模式 - JSONL格式
            print("=== 单次查询模式 (JSONL格式) ===")
            while True:
                print("\n" + "="*60)
                user_input = input("请输入您的问题（输入'quit'退出）: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                    print("再见！")
                    break
                
                if not user_input:
                    print("请输入有效内容！")
                    continue
                
                print("\n🔄 正在思考中...")
                result = call_deepseek_jsonl(user_input, api_key)
                
                print(f"\n📋 JSONL格式结果:")
                print(result)
        
        elif mode == "2":
            # 原始模式
            print("=== 原始模式 ===")
            while True:
                print("\n" + "="*50)
                user_input = input("请输入您的问题（输入'quit'退出）: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q', '退出']:
                    print("再见！")
                    break
                
                if not user_input:
                    print("请输入有效内容！")
                    continue
                
                print("\n🔄 正在思考中...")
                response = call_deepseek(user_input, api_key)
                print(f"\n💬 DeepSeek回复: {response}")
                
        elif mode == "3":
            # 并发批处理模式
            print("\n=== 并发批处理模式 ===")
            directory_path = input("请输入要处理的目录路径: ").strip()
            
            if not directory_path:
                print("❌ 目录路径不能为空")
                return
            
            if not os.path.exists(directory_path):
                print(f"❌ 目录 {directory_path} 不存在")
                return
            
            output_dir = input("请输入输出目录路径（直接回车使用输入目录）: ").strip()
            if not output_dir:
                output_dir = None
            
            max_workers = input("请输入最大并发数（直接回车使用5）: ").strip()
            if not max_workers:
                max_workers = 5
            else:
                try:
                    max_workers = int(max_workers)
                except ValueError:
                    max_workers = 5
            
            print(f"\n🚀 开始并发批处理目录: {directory_path}")
            print(f"最大并发数: {max_workers}")
            print("🔧 顺序保证: 预分配结果数组 + 按原始索引保存，确保结果顺序正确")
            process_directory_batch_concurrent(directory_path, api_key, output_dir, max_workers)
            
        elif mode == "4":
            demo_usage()
            
        else:
            print("❌ 无效的选择，请重新运行程序")
            
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        print("请确保:")
        print("1. 提供了有效的API密钥")
        print("2. 网络连接正常")
        print("3. API服务可用")


if __name__ == "__main__":
    main()