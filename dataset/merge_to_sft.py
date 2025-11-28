#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将user和assistant角色的jsonl文件合并为SFT数据集格式
输入：两个jsonl文件，一个包含user角色，一个包含assistant角色
输出：SFT数据集格式的jsonl文件
"""

import json

def merge_to_sft_format(user_file_path, assistant_file_path, output_path):
    """
    将user和assistant文件合并为SFT格式
    
    Args:
        user_file_path: 包含user角色的jsonl文件路径
        assistant_file_path: 包含assistant角色的jsonl文件路径
        output_path: 输出SFT格式jsonl文件路径
    """
    
    # 读取user文件
    with open(user_file_path, 'r', encoding='utf-8') as f_user:
        user_lines = f_user.readlines()
    
    # 读取assistant文件
    with open(assistant_file_path, 'r', encoding='utf-8') as f_assistant:
        assistant_lines = f_assistant.readlines()
    
    # 检查两个文件行数是否一致
    if len(user_lines) != len(assistant_lines):
        raise ValueError(f"两个文件的行数不一致: user文件有{len(user_lines)}行，assistant文件有{len(assistant_lines)}行")
    
    # 转换为SFT格式
    sft_data = []
    
    for i, (user_line, assistant_line) in enumerate(zip(user_lines, assistant_lines)):
        try:
            # 解析JSON
            user_data = json.loads(user_line.strip())
            assistant_data = json.loads(assistant_line.strip())
            
            # 验证role字段
            if user_data.get('role') != 'user':
                print(f"警告：第{i+1}行user文件的role不是'user'，而是'{user_data.get('role')}'")
            
            if assistant_data.get('role') != 'assistant':
                print(f"警告：第{i+1}行assistant文件的role不是'assistant'，而是'{assistant_data.get('role')}'")
            
            # 创建SFT格式的对话
            conversation = {
                "conversations": [
                    {
                        "role": "user",
                        "content": user_data['content']
                    },
                    {
                        "role": "assistant", 
                        "content": assistant_data['content']
                    }
                ]
            }
            
            sft_data.append(conversation)
            
        except json.JSONDecodeError as e:
            print(f"警告：第{i+1}行JSON解析失败")
            print(f"User文件: {user_line}")
            print(f"Assistant文件: {assistant_line}")
            print(f"错误: {e}")
            continue
    
    # 写入SFT格式文件
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for conversation in sft_data:
            f_out.write(json.dumps(conversation, ensure_ascii=False) + '\n')
    
    print(f"✅ SFT格式合并完成！")
    print(f"📁 User文件: {user_file_path}")
    print(f"📁 Assistant文件: {assistant_file_path}")
    print(f"📁 输出文件: {output_path}")
    print(f"🔢 转换了 {len(sft_data)} 条对话")
    
    return len(sft_data)

def preview_sft_sample(file_path, sample_index=0):
    """预览SFT格式文件的示例"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            if sample_index < len(lines):
                sample = json.loads(lines[sample_index])
                print(f"\n📋 SFT格式示例 (第{sample_index+1}行):")
                print(f"对话轮次数量: {len(sample['conversations'])}")
                for i, turn in enumerate(sample['conversations']):
                    print(f"  轮次 {i+1}: {turn['role']} - 内容长度 {len(turn['content'])} 字符")
                return sample
    except Exception as e:
        print(f"❌ 预览失败: {e}")
    return None

def main():
    # 文件路径
    user_file = "RLHF_time/dataset_test/demo_prompts.jsonl"
    assistant_file = "merged_output.jsonl"
    output_file = "sft_dataset.jsonl"
    
    try:
        # 转换为SFT格式
        count = merge_to_sft_format(user_file, assistant_file, output_file)
        
        # 预览结果
        if count > 0:
            preview_sft_sample(output_file, 0)
            
            # 显示第一行的部分内容作为示例
            print(f"\n📄 内容预览 (第一行开头):")
            with open(output_file, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                data = json.loads(first_line)
                print(f"User: {data['conversations'][0]['content'][:100]}...")
                print(f"Assistant: {data['conversations'][1]['content'][:100]}...")
        
    except Exception as e:
        print(f"❌ 转换失败: {e}")

if __name__ == "__main__":
    main()