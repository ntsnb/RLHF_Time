#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
合并两个jsonl文件，将对应行的content内容拼接
输入文件格式：{"content": ..., "role": ...}
输出文件格式：{"content": "拼接后的内容", "role": ...}
"""

import json

def merge_jsonl_files(file1_path, file2_path, output_path, separator="\n\n"):
    """
    合并两个jsonl文件
    
    Args:
        file1_path: 第一个jsonl文件路径
        file2_path: 第二个jsonl文件路径  
        output_path: 输出文件路径
        separator: 拼接时的分隔符
    """
    
    # 读取第一个文件
    with open(file1_path, 'r', encoding='utf-8') as f1:
        lines1 = f1.readlines()
    
    # 读取第二个文件
    with open(file2_path, 'r', encoding='utf-8') as f2:
        lines2 = f2.readlines()
    
    # 检查两个文件行数是否一致
    if len(lines1) != len(lines2):
        raise ValueError(f"两个文件的行数不一致: 文件1有{len(lines1)}行，文件2有{len(lines2)}行")
    
    # 合并数据
    merged_data = []
    
    for i, (line1, line2) in enumerate(zip(lines1, lines2)):
        try:
            # 解析JSON
            data1 = json.loads(line1.strip())
            data2 = json.loads(line2.strip())
            
            # 拼接content
            merged_content = data1['content'] + separator + data2['content']
            
            # 创建新的记录，保持第一个文件的role
            merged_record = {
                'content': merged_content,
                'role': data1['role']
            }
            
            merged_data.append(merged_record)
            
        except json.JSONDecodeError as e:
            print(f"警告：第{i+1}行JSON解析失败")
            print(f"文件1: {line1}")
            print(f"文件2: {line2}")
            print(f"错误: {e}")
            continue
    
    # 写入合并后的文件
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for record in merged_data:
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    print(f"合并完成！")
    print(f"输入文件1: {file1_path}")
    print(f"输入文件2: {file2_path}")
    print(f"输出文件: {output_path}")
    print(f"合并了 {len(merged_data)} 行数据")

def main():
    # 文件路径
    file1 = "/mnt/sda/home/niutiansen/RLHF_time/dataset_test/answers_20251127_195800.jsonl"
    file2 = "/mnt/sda/home/niutiansen/RLHF_time/dataset_test/reasoning_20251127_195800.jsonl"
    output = "merged_output.jsonl"
    
    try:
        merge_jsonl_files(file1, file2, output)
        print(f"\n✅ 合并成功完成！")
        print(f"📁 输出文件: {output}")
        
        # 显示第一行合并结果作为示例
        print(f"\n📄 预览合并结果（第一行）:")
        with open(output, 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()
            data = json.loads(first_line)
            print(f"Content长度: {len(data['content'])} 字符")
            print(f"Role: {data['role']}")
            
    except Exception as e:
        print(f"❌ 合并失败: {e}")

if __name__ == "__main__":
    main()