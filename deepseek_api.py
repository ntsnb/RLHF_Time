#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek API调用工具
提供简单的接口来调用DeepSeek API进行文本处理
"""

import requests
import json
import os
from typing import Optional, Dict, Any


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
                timeout=60
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


def main():
    """主函数，提供简单的交互界面"""
    print("=== DeepSeek API 测试工具 ===")
    
    # 检查API密钥
    api_key = "sk-2d89cfdfa645428ebd5af2a8b5c5df72"

    
    try:
        client = DeepSeekAPI(api_key=api_key)
        print("✅ API连接成功！")
        
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
            
    except Exception as e:
        print(f"❌ 初始化失败: {str(e)}")
        print("请确保:")
        print("1. 提供了有效的API密钥")
        print("2. 网络连接正常")
        print("3. API服务可用")


if __name__ == "__main__":
    main()