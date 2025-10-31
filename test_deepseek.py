#!/usr/bin/env python3
"""测试DeepSeek API连接"""

import os
from dotenv import load_dotenv
from openai import OpenAI

# 加载环境变量
load_dotenv()

# 初始化DeepSeek客户端
deepseek_client = OpenAI(
    api_key=os.getenv('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

def test_deepseek_api():
    """测试DeepSeek API连接"""
    try:
        print("正在测试DeepSeek API连接...")
        print(f"API密钥: {os.getenv('DEEPSEEK_API_KEY')[:10]}...")
        
        response = deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个AI助手。"},
                {"role": "user", "content": "请回复：API连接测试成功"}
            ],
            stream=False,
            temperature=0.1
        )
        
        print("✅ API调用成功！")
        print(f"响应: {response.choices[0].message.content}")
        return True
        
    except Exception as e:
        print(f"❌ API调用失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_deepseek_api()