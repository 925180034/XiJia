"""
直接使用requests测试API连接
"""
import yaml
import requests
import json

def test_api_connection():
    """测试API连接"""
    print("加载配置...")
    try:
        with open("config/config.yaml", "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        api_key = config["openai"]["api_key"]
        api_base_url = config["openai"].get("api_base_url")
        model = config["openai"]["model"]
        
        print(f"API Key: {api_key[:5]}...{api_key[-4:] if len(api_key) > 8 else ''}")
        print(f"API Base URL: {api_base_url}")
        print(f"Model: {model}")
        
        # 构建请求
        api_url = f"{api_base_url}/chat/completions" if api_base_url else "https://api.openai.com/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": "简单的测试消息：请回复API连接成功"}],
            "temperature": 0.1,
            "max_tokens": 30
        }
        
        print(f"\n发送请求到: {api_url}")
        print("请求数据:", json.dumps(data, ensure_ascii=False, indent=2))
        
        # 发送请求
        response = requests.post(api_url, headers=headers, json=data, timeout=30)
        
        # 检查响应状态
        print(f"\n响应状态码: {response.status_code}")
        
        if response.status_code != 200:
            print(f"API调用失败: {response.text}")
            return
        
        # 打印完整响应以便调试
        print("\n完整响应:")
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        
        # 尝试解析响应中的消息内容
        try:
            resp_json = response.json()
            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                if "message" in resp_json["choices"][0] and "content" in resp_json["choices"][0]["message"]:
                    content = resp_json["choices"][0]["message"]["content"]
                    print(f"\nAPI响应内容: {content}")
                elif "text" in resp_json["choices"][0]:
                    content = resp_json["choices"][0]["text"]
                    print(f"\nAPI响应内容: {content}")
                else:
                    print("\n无法找到响应内容，检查响应结构")
            else:
                print("\n响应中没有choices字段，检查响应结构")
                
            print("\nAPI连接测试完成！")
            
        except Exception as e:
            print(f"\n解析响应失败: {e}")
            
    except Exception as e:
        print(f"API连接测试失败: {e}")

if __name__ == "__main__":
    test_api_connection()