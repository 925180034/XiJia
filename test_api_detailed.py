"""
详细的API测试脚本 - 诊断API调用问题
"""
import yaml
import requests
import json
import time

def load_config():
    """加载配置"""
    with open("config/config_enhanced.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def test_api_basic():
    """基础API测试"""
    print("=== 基础API测试 ===")
    
    config = load_config()
    api_key = config["openai"]["api_key"]
    api_base_url = config["openai"]["api_base_url"]
    model = config["openai"]["model"]
    
    print(f"API Key: {api_key[:10]}...{api_key[-5:]}")
    print(f"API Base URL: {api_base_url}")
    print(f"Model: {model}")
    
    # 测试API连接
    api_url = f"{api_base_url}/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello, test connection"}],
        "temperature": 0.1,
        "max_tokens": 50
    }
    
    try:
        print(f"\n发送请求到: {api_url}")
        response = requests.post(api_url, headers=headers, json=data, timeout=30)
        
        print(f"响应状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            print("✅ API调用成功")
            resp_json = response.json()
            if "choices" in resp_json:
                content = resp_json["choices"][0]["message"]["content"]
                print(f"响应内容: {content}")
            return True
        else:
            print("❌ API调用失败")
            print(f"错误响应: {response.text}")
            
            # 分析具体错误
            try:
                error_json = response.json()
                if "error" in error_json:
                    error_msg = error_json["error"].get("message", "未知错误")
                    error_type = error_json["error"].get("type", "未知类型")
                    print(f"错误类型: {error_type}")
                    print(f"错误信息: {error_msg}")
            except:
                pass
            
            return False
    except Exception as e:
        print(f"❌ 请求异常: {e}")
        return False

def test_different_models():
    """测试不同的模型"""
    print("\n=== 测试不同模型 ===")
    
    config = load_config()
    api_key = config["openai"]["api_key"]
    api_base_url = config["openai"]["api_base_url"]
    
    # 常见的模型列表
    models_to_test = [
        "gpt-4o",
        "gpt-4o-mini", 
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k"
    ]
    
    api_url = f"{api_base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    for model in models_to_test:
        print(f"\n测试模型: {model}")
        
        data = {
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0.1,
            "max_tokens": 10
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=15)
            if response.status_code == 200:
                print(f"✅ {model} 可用")
            else:
                print(f"❌ {model} 不可用: {response.status_code}")
                if response.status_code == 404:
                    print("  可能此服务商不支持该模型")
        except Exception as e:
            print(f"❌ {model} 测试异常: {e}")
        
        time.sleep(1)

def test_alternative_headers():
    """测试不同的请求头组合"""
    print("\n=== 测试不同请求头 ===")
    
    config = load_config()
    api_key = config["openai"]["api_key"]
    api_base_url = config["openai"]["api_base_url"]
    
    api_url = f"{api_base_url}/chat/completions"
    
    # 不同的请求头组合
    header_combinations = [
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        },
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "OpenAI/Python"
        },
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json"
        },
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-API-Key": api_key
        }
    ]
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test headers"}],
        "temperature": 0.1,
        "max_tokens": 10
    }
    
    for i, headers in enumerate(header_combinations):
        print(f"\n测试请求头组合 {i+1}: {list(headers.keys())}")
        
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=15)
            if response.status_code == 200:
                print(f"✅ 请求头组合 {i+1} 成功")
                return headers
            else:
                print(f"❌ 请求头组合 {i+1} 失败: {response.status_code}")
        except Exception as e:
            print(f"❌ 请求头组合 {i+1} 异常: {e}")
        
        time.sleep(1)
    
    return None

def test_api_key_formats():
    """测试不同的API Key格式"""
    print("\n=== 测试API Key格式 ===")
    
    config = load_config()
    original_api_key = config["openai"]["api_key"]
    api_base_url = config["openai"]["api_base_url"]
    
    api_url = f"{api_base_url}/chat/completions"
    
    # 不同的API Key格式
    key_formats = [
        original_api_key,  # 原始格式
        original_api_key.strip(),  # 去除空格
    ]
    
    # 如果API key中有特殊字符，尝试不同的编码
    if any(char in original_api_key for char in ['+', '/', '=']):
        import urllib.parse
        key_formats.append(urllib.parse.quote(original_api_key))
    
    data = {
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "test key format"}],
        "temperature": 0.1,
        "max_tokens": 10
    }
    
    for i, api_key in enumerate(key_formats):
        print(f"\n测试API Key格式 {i+1}: {api_key[:10]}...{api_key[-5:]}")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=15)
            if response.status_code == 200:
                print(f"✅ API Key格式 {i+1} 成功")
                return api_key
            else:
                print(f"❌ API Key格式 {i+1} 失败: {response.status_code}")
                print(f"   错误响应: {response.text[:100]}")
        except Exception as e:
            print(f"❌ API Key格式 {i+1} 异常: {e}")
        
        time.sleep(1)
    
    return None

def generate_recommendations():
    """生成解决建议"""
    print("\n" + "="*60)
    print("API问题解决建议")
    print("="*60)
    
    print("\n1. 检查API Key是否有效:")
    print("   - 确认API Key没有过期")
    print("   - 确认API Key有正确的权限")
    print("   - 联系API服务提供商确认Key状态")
    
    print("\n2. 检查服务提供商设置:")
    print("   - 确认xiaoai.plus服务正常")
    print("   - 检查是否有IP限制或地区限制")
    print("   - 确认账户余额充足")
    
    print("\n3. 尝试替代方案:")
    print("   - 使用其他API服务商")
    print("   - 使用纯相似度匹配模式")
    print("   - 联系xiaoai.plus技术支持")
    
    print("\n4. 暂时的解决方案:")
    print("   - 当前系统已使用回退策略")
    print("   - 基于相似度的匹配结果仍然有效")
    print("   - 可以继续使用这些结果")

def main():
    """主函数"""
    print("开始详细API诊断...")
    
    tests = [
        ("基础API测试", test_api_basic),
        ("不同模型测试", test_different_models),
        ("请求头测试", test_alternative_headers),
        ("API Key格式测试", test_api_key_formats)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"测试 {test_name} 出现异常: {e}")
            results[test_name] = False
    
    # 总结结果
    print(f"\n{'='*50}")
    print("诊断结果总结")
    print("="*50)
    
    any_success = False
    for test_name, result in results.items():
        if result:
            status = "✅ 成功"
            any_success = True
        else:
            status = "❌ 失败"
        print(f"{test_name}: {status}")
    
    if not any_success:
        print("\n❌ 所有API测试都失败了")
        generate_recommendations()
    else:
        print("\n✅ 找到了可用的API配置")

if __name__ == "__main__":
    main()