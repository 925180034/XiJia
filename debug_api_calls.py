"""
API调用诊断脚本 - 找出测试成功但实际失败的原因
"""
import yaml
import requests
import json
import time

def load_config():
    """加载配置"""
    with open("config/config_enhanced.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def test_simple_request(config):
    """测试简单请求（与test_api_direct.py相同）"""
    print("=== 测试1: 简单请求 ===")
    
    api_url = f"{config['openai']['api_base_url']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['openai']['api_key']}"
    }
    
    data = {
        "model": config['openai']['model'],
        "messages": [{"role": "user", "content": "简单测试"}],
        "temperature": 0.1,
        "max_tokens": 30
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=30)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("✅ 简单请求成功")
            return True
        else:
            print(f"❌ 简单请求失败: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 简单请求异常: {e}")
        return False

def test_matching_style_request(config):
    """测试匹配风格的请求（模拟实际匹配时的请求）"""
    print("\n=== 测试2: 匹配风格请求 ===")
    
    # 模拟实际匹配时的长提示
    prompt = """系统角色：您是数据集成和Schema匹配专家，擅长分析表结构和字段关系。

任务描述：判断两个字段是否语义等价。每个字段有名称和描述（注释）。

源字段：
- 表名：T_TEST_TABLE
- 表描述：测试表
- 字段名：id
- 字段描述：唯一标识符
- 字段类型：VARCHAR

目标字段：
- 表名：T_TARGET_TABLE
- 表描述：目标表
- 字段名：ID
- 字段描述：主键
- 字段类型：VARCHAR

计算的相似度：1.00

分析问题：
1. 分析字段名称的语义关系（考虑缩写、拼音转换等）
2. 比较字段描述的语义相似度
3. 考虑中英文专业术语对应关系
4. 分析字段在各自表中的作用是否相同

源字段和目标字段是否语义等价？
1. 回答[是/否]
2. 给出判断的置信度（0-1之间的数字）
3. 简要解释理由

请按照以下格式回答：
判断：[是/否]
置信度：[0-1之间的数字]
理由：[简要解释]"""

    api_url = f"{config['openai']['api_base_url']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['openai']['api_key']}"
    }
    
    data = {
        "model": config['openai']['model'],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config['openai']['temperature'],
        "max_tokens": config['openai']['max_tokens']
    }
    
    print(f"请求长度: {len(json.dumps(data))} 字符")
    print(f"提示长度: {len(prompt)} 字符")
    
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=60)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("✅ 匹配风格请求成功")
            result = response.json()
            if "choices" in result:
                content = result["choices"][0]["message"]["content"]
                print(f"响应内容: {content[:100]}...")
            return True
        else:
            print(f"❌ 匹配风格请求失败: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 匹配风格请求异常: {e}")
        return False

def test_with_user_agent(config):
    """测试带User-Agent的请求"""
    print("\n=== 测试3: 带User-Agent请求 ===")
    
    api_url = f"{config['openai']['api_base_url']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['openai']['api_key']}",
        "User-Agent": "Schema-Matching-Tool/1.0"
    }
    
    data = {
        "model": config['openai']['model'],
        "messages": [{"role": "user", "content": "测试User-Agent"}],
        "temperature": 0.1,
        "max_tokens": 30
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=30)
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            print("✅ 带User-Agent请求成功")
            return True
        else:
            print(f"❌ 带User-Agent请求失败: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 带User-Agent请求异常: {e}")
        return False

def test_rapid_requests(config):
    """测试快速连续请求"""
    print("\n=== 测试4: 快速连续请求 ===")
    
    api_url = f"{config['openai']['api_base_url']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['openai']['api_key']}"
    }
    
    success_count = 0
    total_requests = 3
    
    for i in range(total_requests):
        data = {
            "model": config['openai']['model'],
            "messages": [{"role": "user", "content": f"快速请求 {i+1}"}],
            "temperature": 0.1,
            "max_tokens": 20
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            print(f"请求 {i+1} 状态码: {response.status_code}")
            if response.status_code == 200:
                success_count += 1
                print(f"✅ 请求 {i+1} 成功")
            else:
                print(f"❌ 请求 {i+1} 失败: {response.text[:100]}")
            
            # 短暂间隔
            time.sleep(0.5)
            
        except Exception as e:
            print(f"❌ 请求 {i+1} 异常: {e}")
    
    print(f"快速请求结果: {success_count}/{total_requests} 成功")
    return success_count == total_requests

def test_enhanced_llm_style(config):
    """测试增强LLM风格的请求（模拟实际代码）"""
    print("\n=== 测试5: 增强LLM风格请求 ===")
    
    # 完全模拟 EnhancedLLMMatcher 的请求格式
    api_url = f"{config['openai']['api_base_url']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['openai']['api_key']}",
        "User-Agent": "Schema-Matching-Tool/1.0"
    }
    
    # 模拟增强提示
    prompt = """系统角色：您是数据集成专家，擅长识别字段间的语义等价关系。

任务描述：判断两个字段是否语义等价。这两个字段已经通过初步筛选，具有较高的相似度(1.00)，请仔细分析它们是否表示相同的业务概念。

源字段：
- 表名：T_UNKNOWN_xj_code_table_field
- 表描述：(无)
- 字段名：id
- 字段描述：(无)
- 字段类型：(未知)

目标字段：
- 表名：T_UNKNOWN_META_DATA_FIELD
- 表描述：(无)
- 字段名：ID
- 字段描述：(无)
- 字段类型：(未知)

分析要点：
1. 字段名语义关系：考虑中英文对应、拼音缩写（如"XSBH"="学生编号"对应"XH"="学号"）
2. 业务概念匹配：分析字段在业务流程中的作用是否相同
3. 描述语义：比较字段描述的语义相似度
4. 数据类型兼容性：检查数据类型是否兼容

常见等价关系示例：
- ID/编号/代码/标识 → 通常表示唯一标识符
- 姓名/名称/名字 → 通常表示人或物的名称
- 时间/日期/创建时间/更新时间 → 通常表示时间概念
- 状态/类型/种类 → 通常表示分类信息

判断标准：
- 如果两个字段表示相同的业务概念，即使名称不完全相同，也应该判断为匹配
- 中英文字段如果语义相同，应判断为匹配
- 拼音缩写与原词如果对应，应判断为匹配

请给出判断和置信度：
格式：
判断：[是/否]
置信度：[0-1之间的数字]
理由：[详细说明]"""
    
    data = {
        "model": config['openai']['model'],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": config['openai']['temperature'],
        "max_tokens": config['openai']['max_tokens']
    }
    
    print(f"完整请求数据大小: {len(json.dumps(data))} 字符")
    
    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=120)
        print(f"状态码: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ 增强LLM风格请求成功")
            result = response.json()
            if "choices" in result:
                content = result["choices"][0]["message"]["content"]
                print(f"响应内容: {content[:200]}...")
            return True
        else:
            print(f"❌ 增强LLM风格请求失败")
            print(f"响应内容: {response.text}")
            
            # 详细分析错误
            try:
                error_data = response.json()
                if "error" in error_data:
                    print(f"错误详情: {error_data['error']}")
            except:
                pass
            
            return False
    except Exception as e:
        print(f"❌ 增强LLM风格请求异常: {e}")
        return False

def main():
    """主函数"""
    print("API调用诊断开始...")
    
    try:
        config = load_config()
        print(f"API Base URL: {config['openai']['api_base_url']}")
        print(f"Model: {config['openai']['model']}")
        print(f"API Key: {config['openai']['api_key'][:10]}...{config['openai']['api_key'][-5:]}")
    except Exception as e:
        print(f"配置加载失败: {e}")
        return
    
    # 运行所有测试
    tests = [
        ("简单请求", test_simple_request),
        ("匹配风格请求", test_matching_style_request),
        ("带User-Agent请求", test_with_user_agent),
        ("快速连续请求", test_rapid_requests),
        ("增强LLM风格请求", test_enhanced_llm_style)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func(config)
        except Exception as e:
            print(f"测试 {test_name} 出现异常: {e}")
            results[test_name] = False
        
        time.sleep(1)  # 测试间隔
    
    # 总结
    print("\n" + "="*50)
    print("测试结果总结")
    print("="*50)
    
    for test_name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    # 分析问题
    print("\n问题分析:")
    if results.get("简单请求", False) and not results.get("增强LLM风格请求", False):
        print("🔍 简单请求成功但复杂请求失败，可能原因:")
        print("  - 请求内容过长")
        print("  - 请求格式问题")
        print("  - API对复杂请求有特殊限制")
    
    if not results.get("快速连续请求", False):
        print("🔍 快速请求失败，可能原因:")
        print("  - API有频率限制")
        print("  - 需要增加请求间隔")
    
    print("\n建议解决方案:")
    print("1. 如果复杂请求失败，尝试简化提示内容")
    print("2. 如果快速请求失败，增加请求间隔时间")
    print("3. 检查API服务商的使用限制和文档")
    print("4. 考虑使用不同的请求头或参数")

if __name__ == "__main__":
    main()