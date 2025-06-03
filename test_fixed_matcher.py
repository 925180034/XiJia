"""
测试修复版LLM匹配器
"""
import sys
import os

# 添加src目录到Python路径
sys.path.append('src')

from src.matching.fixed_llm_matcher import FixedLLMMatcher

def test_fixed_matcher():
    """测试修复版匹配器"""
    
    print("=== 测试修复版LLM匹配器 ===")
    
    # 初始化匹配器
    try:
        matcher = FixedLLMMatcher("config/config.yaml")
        print("✅ 匹配器初始化成功")
    except Exception as e:
        print(f"❌ 匹配器初始化失败: {e}")
        return False
    
    # 创建测试数据
    source_schema = {
        "table_name": "TEST_SOURCE_TABLE",
        "table_desc": "测试源表"
    }
    
    source_field = {
        "name": "id",
        "desc": "唯一标识符",
        "type": "VARCHAR"
    }
    
    target_schema = {
        "table_name": "TEST_TARGET_TABLE", 
        "table_desc": "测试目标表"
    }
    
    target_field = {
        "name": "ID",
        "desc": "主键",
        "type": "VARCHAR"
    }
    
    # 测试单对匹配
    print("\n测试单对字段匹配...")
    try:
        result = matcher.match_field_pair(
            source_schema, source_field, 
            target_schema, target_field, 
            1.0
        )
        
        print(f"✅ 匹配成功")
        print(f"  匹配结果: {result['match']}")
        print(f"  置信度: {result['confidence']}")
        print(f"  理由: {result['reason'][:100]}...")
        
        if "fallback" in result:
            print(f"  ⚠️  使用了回退策略")
        
        return True
        
    except Exception as e:
        print(f"❌ 匹配失败: {e}")
        return False

def test_multiple_calls():
    """测试多次调用"""
    print("\n=== 测试多次连续调用 ===")
    
    try:
        matcher = FixedLLMMatcher("config/config.yaml")
        
        test_pairs = [
            ({"name": "id", "desc": "标识"}, {"name": "ID", "desc": "主键"}),
            ({"name": "name", "desc": "姓名"}, {"name": "NAME", "desc": "名称"}),
            ({"name": "create_time", "desc": "创建时间"}, {"name": "CREATE_TIME", "desc": "创建时间"})
        ]
        
        source_schema = {"table_name": "SOURCE", "table_desc": "源表"}
        target_schema = {"table_name": "TARGET", "table_desc": "目标表"}
        
        success_count = 0
        
        for i, (s_field, t_field) in enumerate(test_pairs):
            print(f"\n测试第 {i+1} 对字段: {s_field['name']} <-> {t_field['name']}")
            
            try:
                result = matcher.match_field_pair(
                    source_schema, s_field,
                    target_schema, t_field,
                    0.9
                )
                
                if "fallback" not in result:
                    success_count += 1
                    print(f"✅ API调用成功，匹配: {result['match']}, 置信度: {result['confidence']}")
                else:
                    print(f"⚠️  使用回退策略: {result['reason'][:100]}")
                
            except Exception as e:
                print(f"❌ 调用失败: {e}")
        
        print(f"\n多次调用结果: {success_count}/{len(test_pairs)} 次API调用成功")
        return success_count > 0
        
    except Exception as e:
        print(f"❌ 多次调用测试失败: {e}")
        return False

if __name__ == "__main__":
    # 测试修复版匹配器
    test1_success = test_fixed_matcher()
    test2_success = test_multiple_calls()
    
    print("\n" + "="*50)
    print("测试结果总结:")
    print(f"单次匹配测试: {'✅ 成功' if test1_success else '❌ 失败'}")
    print(f"多次调用测试: {'✅ 成功' if test2_success else '❌ 失败'}")
    
    if test1_success and test2_success:
        print("\n🎉 修复版匹配器测试通过！可以在主脚本中使用。")
    else:
        print("\n⚠️  修复版匹配器仍有问题，需要进一步调试。")