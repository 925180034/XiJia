"""
调试enhanced_matching.py脚本的问题
"""
import os
import sys
import yaml
import traceback

def debug_config_loading():
    """调试配置文件加载"""
    print("=== 1. 调试配置文件加载 ===")
    
    config_path = "config/config_enhanced.yaml"
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"❌ 配置文件不存在: {config_path}")
        return False
    else:
        print(f"✅ 配置文件存在: {config_path}")
    
    # 尝试加载配置文件
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print("✅ 配置文件加载成功")
        
        # 检查必要的配置键
        required_keys = [
            ["openai", "api_key"],
            ["openai", "api_base_url"],
            ["openai", "model"],
            ["similarity", "char_weight"],
            ["similarity", "semantic_weight"],
            ["similarity", "struct_weight"],
            ["similarity", "pinyin_boost"],
            ["thresholds", "similarity_threshold"],
            ["matching_strategy", "enable_aggressive_mode"],
            ["system", "batch_size"]
        ]
        
        for key_path in required_keys:
            try:
                current = config
                for key in key_path:
                    current = current[key]
                print(f"✅ 配置键存在: {'.'.join(key_path)} = {current}")
            except KeyError:
                print(f"❌ 配置键缺失: {'.'.join(key_path)}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件加载失败: {e}")
        traceback.print_exc()
        return False

def debug_imports():
    """调试模块导入"""
    print("\n=== 2. 调试模块导入 ===")
    
    imports_to_test = [
        ("src.data.data_loader", "DataLoader"),
        ("src.data.data_preprocessor", "MetadataPreprocessor"),
        ("src.features.enhanced_similarity_calculator", "EnhancedSimilarityCalculator"),
        ("src.matching.candidate_filter", "CandidateFilter"),
        ("src.matching.enhanced_llm_matcher", "EnhancedLLMMatcher"),
        ("src.matching.result_processor", "ResultProcessor")
    ]
    
    success_count = 0
    
    for module_name, class_name in imports_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✅ 导入成功: {module_name}.{class_name}")
            success_count += 1
        except Exception as e:
            print(f"❌ 导入失败: {module_name}.{class_name} - {e}")
    
    print(f"导入结果: {success_count}/{len(imports_to_test)} 成功")
    return success_count == len(imports_to_test)

def debug_enhanced_similarity_calculator():
    """调试增强相似度计算器"""
    print("\n=== 3. 调试增强相似度计算器 ===")
    
    try:
        from src.features.enhanced_similarity_calculator import EnhancedSimilarityCalculator
        
        # 尝试创建实例
        calculator = EnhancedSimilarityCalculator(
            char_weight=0.3,
            semantic_weight=0.6,
            struct_weight=0.1,
            pinyin_boost=1.5
        )
        print("✅ EnhancedSimilarityCalculator 创建成功")
        
        # 测试基本方法
        test_source_field = {
            "name": "id",
            "desc": "标识符",
            "type": "VARCHAR"
        }
        
        test_target_field = {
            "name": "ID",
            "desc": "主键",
            "type": "VARCHAR"
        }
        
        similarity = calculator.calculate_similarity(test_source_field, test_target_field)
        print(f"✅ 相似度计算成功: {similarity:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ EnhancedSimilarityCalculator 测试失败: {e}")
        traceback.print_exc()
        return False

def debug_enhanced_llm_matcher():
    """调试增强LLM匹配器"""
    print("\n=== 4. 调试增强LLM匹配器 ===")
    
    try:
        from src.matching.enhanced_llm_matcher import EnhancedLLMMatcher
        
        # 尝试创建实例
        matcher = EnhancedLLMMatcher(config_path="config/config_enhanced.yaml")
        print("✅ EnhancedLLMMatcher 创建成功")
        
        return True
        
    except Exception as e:
        print(f"❌ EnhancedLLMMatcher 测试失败: {e}")
        traceback.print_exc()
        return False

def debug_data_files():
    """调试数据文件"""
    print("\n=== 5. 调试数据文件 ===")
    
    data_files = [
        "data/源数据字典.xlsx",
        "data/项目匹配字典.xlsx",
        "data/项目匹配字典_列类型注释.xlsx"
    ]
    
    success_count = 0
    
    for file_path in data_files:
        if os.path.exists(file_path):
            print(f"✅ 数据文件存在: {file_path}")
            success_count += 1
        else:
            print(f"❌ 数据文件不存在: {file_path}")
    
    print(f"数据文件检查: {success_count}/{len(data_files)} 存在")
    return success_count > 0

def main():
    """主调试函数"""
    print("开始调试enhanced_matching.py脚本...")
    
    # 确保当前目录正确
    if not os.path.exists("src"):
        print("❌ 当前目录下没有src文件夹，请确保在项目根目录下运行")
        return
    
    # 添加src到Python路径
    sys.path.insert(0, '.')
    
    tests = [
        ("配置文件加载", debug_config_loading),
        ("模块导入", debug_imports),
        ("增强相似度计算器", debug_enhanced_similarity_calculator),
        ("增强LLM匹配器", debug_enhanced_llm_matcher),
        ("数据文件", debug_data_files)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ 测试 {test_name} 出现异常: {e}")
            results[test_name] = False
    
    # 总结
    print("\n" + "="*50)
    print("调试结果总结")
    print("="*50)
    
    for test_name, success in results.items():
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 所有测试通过！enhanced_matching.py应该可以正常运行。")
    else:
        print("\n⚠️  存在问题，请根据上述错误信息进行修复。")
        print("\n常见解决方案:")
        print("1. 确保config/config_enhanced.yaml文件存在且格式正确")
        print("2. 检查src/matching/enhanced_llm_matcher.py是否有类定义冲突")
        print("3. 确保所有依赖库已安装（pip install -r requirements.txt）")
        print("4. 确保数据文件存在")

if __name__ == "__main__":
    main()