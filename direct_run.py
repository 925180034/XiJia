"""
直接使用项目知识库中的原始表格进行Schema匹配实验
"""
import os
import sys
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import time

from src.data.data_loader import DataLoader, SchemaMetadata
from src.data.data_preprocessor import MetadataPreprocessor
from src.features.similarity_calculator import SimilarityCalculator
from src.matching.candidate_filter import CandidateFilter
from src.matching.llm_matcher import LLMMatcher
from src.matching.result_processor import ResultProcessor

def load_schemas_from_files(source_path, target_path):
    """
    直接从Excel文件加载表结构
    
    Args:
        source_path: 源表Excel文件路径
        target_path: 目标表Excel文件路径
    
    Returns:
        源表元数据列表和目标表元数据列表
    """
    data_loader = DataLoader()
    
    print(f"加载源表数据: {source_path}")
    source_schemas = data_loader.load_excel_dictionary(source_path)
    
    print(f"加载目标表数据: {target_path}")
    target_schemas = data_loader.load_excel_dictionary(target_path)
    
    return source_schemas, target_schemas

def main():
    """运行匹配实验"""
    print("=== 基于元数据的Schema匹配实验（直接使用项目知识库表格）===")
    
    # 配置路径
    config_path = "config/config.yaml"
    source_file = "data/源数据字典.xlsx"
    target_file = "data/项目匹配字典.xlsx"
    output_dir = "output"
    
    # 确保文件存在
    if not os.path.exists(config_path):
        print(f"错误: 配置文件不存在: {config_path}")
        sys.exit(1)
    
    if not os.path.exists(source_file):
        print(f"警告: 源数据字典文件不存在: {source_file}")
        print("请确保源数据字典文件放在正确的位置。")
        sys.exit(1)
    
    if not os.path.exists(target_file):
        print(f"警告: 项目匹配字典文件不存在: {target_file}")
        print("请确保项目匹配字典文件放在正确的位置。")
        sys.exit(1)
    
    # 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print(f"成功加载配置文件: {config_path}")
    print(f"使用模型: {config['openai']['model']}")
    print(f"API Base URL: {config['openai'].get('api_base_url', 'Default OpenAI API')}")
    
    # 1. 数据加载
    print("\n1. 数据加载...")
    start_time = time.time()
    try:
        source_schemas, target_schemas = load_schemas_from_files(source_file, target_file)
        print(f"加载完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"源表数量: {len(source_schemas)}, 目标表数量: {len(target_schemas)}")
        
        # 打印部分源表和目标表信息做验证
        if source_schemas:
            print(f"\n源表示例: {source_schemas[0].table_name}")
            print(f"源表字段数: {len(source_schemas[0].fields)}")
            if source_schemas[0].fields:
                print(f"前3个字段: {', '.join([f['name'] for f in source_schemas[0].fields[:3]])}")
        
        if target_schemas:
            print(f"\n目标表示例: {target_schemas[0].table_name}")
            print(f"目标表字段数: {len(target_schemas[0].fields)}")
            if target_schemas[0].fields:
                print(f"前3个字段: {', '.join([f['name'] for f in target_schemas[0].fields[:3]])}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        sys.exit(1)
    
    # 2. 元数据预处理
    print("\n2. 元数据预处理...")
    start_time = time.time()
    preprocessor = MetadataPreprocessor(
        enable_pinyin=config["chinese"]["enable_pinyin"],
        enable_abbreviation=config["chinese"]["enable_abbreviation"]
    )
    
    processed_source_schemas = {}
    for schema in source_schemas:
        processed = preprocessor.preprocess_schema(schema)
        processed_source_schemas[schema.table_name] = processed
    
    processed_target_schemas = {}
    for schema in target_schemas:
        processed = preprocessor.preprocess_schema(schema)
        processed_target_schemas[schema.table_name] = processed
    
    print(f"预处理完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"处理了 {len(processed_source_schemas)} 个源表和 {len(processed_target_schemas)} 个目标表")
    
    # 3. 特征计算与相似度计算
    print("\n3. 特征计算与相似度计算...")
    start_time = time.time()
    
    similarity_calculator = SimilarityCalculator(
        char_weight=config["similarity"]["char_weight"],
        semantic_weight=config["similarity"]["semantic_weight"],
        struct_weight=config["similarity"]["struct_weight"]
    )
    
    similarity_matrices = {}
    total_pairs = 0
    
    for source_name, source_schema in processed_source_schemas.items():
        for target_name, target_schema in processed_target_schemas.items():
            print(f"计算表 {source_name} 和 {target_name} 之间的相似度...")
            
            # 计算相似度矩阵
            matrix = similarity_calculator.calculate_similarity_matrix(
                source_schema["fields"],
                target_schema["fields"]
            )
            
            similarity_matrices[(source_name, target_name)] = matrix
            pair_count = len(source_schema["fields"]) * len(target_schema["fields"])
            total_pairs += pair_count
            print(f"  - 计算了 {pair_count} 对字段相似度")
    
    print(f"特征计算完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"共计算了 {total_pairs} 对字段相似度")
    
    # 输出高相似度的字段对，帮助了解数据
    print("\n高相似度字段对:")
    for (source_name, target_name), matrix in similarity_matrices.items():
        source_schema = processed_source_schemas[source_name]
        target_schema = processed_target_schemas[target_name]
        
        # 找出相似度大于0.5的字段对
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                sim = matrix[i, j]
                if sim > 0.5:
                    source_field = source_schema["fields"][i]
                    target_field = target_schema["fields"][j]
                    print(f"  - {source_name}.{source_field['name']} <-> {target_name}.{target_field['name']}: {sim:.4f}")
    
    # 4. 候选对筛选
    print("\n4. 候选对筛选...")
    start_time = time.time()
    
    # 降低相似度阈值，找到更多候选匹配对
    similarity_threshold = 0.3
    print(f"使用相似度阈值: {similarity_threshold}")
    
    candidate_filter = CandidateFilter(
        similarity_threshold=similarity_threshold
    )
    
    candidates = candidate_filter.filter_candidates(
        list(processed_source_schemas.values()),
        list(processed_target_schemas.values()),
        similarity_matrices
    )
    
    # 应用规则进一步筛选
    filtered_candidates = candidate_filter.apply_matching_rules(candidates)
    
    print(f"筛选完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"从 {total_pairs} 对字段中筛选出 {len(candidates)} 对候选匹配，" 
          f"应用规则后保留 {len(filtered_candidates)} 对")
    
    if filtered_candidates:
        print("\n候选匹配对示例:")
        for i, candidate in enumerate(filtered_candidates[:10]):  # 显示前10个
            print(f"{i+1}. 源表: {candidate['source_table']}, 源字段: {candidate['source_field']}")
            print(f"   目标表: {candidate['target_table']}, 目标字段: {candidate['target_field']}")
            print(f"   相似度: {candidate['similarity']:.4f}")
    
    # 5. LLM语义匹配
    print("\n5. LLM语义匹配...")
    start_time = time.time()
    try:
        llm_matcher = LLMMatcher(config_path=config_path)
        
        # 如果候选对太多，只处理相似度较高的前20个
        if len(filtered_candidates) > 20:
            print(f"候选匹配对数量较多({len(filtered_candidates)}个)，只处理相似度较高的前20个")
            filtered_candidates = filtered_candidates[:20]
        
        matching_results = llm_matcher.batch_process_candidates(
            filtered_candidates,
            processed_source_schemas,
            processed_target_schemas
        )
        
        print(f"LLM匹配完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"处理了 {len(filtered_candidates)} 对候选匹配")
        
        # 显示LLM匹配结果
        print("\nLLM匹配结果:")
        for i, result in enumerate(matching_results):
            match_status = "匹配" if result["match"] else "不匹配"
            print(f"{i+1}. {result['source_table']}.{result['source_field']} <-> {result['target_table']}.{result['target_field']}: {match_status} (置信度: {result.get('confidence', 0):.2f})")
    
    except Exception as e:
        print(f"LLM匹配失败: {e}")
        import traceback
        traceback.print_exc()
        # 如果LLM匹配失败，使用相似度结果作为替代
        print("使用相似度结果作为替代...")
        matching_results = []
        for candidate in filtered_candidates:
            matching_results.append({
                "source_table": candidate["source_table"],
                "source_field": candidate["source_field"],
                "target_table": candidate["target_table"],
                "target_field": candidate["target_field"],
                "match": candidate["similarity"] > 0.8,  # 使用较高的相似度阈值
                "confidence": candidate["similarity"],
                "reason": f"基于相似度 {candidate['similarity']:.4f}",
                "similarity": candidate["similarity"]
            })
    
    # 6. 结果处理
    print("\n6. 结果处理...")
    start_time = time.time()
    
    # 降低置信度阈值，获取更多匹配结果
    confidence_threshold = 0.3
    print(f"使用置信度阈值: {confidence_threshold}")
    
    result_processor = ResultProcessor(
        confidence_threshold=confidence_threshold
    )
    
    final_results = result_processor.process_matching_results(
        matching_results,
        processed_source_schemas,
        processed_target_schemas
    )
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_files = result_processor.save_results(final_results, output_dir)
    
    print(f"结果处理完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"找到 {len(final_results)} 对匹配字段")
    print(f"结果已保存至: {output_files['excel']}")
    
    # 输出匹配结果示例
    print("\n=== 匹配结果示例 ===")
    for i, result in enumerate(final_results):
        print(f"{i+1}. 源表: {result['source_table']}, 源字段: {result['source_field']}")
        print(f"   目标表: {result['target_table']}, 目标字段: {result['target_field']}")
        print(f"   匹配置信度: {result['confidence']:.2f}")
        print(f"   匹配依据: {result.get('matching_basis', '')}")
        print()
    
    print("匹配实验完成！")

if __name__ == "__main__":
    main()