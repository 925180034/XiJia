"""
Schema匹配系统主程序
"""
import os
import time
import yaml
import numpy as np
from typing import Dict, List, Tuple, Any
import argparse

from src.data.data_loader import DataLoader
from src.data.data_preprocessor import MetadataPreprocessor
from src.features.similarity_calculator import SimilarityCalculator
from src.matching.candidate_filter import CandidateFilter
from src.matching.llm_matcher import LLMMatcher
from src.matching.result_processor import ResultProcessor


def main():
    """主程序入口"""
    parser = argparse.ArgumentParser(description='基于元数据的Schema匹配系统')
    parser.add_argument('--source', type=str, default='data/源数据字典.xlsx', help='源表数据字典文件路径')
    parser.add_argument('--target', type=str, default='data/项目匹配字典.xlsx', help='目标表数据字典文件路径')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print("=== 基于元数据的Schema匹配系统 ===")
    print(f"正在处理源表文件: {args.source}")
    print(f"正在处理目标表文件: {args.target}")
    
    # 1. 数据加载
    print("\n1. 数据加载...")
    start_time = time.time()
    data_loader = DataLoader()
    source_schemas, target_schemas = data_loader.load_schemas(args.source, args.target)
    print(f"加载完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"源表数量: {len(source_schemas)}, 目标表数量: {len(target_schemas)}")
    
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
            # 计算相似度矩阵
            matrix = similarity_calculator.calculate_similarity_matrix(
                source_schema["fields"],
                target_schema["fields"]
            )
            
            similarity_matrices[(source_name, target_name)] = matrix
            total_pairs += len(source_schema["fields"]) * len(target_schema["fields"])
    
    print(f"特征计算完成，计算了{total_pairs}对字段相似度，耗时: {time.time() - start_time:.2f}秒")
    
    # 4. 候选对筛选
    print("\n4. 候选对筛选...")
    start_time = time.time()
    
    candidate_filter = CandidateFilter(
        similarity_threshold=config["thresholds"]["similarity_threshold"]
    )
    
    candidates = candidate_filter.filter_candidates(
        list(processed_source_schemas.values()),
        list(processed_target_schemas.values()),
        similarity_matrices
    )
    
    # 应用规则进一步筛选
    filtered_candidates = candidate_filter.apply_matching_rules(candidates)
    
    print(f"筛选完成，从{total_pairs}对字段中筛选出{len(candidates)}对候选匹配，" 
          f"应用规则后保留{len(filtered_candidates)}对，耗时: {time.time() - start_time:.2f}秒")
    
    # 5. LLM语义匹配
    print("\n5. LLM语义匹配...")
    start_time = time.time()
    
    llm_matcher = LLMMatcher(config_path=args.config)
    
    matching_results = llm_matcher.batch_process_candidates(
        filtered_candidates,
        processed_source_schemas,
        processed_target_schemas
    )
    
    print(f"LLM匹配完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 6. 结果处理
    print("\n6. 结果处理...")
    start_time = time.time()
    
    result_processor = ResultProcessor(
        confidence_threshold=config["thresholds"]["confidence_threshold"]
    )
    
    final_results = result_processor.process_matching_results(
        matching_results,
        processed_source_schemas,
        processed_target_schemas
    )
    
    # 保存结果
    output_files = result_processor.save_results(final_results, args.output)
    
    print(f"结果处理完成，找到{len(final_results)}对匹配字段，耗时: {time.time() - start_time:.2f}秒")
    print(f"结果已保存至: {output_files['excel']}")
    
    # 输出一些示例结果
    print("\n=== 部分匹配结果示例 ===")
    for i, result in enumerate(final_results[:5]):
        print(f"{i+1}. 源表: {result['source_table']}, 源字段: {result['source_field']}")
        print(f"   目标表: {result['target_table']}, 目标字段: {result['target_field']}")
        print(f"   匹配置信度: {result['confidence']:.2f}")
        print(f"   匹配依据: {result.get('matching_basis', '')}")
        print()
    
    if len(final_results) > 5:
        print(f"... 共{len(final_results)}条匹配结果，请查看完整输出文件")


if __name__ == "__main__":
    main()