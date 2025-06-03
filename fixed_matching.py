"""
修复版Schema匹配脚本 - 使用与测试脚本一致的API调用方法
"""
import os
import sys
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
import argparse

from src.data.data_loader import DataLoader, SchemaMetadata
from src.data.data_preprocessor import MetadataPreprocessor
from src.features.similarity_calculator import SimilarityCalculator
from src.matching.candidate_filter import CandidateFilter
from src.matching.fixed_llm_matcher import FixedLLMMatcher
from src.matching.result_processor import ResultProcessor

def main():
    """修复版匹配主函数"""
    parser = argparse.ArgumentParser(description='修复版Schema匹配')
    parser.add_argument('--source', type=str, default='data/源数据字典.xlsx', help='源表数据字典文件路径')
    parser.add_argument('--target', type=str, default='data/项目匹配字典.xlsx', help='目标表数据字典文件路径')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--auto', action='store_true', help='自动模式，不交互')
    parser.add_argument('--max-pairs', type=int, default=20, help='最大表对数量')
    parser.add_argument('--max-llm', type=int, default=50, help='最大LLM处理候选对数量')
    args = parser.parse_args()
    
    print("=== 修复版Schema匹配系统 ===")
    
    # 检查文件
    for file_path, desc in [(args.config, "配置文件"), (args.source, "源数据文件"), (args.target, "目标数据文件")]:
        if not os.path.exists(file_path):
            print(f"错误: {desc}不存在: {file_path}")
            sys.exit(1)
    
    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print(f"使用配置文件: {args.config}")
    print(f"API Base URL: {config['openai'].get('api_base_url', 'Default OpenAI API')}")
    print(f"Model: {config['openai']['model']}")
    
    # 1. 数据加载
    print("\n1. 数据加载...")
    start_time = time.time()
    data_loader = DataLoader()
    
    try:
        source_schemas = data_loader.load_excel_dictionary(args.source)
        target_schemas = data_loader.load_excel_dictionary(args.target)
        
        print(f"加载完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"源表数量: {len(source_schemas)}, 目标表数量: {len(target_schemas)}")
        
        # 计算总字段数
        total_source_fields = sum(len(schema.fields) for schema in source_schemas)
        total_target_fields = sum(len(schema.fields) for schema in target_schemas)
        print(f"总字段数: 源表 {total_source_fields}, 目标表 {total_target_fields}")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
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
    
    # 简化处理：只处理前几个表
    max_tables = args.max_pairs // 5 if args.max_pairs > 5 else 1
    selected_source_schemas = list(processed_source_schemas.values())[:max_tables]
    selected_target_schemas = list(processed_target_schemas.values())[:max_tables]
    
    print(f"为减少计算量，处理前 {len(selected_source_schemas)} 个源表和 {len(selected_target_schemas)} 个目标表")
    
    similarity_matrices = {}
    all_candidates = []
    
    for source_schema in selected_source_schemas:
        for target_schema in selected_target_schemas:
            print(f"计算表 {source_schema['table_name']} 和 {target_schema['table_name']} 之间的相似度...")
            
            # 计算相似度矩阵
            matrix = similarity_calculator.calculate_similarity_matrix(
                source_schema["fields"],
                target_schema["fields"]
            )
            
            similarity_matrices[(source_schema["table_name"], target_schema["table_name"])] = matrix
            
            # 筛选候选对
            for i, s_field in enumerate(source_schema["fields"]):
                for j, t_field in enumerate(target_schema["fields"]):
                    sim = matrix[i, j]
                    if sim >= config["thresholds"]["similarity_threshold"]:
                        all_candidates.append({
                            "source_table": source_schema["table_name"],
                            "source_field": s_field["name"],
                            "target_table": target_schema["table_name"],
                            "target_field": t_field["name"],
                            "similarity": float(sim)
                        })
    
    # 按相似度排序
    all_candidates.sort(key=lambda x: x["similarity"], reverse=True)
    
    print(f"特征计算完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"找到 {len(all_candidates)} 对候选字段匹配")
    
    # 4. 应用匹配规则
    print("\n4. 应用匹配规则...")
    start_time = time.time()
    
    candidate_filter = CandidateFilter(
        similarity_threshold=config["thresholds"]["similarity_threshold"]
    )
    
    filtered_candidates = candidate_filter.apply_matching_rules(all_candidates)
    
    print(f"规则应用完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"应用规则后保留 {len(filtered_candidates)} 对候选匹配")
    
    # 限制处理数量
    max_llm_candidates = min(args.max_llm, len(filtered_candidates))
    if len(filtered_candidates) > max_llm_candidates:
        print(f"候选匹配对数量较多，只处理相似度较高的前 {max_llm_candidates} 个")
        current_candidates = filtered_candidates[:max_llm_candidates]
    else:
        current_candidates = filtered_candidates
    
    # 显示候选匹配示例
    if current_candidates:
        print("\n候选匹配对示例:")
        for i, candidate in enumerate(current_candidates[:5]):
            print(f"{i+1}. {candidate['source_table']}.{candidate['source_field']} <-> "
                  f"{candidate['target_table']}.{candidate['target_field']} "
                  f"(相似度: {candidate['similarity']:.4f})")
    
    # 5. 修复版LLM语义匹配
    print("\n5. 修复版LLM语义匹配...")
    start_time = time.time()
    
    try:
        # 使用修复版LLM匹配器
        llm_matcher = FixedLLMMatcher(config_path=args.config)
        
        print(f"开始处理 {len(current_candidates)} 对候选匹配...")
        
        matching_results = llm_matcher.batch_process_candidates(
            current_candidates,
            processed_source_schemas,
            processed_target_schemas
        )
        
        print(f"修复版LLM匹配完成，耗时: {time.time() - start_time:.2f}秒")
        
    except Exception as e:
        print(f"LLM匹配失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 使用相似度作为回退
        print("使用相似度结果作为替代...")
        matching_results = []
        for candidate in current_candidates:
            matching_results.append({
                "source_table": candidate["source_table"],
                "source_field": candidate["source_field"],
                "target_table": candidate["target_table"],
                "target_field": candidate["target_field"],
                "match": candidate["similarity"] > 0.7,
                "confidence": candidate["similarity"],
                "reason": f"基于相似度 {candidate['similarity']:.4f} (LLM失败回退)",
                "similarity": candidate["similarity"],
                "fallback": True
            })
    
    # 6. 结果处理
    print("\n6. 结果处理...")
    start_time = time.time()
    
    result_processor = ResultProcessor(
        confidence_threshold=config["thresholds"]["confidence_threshold"]
    )
    
    # 计算统计信息
    statistics = result_processor.calculate_matching_statistics(
        matching_results,
        processed_source_schemas,
        processed_target_schemas,
        all_candidates
    )
    
    final_results = result_processor.process_matching_results(
        matching_results,
        processed_source_schemas,
        processed_target_schemas
    )
    
    print(f"结果处理完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 7. 保存结果
    os.makedirs(args.output, exist_ok=True)
    output_files = result_processor.save_results(final_results, statistics, args.output)
    
    # 8. 输出结果
    print("\n" + "="*60)
    print("                  匹配结果总结")
    print("="*60)
    
    print(f"\n【数据规模】")
    print(f"  处理的源表数量: {len(selected_source_schemas)}")
    print(f"  处理的目标表数量: {len(selected_target_schemas)}")
    print(f"  候选匹配对数量: {len(all_candidates)}")
    print(f"  LLM处理的候选对: {len(current_candidates)}")
    
    print(f"\n【匹配结果】")
    successful_matches = [r for r in matching_results if r.get("match", False)]
    fallback_matches = [r for r in matching_results if r.get("fallback", False)]
    
    print(f"  LLM判断为匹配: {len(successful_matches)} 对")
    print(f"  高置信度匹配: {len(final_results)} 对")
    print(f"  使用回退策略: {len(fallback_matches)} 对")
    
    # LLM调用成功率
    if hasattr(llm_matcher, 'success_count'):
        total_calls = llm_matcher.success_count + llm_matcher.fail_count
        if total_calls > 0:
            success_rate = llm_matcher.success_count / total_calls * 100
            print(f"  LLM调用成功率: {success_rate:.1f}%")
    
    print(f"\n【输出文件】")
    print(f"  匹配结果: {output_files['excel']}")
    print(f"  统计信息: {output_files['statistics']}")
    
    # 显示匹配结果示例
    if final_results:
        print(f"\n【匹配结果示例】")
        for i, result in enumerate(final_results[:5]):
            print(f"{i+1}. {result['source_table']}.{result['source_field']} <-> "
                  f"{result['target_table']}.{result['target_field']}")
            print(f"   置信度: {result['confidence']:.2f}")
            print(f"   匹配依据: {result.get('matching_basis', '')[:100]}...")
            print()
        
        if len(final_results) > 5:
            print(f"... 共 {len(final_results)} 条匹配结果，请查看完整输出文件")
    else:
        print(f"\n未找到高置信度的匹配结果")
    
    print("\n修复版匹配实验完成！")

if __name__ == "__main__":
    main()