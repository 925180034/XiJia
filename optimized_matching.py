# """
# 优化版Schema匹配脚本 - 针对大量表的情况
# """
# import os
# import sys
# import yaml
# import pandas as pd
# import numpy as np
# from typing import Dict, List, Tuple, Any, Optional
# import time
# import multiprocessing
# from itertools import product
# import argparse

# from src.data.data_loader import DataLoader, SchemaMetadata
# from src.data.data_preprocessor import MetadataPreprocessor
# from src.features.similarity_calculator import SimilarityCalculator
# from src.matching.candidate_filter import CandidateFilter
# from src.matching.llm_matcher import LLMMatcher
# from src.matching.result_processor import ResultProcessor

# def calculate_table_similarity(source_schema, target_schema):
#     """
#     计算表级别相似度以快速筛选潜在匹配的表
#     """
#     # 表名相似度
#     source_name = source_schema.table_name.lower()
#     target_name = target_schema.table_name.lower()
    
#     import difflib
#     name_sim = difflib.SequenceMatcher(None, source_name, target_name).ratio()
    
#     # 表描述相似度
#     source_desc = source_schema.table_desc.lower() if source_schema.table_desc else ""
#     target_desc = target_schema.table_desc.lower() if target_schema.table_desc else ""
    
#     if source_desc and target_desc:
#         desc_sim = difflib.SequenceMatcher(None, source_desc, target_desc).ratio()
#     else:
#         desc_sim = 0
    
#     # 组合相似度
#     table_sim = 0.7 * name_sim + 0.3 * desc_sim
#     return table_sim

# def filter_table_pairs(source_schemas, target_schemas, table_threshold=0.3, max_pairs=20):
#     """
#     基于表相似度筛选表对，避免所有表两两比较
#     """
#     print(f"开始筛选潜在匹配的表对，总共 {len(source_schemas)} 个源表和 {len(target_schemas)} 个目标表...")
    
#     # 计算所有表对的相似度
#     table_similarities = []
    
#     for source_schema in source_schemas:
#         for target_schema in target_schemas:
#             similarity = calculate_table_similarity(source_schema, target_schema)
#             if similarity >= table_threshold:
#                 table_similarities.append((source_schema, target_schema, similarity))
    
#     # 按相似度排序
#     table_similarities.sort(key=lambda x: x[2], reverse=True)
    
#     # 如果表对太多，只保留相似度最高的一些
#     if len(table_similarities) > max_pairs:
#         print(f"表对数量 ({len(table_similarities)}) 超过限制，只保留相似度最高的 {max_pairs} 对")
#         table_similarities = table_similarities[:max_pairs]
    
#     selected_pairs = [(s, t) for s, t, _ in table_similarities]
    
#     print(f"筛选出 {len(selected_pairs)} 对潜在匹配的表")
#     for i, (source, target, sim) in enumerate(table_similarities):
#         print(f"{i+1}. {source.table_name} <-> {target.table_name} (相似度: {sim:.4f})")
    
#     return selected_pairs

# def process_table_pair(pair_info):
#     """
#     处理单对表，用于并行计算
#     """
#     source_schema, target_schema, preprocessor, similarity_calculator, similarity_threshold = pair_info
    
#     # 预处理
#     processed_source = preprocessor.preprocess_schema(source_schema)
#     processed_target = preprocessor.preprocess_schema(target_schema)
    
#     # 计算相似度矩阵
#     matrix = similarity_calculator.calculate_similarity_matrix(
#         processed_source["fields"],
#         processed_target["fields"]
#     )
    
#     # 筛选候选对
#     candidates = []
#     for i, s_field in enumerate(processed_source["fields"]):
#         for j, t_field in enumerate(processed_target["fields"]):
#             sim = matrix[i, j]
#             if sim >= similarity_threshold:
#                 candidates.append({
#                     "source_table": source_schema.table_name,
#                     "source_field": s_field["name"],
#                     "target_table": target_schema.table_name,
#                     "target_field": t_field["name"],
#                     "similarity": float(sim)
#                 })
    
#     # 排序
#     candidates.sort(key=lambda x: x["similarity"], reverse=True)
    
#     return {
#         "source_schema": processed_source,
#         "target_schema": processed_target,
#         "matrix": matrix,
#         "candidates": candidates
#     }

# def interactive_table_selection(source_schemas, target_schemas):
#     """
#     交互式选择要匹配的表
#     """
#     print("\n=== 交互式表选择 ===")
    
#     # 显示源表列表
#     print("\n源表列表:")
#     for i, schema in enumerate(source_schemas):
#         print(f"{i+1}. {schema.table_name} - {schema.table_desc}")
    
#     # 显示目标表列表
#     print("\n目标表列表:")
#     for i, schema in enumerate(target_schemas):
#         print(f"{i+1}. {schema.table_name} - {schema.table_desc}")
    
#     print("\n选择匹配方式:")
#     print("1. 自动选择潜在匹配的表 (基于相似度)")
#     print("2. 手动选择特定表")
#     print("3. 处理所有表")
    
#     choice = input("请选择 (1/2/3): ")
    
#     if choice == "1":
#         table_threshold = float(input("请输入表相似度阈值 (0-1, 建议0.3): ") or "0.3")
#         max_pairs = int(input("请输入最大表对数量 (建议20): ") or "20")
#         return filter_table_pairs(source_schemas, target_schemas, table_threshold, max_pairs), False
    
#     elif choice == "2":
#         selected_pairs = []
        
#         # 选择源表
#         source_indices = input("请输入源表序号 (多个用逗号分隔，如1,3,5): ")
#         source_indices = [int(idx.strip()) - 1 for idx in source_indices.split(",") if idx.strip().isdigit()]
#         selected_sources = [source_schemas[idx] for idx in source_indices if 0 <= idx < len(source_schemas)]
        
#         # 选择目标表
#         target_indices = input("请输入目标表序号 (多个用逗号分隔，如1,3,5): ")
#         target_indices = [int(idx.strip()) - 1 for idx in target_indices.split(",") if idx.strip().isdigit()]
#         selected_targets = [target_schemas[idx] for idx in target_indices if 0 <= idx < len(target_schemas)]
        
#         # 生成表对
#         for source, target in product(selected_sources, selected_targets):
#             selected_pairs.append((source, target))
        
#         return selected_pairs, False
    
#     else:
#         return [(s, t) for s, t in product(source_schemas, target_schemas)], True

# def main():
#     """优化版匹配主函数"""
#     parser = argparse.ArgumentParser(description='优化版Schema匹配')
#     parser.add_argument('--source', type=str, default='data/源数据字典.xlsx', help='源表数据字典文件路径')
#     parser.add_argument('--target', type=str, default='data/项目匹配字典.xlsx', help='目标表数据字典文件路径')
#     parser.add_argument('--config', type=str, default='config/config.yaml', help='配置文件路径')
#     parser.add_argument('--output', type=str, default='output', help='输出目录')
#     parser.add_argument('--auto', action='store_true', help='自动模式，不交互')
#     args = parser.parse_args()
    
#     print("=== 优化版Schema匹配系统 ===")
    
#     # 确保文件存在
#     if not os.path.exists(args.config):
#         print(f"错误: 配置文件不存在: {args.config}")
#         sys.exit(1)
    
#     if not os.path.exists(args.source):
#         print(f"错误: 源数据字典文件不存在: {args.source}")
#         sys.exit(1)
    
#     if not os.path.exists(args.target):
#         print(f"错误: 项目匹配字典文件不存在: {args.target}")
#         sys.exit(1)
    
#     # 加载配置
#     with open(args.config, "r", encoding="utf-8") as f:
#         config = yaml.safe_load(f)
    
#     print(f"成功加载配置文件: {args.config}")
    
#     # 1. 数据加载
#     print("\n1. 数据加载...")
#     start_time = time.time()
#     data_loader = DataLoader()
    
#     try:
#         source_schemas = data_loader.load_excel_dictionary(args.source)
#         target_schemas = data_loader.load_excel_dictionary(args.target)
        
#         print(f"加载完成，耗时: {time.time() - start_time:.2f}秒")
#         print(f"源表数量: {len(source_schemas)}, 目标表数量: {len(target_schemas)}")
#     except Exception as e:
#         print(f"数据加载失败: {e}")
#         return
    
#     # 初始化预处理器和相似度计算器
#     preprocessor = MetadataPreprocessor(
#         enable_pinyin=config["chinese"]["enable_pinyin"],
#         enable_abbreviation=config["chinese"]["enable_abbreviation"]
#     )
    
#     similarity_calculator = SimilarityCalculator(
#         char_weight=config["similarity"]["char_weight"],
#         semantic_weight=config["similarity"]["semantic_weight"],
#         struct_weight=config["similarity"]["struct_weight"]
#     )
    
#     # 2. 表对筛选
#     if args.auto:
#         # 自动模式，使用默认参数筛选表
#         table_pairs, process_all = filter_table_pairs(source_schemas, target_schemas), False
#     else:
#         # 交互式选择表
#         table_pairs, process_all = interactive_table_selection(source_schemas, target_schemas)
    
#     # 如果要处理所有表
#     if process_all:
#         print(f"警告: 将处理所有 {len(source_schemas) * len(target_schemas)} 对表，可能需要大量时间")
#         confirmation = input("是否继续? (y/n): ")
#         if confirmation.lower() != 'y':
#             print("已取消操作")
#             return
    
#     # 3. 并行处理表对
#     print(f"\n3. 开始处理 {len(table_pairs)} 对表...")
#     start_time = time.time()
    
#     # 准备并行处理参数
#     pair_info_list = [
#         (source, target, preprocessor, similarity_calculator, config["thresholds"]["similarity_threshold"])
#         for source, target in table_pairs
#     ]
    
#     # 使用进程池并行处理表对
#     with multiprocessing.Pool(processes=min(os.cpu_count(), len(pair_info_list))) as pool:
#         results = pool.map(process_table_pair, pair_info_list)
    
#     print(f"表对处理完成，耗时: {time.time() - start_time:.2f}秒")
    
#     # 4. 整合结果
#     processed_source_schemas = {}
#     processed_target_schemas = {}
#     similarity_matrices = {}
#     all_candidates = []
    
#     for result in results:
#         source_schema = result["source_schema"]
#         target_schema = result["target_schema"]
        
#         processed_source_schemas[source_schema["table_name"]] = source_schema
#         processed_target_schemas[target_schema["table_name"]] = target_schema
        
#         matrix_key = (source_schema["table_name"], target_schema["table_name"])
#         similarity_matrices[matrix_key] = result["matrix"]
        
#         all_candidates.extend(result["candidates"])
    
#     # 按相似度排序所有候选对
#     all_candidates.sort(key=lambda x: x["similarity"], reverse=True)
    
#     print(f"找到 {len(all_candidates)} 对候选字段匹配")
    
#     # 5. 筛选并应用规则
#     print("\n4. 应用匹配规则...")
#     start_time = time.time()
    
#     candidate_filter = CandidateFilter(
#         similarity_threshold=config["thresholds"]["similarity_threshold"]
#     )
    
#     # 应用规则进一步筛选
#     filtered_candidates = candidate_filter.apply_matching_rules(all_candidates)
    
#     print(f"规则应用完成，耗时: {time.time() - start_time:.2f}秒")
#     print(f"应用规则后保留 {len(filtered_candidates)} 对候选匹配")
    
#     # 显示部分候选匹配
#     if filtered_candidates:
#         print("\n候选匹配对示例:")
#         for i, candidate in enumerate(filtered_candidates[:10]):  # 显示前10个
#             print(f"{i+1}. {candidate['source_table']}.{candidate['source_field']} <-> "
#                   f"{candidate['target_table']}.{candidate['target_field']} "
#                   f"(相似度: {candidate['similarity']:.4f})")
    
#     # 6. LLM语义匹配
#     print("\n5. LLM语义匹配...")
#     start_time = time.time()
    
#     # 如果候选对太多，只处理前N个
#     max_llm_candidates = 50
#     if len(filtered_candidates) > max_llm_candidates:
#         print(f"候选匹配对数量较多({len(filtered_candidates)}个)，只处理相似度较高的前{max_llm_candidates}个")
#         current_candidates = filtered_candidates[:max_llm_candidates]
#     else:
#         current_candidates = filtered_candidates
    
#     try:
#         llm_matcher = LLMMatcher(config_path=args.config)
#         matching_results = llm_matcher.batch_process_candidates(
#             current_candidates,
#             processed_source_schemas,
#             processed_target_schemas
#         )
        
#         print(f"LLM匹配完成，耗时: {time.time() - start_time:.2f}秒")
#         print(f"处理了 {len(current_candidates)} 对候选匹配")
#     except Exception as e:
#         print(f"LLM匹配失败: {e}")
#         import traceback
#         traceback.print_exc()
        
#         # 回退方案：使用相似度作为匹配结果
#         print("使用相似度结果作为替代...")
#         matching_results = []
#         for candidate in current_candidates:
#             matching_results.append({
#                 "source_table": candidate["source_table"],
#                 "source_field": candidate["source_field"],
#                 "target_table": candidate["target_table"],
#                 "target_field": candidate["target_field"],
#                 "match": candidate["similarity"] > 0.8,
#                 "confidence": candidate["similarity"],
#                 "reason": f"基于相似度 {candidate['similarity']:.4f}",
#                 "similarity": candidate["similarity"]
#             })
    
#     # 7. 结果处理
#     print("\n6. 结果处理...")
#     start_time = time.time()
    
#     result_processor = ResultProcessor(
#         confidence_threshold=config["thresholds"]["confidence_threshold"]
#     )
    
#     final_results = result_processor.process_matching_results(
#         matching_results,
#         processed_source_schemas,
#         processed_target_schemas
#     )
    
#     # 保存结果
#     os.makedirs(args.output, exist_ok=True)
#     output_files = result_processor.save_results(final_results, args.output)
    
#     print(f"结果处理完成，耗时: {time.time() - start_time:.2f}秒")
#     print(f"找到 {len(final_results)} 对匹配字段")
#     print(f"结果已保存至: {output_files['excel']}")
    
#     # 输出匹配结果示例
#     print("\n=== 匹配结果示例 ===")
#     for i, result in enumerate(final_results[:10]):  # 最多显示10个结果
#         print(f"{i+1}. {result['source_table']}.{result['source_field']} <-> "
#               f"{result['target_table']}.{result['target_field']}")
#         print(f"   匹配置信度: {result['confidence']:.2f}")
#         print(f"   匹配依据: {result.get('matching_basis', '')}")
#         print()
    
#     if len(final_results) > 10:
#         print(f"... 共 {len(final_results)} 条匹配结果，请查看完整输出文件")
    
#     print("匹配实验完成！")

# if __name__ == "__main__":
#     main()
