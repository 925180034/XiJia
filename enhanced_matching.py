# """
# 增强版Schema匹配脚本 - 提升匹配召回率同时保持准确性
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
# from src.features.enhanced_similarity_calculator import EnhancedSimilarityCalculator
# from src.matching.candidate_filter import CandidateFilter
# from src.matching.enhanced_llm_matcher import EnhancedLLMMatcher
# from src.matching.result_processor import ResultProcessor

# def calculate_table_similarity(source_schema, target_schema):
#     """计算表级别相似度"""
#     source_name = source_schema.table_name.lower()
#     target_name = target_schema.table_name.lower()
    
#     import difflib
#     name_sim = difflib.SequenceMatcher(None, source_name, target_name).ratio()
    
#     source_desc = source_schema.table_desc.lower() if source_schema.table_desc else ""
#     target_desc = target_schema.table_desc.lower() if target_schema.table_desc else ""
    
#     if source_desc and target_desc:
#         desc_sim = difflib.SequenceMatcher(None, source_desc, target_desc).ratio()
#     else:
#         desc_sim = 0
    
#     table_sim = 0.7 * name_sim + 0.3 * desc_sim
#     return table_sim

# def filter_table_pairs(source_schemas, target_schemas, table_threshold=0.2, max_pairs=50):
#     """基于表相似度筛选表对（降低阈值）"""
#     print(f"开始筛选潜在匹配的表对，总共 {len(source_schemas)} 个源表和 {len(target_schemas)} 个目标表...")
    
#     table_similarities = []
    
#     for source_schema in source_schemas:
#         for target_schema in target_schemas:
#             similarity = calculate_table_similarity(source_schema, target_schema)
#             if similarity >= table_threshold:
#                 table_similarities.append((source_schema, target_schema, similarity))
    
#     # 按相似度排序
#     table_similarities.sort(key=lambda x: x[2], reverse=True)
    
#     if len(table_similarities) > max_pairs:
#         print(f"表对数量 ({len(table_similarities)}) 超过限制，只保留相似度最高的 {max_pairs} 对")
#         table_similarities = table_similarities[:max_pairs]
    
#     selected_pairs = [(s, t) for s, t, _ in table_similarities]
    
#     print(f"筛选出 {len(selected_pairs)} 对潜在匹配的表")
#     for i, (source, target, sim) in enumerate(table_similarities):
#         print(f"{i+1}. {source.table_name} <-> {target.table_name} (相似度: {sim:.4f})")
    
#     return selected_pairs

# def process_table_pair_enhanced(pair_info):
#     """处理单对表（增强版）"""
#     source_schema, target_schema, preprocessor, similarity_calculator, similarity_threshold = pair_info
    
#     # 预处理
#     processed_source = preprocessor.preprocess_schema(source_schema)
#     processed_target = preprocessor.preprocess_schema(target_schema)
    
#     # 使用增强版相似度计算
#     matrix = similarity_calculator.calculate_similarity_matrix(
#         processed_source["fields"],
#         processed_target["fields"]
#     )
    
#     # 筛选候选对（使用更低的阈值）
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

# def classify_matches_by_confidence(results: List[Dict], config: Dict) -> Dict[str, List[Dict]]:
#     """根据置信度分类匹配结果"""
#     high_conf_threshold = config["thresholds"]["high_confidence"]
#     medium_conf_threshold = config["thresholds"]["medium_confidence"]
#     low_conf_threshold = config["thresholds"]["low_confidence"]
    
#     classified = {
#         "high_confidence": [],
#         "medium_confidence": [],
#         "low_confidence": [],
#         "potential_matches": []
#     }
    
#     for result in results:
#         if not result.get("match", False):
#             continue
            
#         confidence = result.get("confidence", 0)
        
#         if confidence >= high_conf_threshold:
#             classified["high_confidence"].append(result)
#         elif confidence >= medium_conf_threshold:
#             classified["medium_confidence"].append(result)
#         elif confidence >= low_conf_threshold:
#             classified["low_confidence"].append(result)
#         else:
#             classified["potential_matches"].append(result)
    
#     return classified

# def print_enhanced_statistics(statistics: Dict, classified_matches: Dict):
#     """打印增强的统计信息"""
#     print("\n" + "="*70)
#     print("                    增强版匹配统计信息")
#     print("="*70)
    
#     # 基础统计
#     print(f"\n【数据规模统计】")
#     print(f"  总源表数量: {statistics['总源表数量']}")
#     print(f"  总目标表数量: {statistics['总目标表数量']}")
#     print(f"  总源字段数量: {statistics['总源字段数量']}")
#     print(f"  总目标字段数量: {statistics['总目标字段数量']}")
#     print(f"  理论最大字段对数量: {statistics['总源字段数量'] * statistics['总目标字段数量']:,}")
    
#     # 参与匹配的统计
#     print(f"\n【参与匹配统计】")
#     print(f"  参与匹配的源表数量: {statistics['参与匹配的源表数量']}")
#     print(f"  参与匹配的目标表数量: {statistics['参与匹配的目标表数量']}")
#     print(f"  参与匹配的源字段数量: {statistics['参与匹配的源字段数量']}")
#     print(f"  参与匹配的目标字段数量: {statistics['参与匹配的目标字段数量']}")
#     print(f"  实际比较的字段对数量: {statistics['候选匹配对数量']}")
    
#     # 多层次匹配结果统计
#     print(f"\n【多层次匹配结果统计】")
#     total_matches = sum(len(matches) for matches in classified_matches.values())
#     print(f"  总匹配数量: {total_matches}")
#     print(f"  高置信度匹配 (≥0.8): {len(classified_matches['high_confidence'])}")
#     print(f"  中等置信度匹配 (0.6-0.8): {len(classified_matches['medium_confidence'])}")
#     print(f"  低置信度匹配 (0.4-0.6): {len(classified_matches['low_confidence'])}")
#     print(f"  潜在匹配 (<0.4): {len(classified_matches['potential_matches'])}")
    
#     # 匹配率统计
#     if statistics['总源字段数量'] > 0:
#         total_match_rate = total_matches / statistics['总源字段数量'] * 100
#         high_match_rate = len(classified_matches['high_confidence']) / statistics['总源字段数量'] * 100
#         print(f"\n【匹配率统计】")
#         print(f"  总匹配率: {total_match_rate:.2f}%")
#         print(f"  高置信度匹配率: {high_match_rate:.2f}%")
    
#     # 效率统计
#     total_possible = statistics['总源字段数量'] * statistics['总目标字段数量']
#     actual_comparisons = statistics['候选匹配对数量']
#     if total_possible > 0:
#         efficiency = (1 - actual_comparisons / total_possible) * 100
#         print(f"\n【效率统计】")
#         print(f"  优化效率: 减少了 {efficiency:.2f}% 的字段比较")
#         print(f"  实际比较: {actual_comparisons:,} / {total_possible:,}")
    
#     print("="*70)

# def interactive_table_selection(source_schemas, target_schemas):
#     """交互式选择要匹配的表"""
#     print("\n=== 交互式表选择 ===")
    
#     print("\n源表列表:")
#     for i, schema in enumerate(source_schemas):
#         print(f"{i+1}. {schema.table_name} - {schema.table_desc}")
    
#     print("\n目标表列表:")
#     for i, schema in enumerate(target_schemas):
#         print(f"{i+1}. {schema.table_name} - {schema.table_desc}")
    
#     print("\n选择匹配方式:")
#     print("1. 自动选择潜在匹配的表 (基于相似度)")
#     print("2. 手动选择特定表")
#     print("3. 处理所有表")
    
#     choice = input("请选择 (1/2/3): ")
    
#     if choice == "1":
#         table_threshold = float(input("请输入表相似度阈值 (0-1, 建议0.2): ") or "0.2")
#         max_pairs = int(input("请输入最大表对数量 (建议50): ") or "50")
#         return filter_table_pairs(source_schemas, target_schemas, table_threshold, max_pairs), False
    
#     elif choice == "2":
#         selected_pairs = []
        
#         source_indices = input("请输入源表序号 (多个用逗号分隔，如1,3,5): ")
#         source_indices = [int(idx.strip()) - 1 for idx in source_indices.split(",") if idx.strip().isdigit()]
#         selected_sources = [source_schemas[idx] for idx in source_indices if 0 <= idx < len(source_schemas)]
        
#         target_indices = input("请输入目标表序号 (多个用逗号分隔，如1,3,5): ")
#         target_indices = [int(idx.strip()) - 1 for idx in target_indices.split(",") if idx.strip().isdigit()]
#         selected_targets = [target_schemas[idx] for idx in target_indices if 0 <= idx < len(target_schemas)]
        
#         for source, target in product(selected_sources, selected_targets):
#             selected_pairs.append((source, target))
        
#         return selected_pairs, False
    
#     else:
#         return [(s, t) for s, t in product(source_schemas, target_schemas)], True

# def main():
#     """增强版匹配主函数"""
#     parser = argparse.ArgumentParser(description='增强版Schema匹配（提升召回率）')
#     parser.add_argument('--source', type=str, default='data/源数据字典.xlsx', help='源表数据字典文件路径')
#     # parser.add_argument('--target', type=str, default='data/项目匹配字典.xlsx', help='目标表数据字典文件路径')
#     parser.add_argument('--target', type=str, default='data/项目匹配字典_列类型注释.xlsx', help='目标表数据字典文件路径')
#     parser.add_argument('--config', type=str, default='config/config_enhanced.yaml', help='增强配置文件路径')
#     parser.add_argument('--output', type=str, default='output', help='输出目录')
#     parser.add_argument('--auto', action='store_true', help='自动模式，不交互')
#     parser.add_argument('--max-pairs', type=int, default=50, help='最大表对数量')
#     parser.add_argument('--max-llm', type=int, default=200, help='最大LLM处理候选对数量')
#     args = parser.parse_args()
    
#     print("=== 增强版Schema匹配系统（提升召回率）===")
    
#     # 确保文件存在
#     if not os.path.exists(args.config):
#         print(f"错误: 配置文件不存在: {args.config}")
#         print(f"请确保使用增强配置文件: config/config_enhanced.yaml")
#         sys.exit(1)
    
#     if not os.path.exists(args.source) or not os.path.exists(args.target):
#         print(f"错误: 数据文件不存在")
#         sys.exit(1)
    
#     # 加载配置
#     with open(args.config, "r", encoding="utf-8") as f:
#         config = yaml.safe_load(f)
    
#     print(f"成功加载增强配置文件: {args.config}")
#     print(f"积极匹配模式: {'启用' if config['matching_strategy']['enable_aggressive_mode'] else '禁用'}")
    
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
    
#     # 初始化增强版组件
#     preprocessor = MetadataPreprocessor(
#         enable_pinyin=config["chinese"]["enable_pinyin"],
#         enable_abbreviation=config["chinese"]["enable_abbreviation"]
#     )
    
#     # 使用增强版相似度计算器
#     similarity_calculator = EnhancedSimilarityCalculator(
#         char_weight=config["similarity"]["char_weight"],
#         semantic_weight=config["similarity"]["semantic_weight"],
#         struct_weight=config["similarity"]["struct_weight"],
#         pinyin_boost=config["similarity"]["pinyin_boost"]
#     )
    
#     # 2. 表对筛选
#     if args.auto:
#         table_pairs, process_all = filter_table_pairs(source_schemas, target_schemas, max_pairs=args.max_pairs), False
#     else:
#         table_pairs, process_all = interactive_table_selection(source_schemas, target_schemas)
    
#     if process_all:
#         print(f"警告: 将处理所有 {len(source_schemas) * len(target_schemas)} 对表，可能需要大量时间")
#         confirmation = input("是否继续? (y/n): ")
#         if confirmation.lower() != 'y':
#             print("已取消操作")
#             return
    
#     # 3. 并行处理表对（使用增强版）
#     print(f"\n3. 开始处理 {len(table_pairs)} 对表（增强版相似度计算）...")
#     start_time = time.time()
    
#     pair_info_list = [
#         (source, target, preprocessor, similarity_calculator, config["thresholds"]["similarity_threshold"])
#         for source, target in table_pairs
#     ]
    
#     with multiprocessing.Pool(processes=min(os.cpu_count(), len(pair_info_list))) as pool:
#         results = pool.map(process_table_pair_enhanced, pair_info_list)
    
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
    
#     all_candidates.sort(key=lambda x: x["similarity"], reverse=True)
    
#     print(f"找到 {len(all_candidates)} 对候选字段匹配（使用增强相似度）")
    
#     # 显示相似度分布
#     high_sim = len([c for c in all_candidates if c["similarity"] >= 0.7])
#     medium_sim = len([c for c in all_candidates if 0.4 <= c["similarity"] < 0.7])
#     low_sim = len([c for c in all_candidates if c["similarity"] < 0.4])
#     print(f"  高相似度(≥0.7): {high_sim}")
#     print(f"  中等相似度(0.4-0.7): {medium_sim}")
#     print(f"  低相似度(<0.4): {low_sim}")
    
#     # 5. 应用匹配规则
#     print("\n4. 应用增强匹配规则...")
#     start_time = time.time()
    
#     candidate_filter = CandidateFilter(
#         similarity_threshold=config["thresholds"]["similarity_threshold"]
#     )
    
#     filtered_candidates = candidate_filter.apply_matching_rules(all_candidates)
    
#     print(f"规则应用完成，耗时: {time.time() - start_time:.2f}秒")
#     print(f"应用规则后保留 {len(filtered_candidates)} 对候选匹配")
    
#     # 6. 增强版LLM语义匹配
#     print("\n5. 增强版LLM语义匹配...")
#     start_time = time.time()
    
#     max_llm_candidates = args.max_llm
#     if len(filtered_candidates) > max_llm_candidates:
#         print(f"候选匹配对数量较多({len(filtered_candidates)}个)，只处理相似度较高的前{max_llm_candidates}个")
#         current_candidates = filtered_candidates[:max_llm_candidates]
#     else:
#         current_candidates = filtered_candidates
    
#     try:
#         # 使用增强版LLM匹配器
#         llm_matcher = EnhancedLLMMatcher(config_path=args.config)
#         matching_results = llm_matcher.batch_process_candidates(
#             current_candidates,
#             processed_source_schemas,
#             processed_target_schemas
#         )
        
#         print(f"增强版LLM匹配完成，耗时: {time.time() - start_time:.2f}秒")
#         print(f"处理了 {len(current_candidates)} 对候选匹配")
#     except Exception as e:
#         print(f"LLM匹配失败: {e}")
#         import traceback
#         traceback.print_exc()
        
#         # 增强的回退方案
#         print("使用增强相似度结果作为替代...")
#         matching_results = []
#         for candidate in current_candidates:
#             # 更积极的相似度判断
#             match_threshold = 0.6 if candidate["similarity"] > 0.8 else 0.4
#             matching_results.append({
#                 "source_table": candidate["source_table"],
#                 "source_field": candidate["source_field"],
#                 "target_table": candidate["target_table"],
#                 "target_field": candidate["target_field"],
#                 "match": candidate["similarity"] > match_threshold,
#                 "confidence": min(0.8, candidate["similarity"] + 0.1),
#                 "reason": f"基于增强相似度 {candidate['similarity']:.4f}",
#                 "similarity": candidate["similarity"]
#             })
    
#     # 7. 多层次结果处理
#     print("\n6. 多层次结果处理...")
#     start_time = time.time()
    
#     # 按置信度分类匹配结果
#     classified_matches = classify_matches_by_confidence(matching_results, config)
    
#     # 计算统计信息
#     result_processor = ResultProcessor(
#         confidence_threshold=config["thresholds"]["low_confidence"]  # 使用较低的阈值
#     )
    
#     statistics = result_processor.calculate_matching_statistics(
#         matching_results,
#         processed_source_schemas,
#         processed_target_schemas,
#         all_candidates
#     )
    
#     # 处理所有匹配结果（不仅仅是高置信度的）
#     all_matches = []
#     for category, matches in classified_matches.items():
#         for match in matches:
#             match["confidence_category"] = category
#             all_matches.append(match)
    
#     # 应用一对一约束到高置信度匹配
#     high_confidence_matches = result_processor.process_matching_results(
#         classified_matches["high_confidence"],
#         processed_source_schemas,
#         processed_target_schemas
#     )
    
#     print(f"结果处理完成，耗时: {time.time() - start_time:.2f}秒")
    
#     # 8. 保存增强结果
#     os.makedirs(args.output, exist_ok=True)
    
#     # 保存分层结果
#     timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    
#     # 保存高置信度匹配结果
#     high_conf_files = result_processor.save_results(high_confidence_matches, statistics, args.output)
    
#     # 保存所有匹配结果
#     all_matches_file = os.path.join(args.output, f"all_matches_{timestamp}.xlsx")
    
#     with pd.ExcelWriter(all_matches_file, engine='openpyxl') as writer:
#         for category, matches in classified_matches.items():
#             if matches:
#                 df = pd.DataFrame(matches)
#                 columns = {
#                     "source_table": "源表名",
#                     "source_field": "源字段名", 
#                     "target_table": "目标表名",
#                     "target_field": "目标字段名",
#                     "confidence": "匹配置信度",
#                     "similarity": "特征相似度",
#                     "reason": "匹配理由"
#                 }
#                 existing_columns = {k: v for k, v in columns.items() if k in df.columns}
#                 df_output = df[list(existing_columns.keys())].copy()
#                 df_output.rename(columns=existing_columns, inplace=True)
                
#                 sheet_name = {
#                     "high_confidence": "高置信度匹配",
#                     "medium_confidence": "中等置信度匹配", 
#                     "low_confidence": "低置信度匹配",
#                     "potential_matches": "潜在匹配"
#                 }[category]
                
#                 df_output.to_excel(writer, sheet_name=sheet_name, index=False)
    
#     # 输出增强统计信息
#     print_enhanced_statistics(statistics, classified_matches)
    
#     print(f"\n结果文件:")
#     print(f"  高置信度匹配: {high_conf_files['excel']}")
#     print(f"  所有分层匹配结果: {all_matches_file}")
#     print(f"  统计信息: {high_conf_files['statistics']}")
    
#     # 输出分层匹配结果示例
#     print(f"\n=== 分层匹配结果示例 ===")
    
#     for category, matches in classified_matches.items():
#         if matches:
#             category_name = {
#                 "high_confidence": "高置信度匹配",
#                 "medium_confidence": "中等置信度匹配",
#                 "low_confidence": "低置信度匹配", 
#                 "potential_matches": "潜在匹配"
#             }[category]
            
#             print(f"\n【{category_name}】({len(matches)}个)")
#             for i, result in enumerate(matches[:3]):  # 每类显示3个
#                 print(f"{i+1}. {result['source_table']}.{result['source_field']} <-> "
#                       f"{result['target_table']}.{result['target_field']}")
#                 print(f"   置信度: {result['confidence']:.2f}, 相似度: {result.get('similarity', 0):.2f}")
#                 print(f"   理由: {result.get('reason', '')[:100]}...")
            
#             if len(matches) > 3:
#                 print(f"   ... 还有 {len(matches) - 3} 个匹配")
    
#     total_matches = sum(len(matches) for matches in classified_matches.values())
#     print(f"\n总计找到 {total_matches} 个匹配，其中高置信度匹配 {len(classified_matches['high_confidence'])} 个")
#     print("\n增强版匹配实验完成！")

# if __name__ == "__main__":
#     main()
"""
修复版增强Schema匹配脚本 - 使用与测试脚本一致的API调用方法
"""
import os
import sys
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
import multiprocessing
from itertools import product
import argparse

from src.data.data_loader import DataLoader, SchemaMetadata
from src.data.data_preprocessor import MetadataPreprocessor
from src.features.enhanced_similarity_calculator import EnhancedSimilarityCalculator
from src.matching.candidate_filter import CandidateFilter
from src.matching.result_processor import ResultProcessor

# 修复版LLM匹配器 - 直接在脚本中定义，避免导入问题
import requests
import json
import re
from tqdm import tqdm

class FixedEnhancedLLMMatcher:
    """修复版增强LLM匹配器 - 使用与测试脚本一致的API调用"""
    
    def __init__(self, config_path: str = "config/config_enhanced.yaml"):
        print(f"加载配置文件: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        self.api_key = config["openai"]["api_key"]
        self.api_base_url = config["openai"]["api_base_url"]
        self.model = config["openai"]["model"]
        self.temperature = config["openai"]["temperature"]
        self.max_tokens = config["openai"]["max_tokens"]
        self.batch_size = config["system"]["batch_size"]
        self.cache_enabled = config["system"]["cache_enabled"]
        self.enable_aggressive_mode = config["matching_strategy"]["enable_aggressive_mode"]
        
        print(f"API配置: {self.api_base_url}, 模型: {self.model}")
        print(f"积极匹配模式: {'启用' if self.enable_aggressive_mode else '禁用'}")
        
        # 缓存
        self._cache = {}
        self._cache_path = "cache/fixed_enhanced_responses.json"
        if self.cache_enabled:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            if os.path.exists(self._cache_path):
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
        
        self.success_count = 0
        self.fail_count = 0
        self.cache_hit_count = 0
    
    def _call_api_like_test(self, prompt: str) -> str:
        """完全模仿测试脚本的API调用方法"""
        try:
            # 与test_api_detailed.py完全一致
            api_url = f"{self.api_base_url}/chat/completions"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # 使用与测试脚本相同的超时时间
            response = requests.post(api_url, headers=headers, json=data, timeout=30)
            
            # 与测试脚本一致的状态码检查
            if response.status_code != 200:
                return f"错误：API调用失败，状态码：{response.status_code}，响应：{response.text}"
            
            # 与测试脚本一致的响应解析
            try:
                resp_json = response.json()
            except json.JSONDecodeError:
                return f"错误：JSON解析失败，响应：{response.text[:200]}"
            
            # 与测试脚本完全一致的内容提取
            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                choice = resp_json["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    content = choice["message"]["content"]
                    if content and content.strip():
                        return content
                    else:
                        return "错误：API返回空内容"
                elif "text" in choice:
                    return choice["text"]
                else:
                    return f"错误：响应格式异常"
            else:
                return f"错误：响应中没有choices字段"
                
        except requests.exceptions.Timeout:
            return "错误：请求超时"
        except requests.exceptions.ConnectionError:
            return "错误：连接错误"
        except Exception as e:
            return f"错误：{str(e)}"
    
    def match_field_pair(self, source_field: Dict, target_field: Dict, similarity: float) -> Dict:
        """匹配单对字段"""
        # 缓存键
        cache_key = f"fixed_{source_field['name']}_{target_field['name']}_{similarity:.3f}"
        
        # 检查缓存
        if self.cache_enabled and cache_key in self._cache:
            self.cache_hit_count += 1
            return self._cache[cache_key]
        
        # 创建简化提示（避免过长的内容）
        prompt = f"""判断两个数据库字段是否语义等价：

源字段：{source_field['name']} ({source_field.get('desc', '无描述')})
目标字段：{target_field['name']} ({target_field.get('desc', '无描述')})
相似度：{similarity:.2f}

分析要点：
1. 字段名语义关系（考虑中英文对应、拼音缩写）
2. 业务概念是否相同
3. 描述内容匹配度

回答格式：
判断：是/否
置信度：0-1数字
理由：简要说明"""
        
        # 调用API
        response = self._call_api_like_test(prompt)
        
        if response.startswith("错误："):
            self.fail_count += 1
            result = self._generate_fallback_result(source_field, target_field, similarity, response)
        else:
            self.success_count += 1
            result = self._parse_response(response, similarity)
        
        result["similarity"] = similarity
        
        # 缓存结果
        if self.cache_enabled:
            self._cache[cache_key] = result
            if len(self._cache) % 10 == 0:
                self._save_cache()
        
        return result
    
    def _parse_response(self, response: str, similarity: float) -> Dict:
        """解析LLM响应"""
        try:
            # 提取判断
            match = False
            if any(indicator in response for indicator in ["判断：是", "判断:是", "判断: 是"]):
                match = True
            
            # 提取置信度
            confidence = 0.0
            patterns = [r"置信度：([0-9.]+)", r"置信度:([0-9.]+)", r"置信度: ([0-9.]+)"]
            
            for pattern in patterns:
                match_obj = re.search(pattern, response)
                if match_obj:
                    try:
                        confidence = float(match_obj.group(1))
                        break
                    except ValueError:
                        pass
            
            # 如果没有提取到置信度，根据匹配结果设置
            if confidence == 0.0:
                if match:
                    confidence = max(0.7, similarity + 0.1)
                else:
                    confidence = min(0.4, similarity)
            
            # 提取理由
            reason_patterns = [r"理由：(.*?)(?=\n|$)", r"理由:(.*?)(?=\n|$)", r"理由: (.*?)(?=\n|$)"]
            reason = ""
            
            for pattern in reason_patterns:
                match_obj = re.search(pattern, response, re.DOTALL)
                if match_obj:
                    reason = match_obj.group(1).strip()
                    break
            
            if not reason:
                reason = response[:100] + "..."
            
            return {
                "match": match,
                "confidence": confidence,
                "reason": reason,
                "llm_response": response
            }
            
        except Exception as e:
            return {
                "match": similarity > 0.6,
                "confidence": similarity,
                "reason": f"解析失败: {e}，基于相似度判断",
                "parse_error": True
            }
    
    def _generate_fallback_result(self, source_field: Dict, target_field: Dict, 
                                 similarity: float, error_msg: str) -> Dict:
        """生成回退结果"""
        s_name = source_field.get("name", "").lower()
        t_name = target_field.get("name", "").lower()
        
        # 增强的回退逻辑
        if s_name == t_name:
            match = True
            confidence = 0.95
        elif similarity >= 0.8:
            match = True
            confidence = 0.8
        elif similarity >= 0.6:
            match = True
            confidence = 0.7
        elif similarity >= 0.4 and self.enable_aggressive_mode:
            match = True
            confidence = 0.6
        else:
            match = False
            confidence = similarity
        
        reason = f"API调用失败，基于相似度{similarity:.2f}的智能回退。错误：{error_msg[:50]}"
        
        return {
            "match": match,
            "confidence": confidence,
            "reason": reason,
            "fallback": True
        }
    
    def batch_process_candidates(self, candidate_pairs: List[Dict], 
                                source_schemas: Dict[str, Dict],
                                target_schemas: Dict[str, Dict]) -> List[Dict]:
        """批量处理候选匹配对"""
        results = []
        
        print(f"开始处理 {len(candidate_pairs)} 对候选匹配...")
        
        # 分层处理：高相似度优先
        high_sim = [c for c in candidate_pairs if c["similarity"] >= 0.7]
        medium_sim = [c for c in candidate_pairs if 0.4 <= c["similarity"] < 0.7]
        low_sim = [c for c in candidate_pairs if c["similarity"] < 0.4]
        
        print(f"高相似度: {len(high_sim)}, 中等相似度: {len(medium_sim)}, 低相似度: {len(low_sim)}")
        
        # 处理各层
        for candidates, desc in [(high_sim, "高相似度"), (medium_sim, "中等相似度"), (low_sim, "低相似度")]:
            if not candidates:
                continue
                
            print(f"处理{desc}候选对...")
            
            for i, candidate in enumerate(tqdm(candidates, desc=f"处理{desc}")):
                try:
                    source_schema = source_schemas[candidate["source_table"]]
                    source_field = next(f for f in source_schema["fields"] if f["name"] == candidate["source_field"])
                    
                    target_schema = target_schemas[candidate["target_table"]]
                    target_field = next(f for f in target_schema["fields"] if f["name"] == candidate["target_field"])
                    
                    result = self.match_field_pair(source_field, target_field, candidate["similarity"])
                    
                    result["source_table"] = candidate["source_table"]
                    result["source_field"] = candidate["source_field"]
                    result["target_table"] = candidate["target_table"]
                    result["target_field"] = candidate["target_field"]
                    
                    results.append(result)
                    
                    # 增加间隔，避免API限制
                    if not result.get("fallback", False):  # 只有真正调用API时才等待
                        time.sleep(1)  # 1秒间隔
                        
                except Exception as e:
                    print(f"处理失败: {candidate}, 错误: {e}")
                    error_result = {
                        "source_table": candidate["source_table"],
                        "source_field": candidate["source_field"],
                        "target_table": candidate["target_table"],
                        "target_field": candidate["target_field"],
                        "match": False,
                        "confidence": 0.0,
                        "reason": f"处理失败: {e}",
                        "similarity": candidate["similarity"],
                        "error": True
                    }
                    results.append(error_result)
        
        # 保存缓存
        if self.cache_enabled:
            self._save_cache()
        
        # 打印统计
        print(f"\n=== API调用统计 ===")
        print(f"成功调用: {self.success_count}")
        print(f"失败调用: {self.fail_count}")
        print(f"缓存命中: {self.cache_hit_count}")
        total = self.success_count + self.fail_count
        if total > 0:
            print(f"成功率: {self.success_count/total*100:.1f}%")
        
        return results
    
    def _save_cache(self):
        """保存缓存"""
        try:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存失败: {e}")


def calculate_table_similarity(source_schema, target_schema):
    """计算表级别相似度"""
    source_name = source_schema.table_name.lower()
    target_name = target_schema.table_name.lower()
    
    import difflib
    name_sim = difflib.SequenceMatcher(None, source_name, target_name).ratio()
    
    source_desc = source_schema.table_desc.lower() if source_schema.table_desc else ""
    target_desc = target_schema.table_desc.lower() if target_schema.table_desc else ""
    
    if source_desc and target_desc:
        desc_sim = difflib.SequenceMatcher(None, source_desc, target_desc).ratio()
    else:
        desc_sim = 0
    
    table_sim = 0.7 * name_sim + 0.3 * desc_sim
    return table_sim

def filter_table_pairs(source_schemas, target_schemas, table_threshold=0.2, max_pairs=50):
    """基于表相似度筛选表对"""
    print(f"筛选潜在匹配的表对...")
    
    table_similarities = []
    
    for source_schema in source_schemas:
        for target_schema in target_schemas:
            similarity = calculate_table_similarity(source_schema, target_schema)
            if similarity >= table_threshold:
                table_similarities.append((source_schema, target_schema, similarity))
    
    table_similarities.sort(key=lambda x: x[2], reverse=True)
    
    if len(table_similarities) > max_pairs:
        table_similarities = table_similarities[:max_pairs]
    
    selected_pairs = [(s, t) for s, t, _ in table_similarities]
    
    print(f"筛选出 {len(selected_pairs)} 对潜在匹配的表")
    
    return selected_pairs

def classify_matches_by_confidence(results: List[Dict], config: Dict) -> Dict[str, List[Dict]]:
    """根据置信度分类匹配结果"""
    classified = {
        "high_confidence": [],
        "medium_confidence": [],
        "low_confidence": [],
        "potential_matches": []
    }
    
    for result in results:
        if not result.get("match", False):
            continue
            
        confidence = result.get("confidence", 0)
        
        if confidence >= 0.8:
            classified["high_confidence"].append(result)
        elif confidence >= 0.6:
            classified["medium_confidence"].append(result)
        elif confidence >= 0.4:
            classified["low_confidence"].append(result)
        else:
            classified["potential_matches"].append(result)
    
    return classified

def main():
    """修复版匹配主函数"""
    parser = argparse.ArgumentParser(description='修复版增强Schema匹配')
    parser.add_argument('--source', type=str, default='data/源数据字典.xlsx', help='源表数据字典文件路径')
    parser.add_argument('--target', type=str, default='data/项目匹配字典_列类型注释.xlsx', help='目标表数据字典文件路径')
    parser.add_argument('--config', type=str, default='config/config_enhanced.yaml', help='配置文件路径')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--auto', action='store_true', help='自动模式，不交互')
    parser.add_argument('--max-pairs', type=int, default=20, help='最大表对数量')
    parser.add_argument('--max-llm', type=int, default=50, help='最大LLM处理候选对数量')
    args = parser.parse_args()
    
    print("=== 修复版增强Schema匹配系统 ===")
    print("使用与API测试脚本一致的调用方法")
    
    # 检查文件
    for file_path, desc in [(args.config, "配置文件"), (args.source, "源数据文件"), (args.target, "目标数据文件")]:
        if not os.path.exists(file_path):
            print(f"错误: {desc}不存在: {file_path}")
            sys.exit(1)
    
    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print(f"使用配置文件: {args.config}")
    
    # 1. 数据加载
    print("\n1. 数据加载...")
    start_time = time.time()
    data_loader = DataLoader()
    
    try:
        source_schemas = data_loader.load_excel_dictionary(args.source)
        target_schemas = data_loader.load_excel_dictionary(args.target)
        
        print(f"加载完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"源表数量: {len(source_schemas)}, 目标表数量: {len(target_schemas)}")
        
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
    
    # 3. 表对筛选
    if args.auto:
        table_pairs = filter_table_pairs(source_schemas, target_schemas, max_pairs=args.max_pairs)
    else:
        # 简化版本，只取前几个表
        max_tables = min(3, len(source_schemas), len(target_schemas))
        table_pairs = [(source_schemas[i], target_schemas[j]) 
                      for i in range(max_tables) for j in range(max_tables)]
    
    print(f"将处理 {len(table_pairs)} 对表")
    
    # 4. 增强相似度计算
    print("\n3. 增强相似度计算...")
    start_time = time.time()
    
    similarity_calculator = EnhancedSimilarityCalculator(
        char_weight=config["similarity"]["char_weight"],
        semantic_weight=config["similarity"]["semantic_weight"],
        struct_weight=config["similarity"]["struct_weight"],
        pinyin_boost=config["similarity"]["pinyin_boost"]
    )
    
    all_candidates = []
    
    for source_schema, target_schema in table_pairs:
        print(f"计算表 {source_schema.table_name} 和 {target_schema.table_name} 的相似度...")
        
        source_processed = processed_source_schemas[source_schema.table_name]
        target_processed = processed_target_schemas[target_schema.table_name]
        
        matrix = similarity_calculator.calculate_similarity_matrix(
            source_processed["fields"],
            target_processed["fields"]
        )
        
        # 筛选候选对
        for i, s_field in enumerate(source_processed["fields"]):
            for j, t_field in enumerate(target_processed["fields"]):
                sim = matrix[i, j]
                if sim >= config["thresholds"]["similarity_threshold"]:
                    all_candidates.append({
                        "source_table": source_schema.table_name,
                        "source_field": s_field["name"],
                        "target_table": target_schema.table_name,
                        "target_field": t_field["name"],
                        "similarity": float(sim)
                    })
    
    all_candidates.sort(key=lambda x: x["similarity"], reverse=True)
    
    print(f"相似度计算完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"找到 {len(all_candidates)} 对候选字段匹配")
    
    # 5. 应用匹配规则
    print("\n4. 应用匹配规则...")
    candidate_filter = CandidateFilter(
        similarity_threshold=config["thresholds"]["similarity_threshold"]
    )
    
    filtered_candidates = candidate_filter.apply_matching_rules(all_candidates)
    print(f"应用规则后保留 {len(filtered_candidates)} 对候选匹配")
    
    # 限制LLM处理数量
    max_llm_candidates = args.max_llm
    if len(filtered_candidates) > max_llm_candidates:
        print(f"候选匹配对较多，只处理前 {max_llm_candidates} 个")
        current_candidates = filtered_candidates[:max_llm_candidates]
    else:
        current_candidates = filtered_candidates
    
    # 6. 修复版LLM匹配
    print("\n5. 修复版LLM匹配...")
    start_time = time.time()
    
    try:
        llm_matcher = FixedEnhancedLLMMatcher(config_path=args.config)
        matching_results = llm_matcher.batch_process_candidates(
            current_candidates,
            processed_source_schemas,
            processed_target_schemas
        )
        
        print(f"LLM匹配完成，耗时: {time.time() - start_time:.2f}秒")
        
    except Exception as e:
        print(f"LLM匹配失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 7. 多层次结果处理
    print("\n6. 多层次结果处理...")
    
    # 按置信度分类
    classified_matches = classify_matches_by_confidence(matching_results, config)
    
    # 计算统计信息
    result_processor = ResultProcessor(
        confidence_threshold=config["thresholds"]["low_confidence"]
    )
    
    statistics = result_processor.calculate_matching_statistics(
        matching_results,
        processed_source_schemas,
        processed_target_schemas,
        all_candidates
    )
    
    # 处理高置信度匹配
    high_confidence_matches = result_processor.process_matching_results(
        classified_matches["high_confidence"],
        processed_source_schemas,
        processed_target_schemas
    )
    
    # 8. 保存结果
    os.makedirs(args.output, exist_ok=True)
    
    # 保存高置信度结果
    high_conf_files = result_processor.save_results(high_confidence_matches, statistics, args.output)
    
    # 保存所有分层结果
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    all_matches_file = os.path.join(args.output, f"fixed_enhanced_matches_{timestamp}.xlsx")
    
    with pd.ExcelWriter(all_matches_file, engine='openpyxl') as writer:
        for category, matches in classified_matches.items():
            if matches:
                df = pd.DataFrame(matches)
                columns = {
                    "source_table": "源表名",
                    "source_field": "源字段名", 
                    "target_table": "目标表名",
                    "target_field": "目标字段名",
                    "confidence": "匹配置信度",
                    "similarity": "特征相似度",
                    "reason": "匹配理由"
                }
                existing_columns = {k: v for k, v in columns.items() if k in df.columns}
                df_output = df[list(existing_columns.keys())].copy()
                df_output.rename(columns=existing_columns, inplace=True)
                
                sheet_name = {
                    "high_confidence": "高置信度匹配",
                    "medium_confidence": "中等置信度匹配", 
                    "low_confidence": "低置信度匹配",
                    "potential_matches": "潜在匹配"
                }[category]
                
                df_output.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # 9. 输出结果
    print("\n" + "="*60)
    print("                  修复版匹配结果总结")
    print("="*60)
    
    total_matches = sum(len(matches) for matches in classified_matches.values())
    
    print(f"\n【匹配结果】")
    print(f"  总匹配数量: {total_matches}")
    print(f"  高置信度匹配: {len(classified_matches['high_confidence'])}")
    print(f"  中等置信度匹配: {len(classified_matches['medium_confidence'])}")
    print(f"  低置信度匹配: {len(classified_matches['low_confidence'])}")
    print(f"  潜在匹配: {len(classified_matches['potential_matches'])}")
    
    print(f"\n【输出文件】")
    print(f"  高置信度匹配: {high_conf_files['excel']}")
    print(f"  所有分层匹配结果: {all_matches_file}")
    print(f"  统计信息: {high_conf_files['statistics']}")
    
    # 显示匹配结果示例
    if total_matches > 0:
        print(f"\n【匹配结果示例】")
        for category, matches in classified_matches.items():
            if matches:
                category_name = {
                    "high_confidence": "高置信度匹配",
                    "medium_confidence": "中等置信度匹配",
                    "low_confidence": "低置信度匹配", 
                    "potential_matches": "潜在匹配"
                }[category]
                
                print(f"\n{category_name} ({len(matches)}个):")
                for i, result in enumerate(matches[:2]):  # 每类显示2个
                    fallback_mark = " [回退]" if result.get("fallback", False) else ""
                    print(f"  {i+1}. {result['source_table']}.{result['source_field']} <-> "
                          f"{result['target_table']}.{result['target_field']}{fallback_mark}")
                    print(f"     置信度: {result['confidence']:.2f}, 相似度: {result.get('similarity', 0):.2f}")
    else:
        print(f"\n未找到匹配结果")
    
    print("\n修复版增强匹配实验完成！")

if __name__ == "__main__":
    main()