"""
增强版Schema匹配脚本 - 提升匹配召回率同时保持准确性
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
from src.matching.enhanced_llm_matcher import EnhancedLLMMatcher
from src.matching.result_processor import ResultProcessor

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
    """基于表相似度筛选表对（降低阈值）"""
    print(f"开始筛选潜在匹配的表对，总共 {len(source_schemas)} 个源表和 {len(target_schemas)} 个目标表...")
    
    table_similarities = []
    
    for source_schema in source_schemas:
        for target_schema in target_schemas:
            similarity = calculate_table_similarity(source_schema, target_schema)
            if similarity >= table_threshold:
                table_similarities.append((source_schema, target_schema, similarity))
    
    # 按相似度排序
    table_similarities.sort(key=lambda x: x[2], reverse=True)
    
    if len(table_similarities) > max_pairs:
        print(f"表对数量 ({len(table_similarities)}) 超过限制，只保留相似度最高的 {max_pairs} 对")
        table_similarities = table_similarities[:max_pairs]
    
    selected_pairs = [(s, t) for s, t, _ in table_similarities]
    
    print(f"筛选出 {len(selected_pairs)} 对潜在匹配的表")
    for i, (source, target, sim) in enumerate(table_similarities):
        print(f"{i+1}. {source.table_name} <-> {target.table_name} (相似度: {sim:.4f})")
    
    return selected_pairs

def process_table_pair_enhanced(pair_info):
    """处理单对表（增强版）"""
    source_schema, target_schema, preprocessor, similarity_calculator, similarity_threshold = pair_info
    
    # 预处理
    processed_source = preprocessor.preprocess_schema(source_schema)
    processed_target = preprocessor.preprocess_schema(target_schema)
    
    # 使用增强版相似度计算
    matrix = similarity_calculator.calculate_similarity_matrix(
        processed_source["fields"],
        processed_target["fields"]
    )
    
    # 筛选候选对（使用更低的阈值）
    candidates = []
    for i, s_field in enumerate(processed_source["fields"]):
        for j, t_field in enumerate(processed_target["fields"]):
            sim = matrix[i, j]
            if sim >= similarity_threshold:
                candidates.append({
                    "source_table": source_schema.table_name,
                    "source_field": s_field["name"],
                    "target_table": target_schema.table_name,
                    "target_field": t_field["name"],
                    "similarity": float(sim)
                })
    
    # 排序
    candidates.sort(key=lambda x: x["similarity"], reverse=True)
    
    return {
        "source_schema": processed_source,
        "target_schema": processed_target,
        "matrix": matrix,
        "candidates": candidates
    }

def classify_matches_by_confidence(results: List[Dict], config: Dict) -> Dict[str, List[Dict]]:
    """根据置信度分类匹配结果"""
    high_conf_threshold = config["thresholds"]["high_confidence"]
    medium_conf_threshold = config["thresholds"]["medium_confidence"]
    low_conf_threshold = config["thresholds"]["low_confidence"]
    
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
        
        if confidence >= high_conf_threshold:
            classified["high_confidence"].append(result)
        elif confidence >= medium_conf_threshold:
            classified["medium_confidence"].append(result)
        elif confidence >= low_conf_threshold:
            classified["low_confidence"].append(result)
        else:
            classified["potential_matches"].append(result)
    
    return classified

def print_enhanced_statistics(statistics: Dict, classified_matches: Dict):
    """打印增强的统计信息"""
    print("\n" + "="*70)
    print("                    增强版匹配统计信息")
    print("="*70)
    
    # 基础统计
    print(f"\n【数据规模统计】")
    print(f"  总源表数量: {statistics['总源表数量']}")
    print(f"  总目标表数量: {statistics['总目标表数量']}")
    print(f"  总源字段数量: {statistics['总源字段数量']}")
    print(f"  总目标字段数量: {statistics['总目标字段数量']}")
    print(f"  理论最大字段对数量: {statistics['总源字段数量'] * statistics['总目标字段数量']:,}")
    
    # 参与匹配的统计
    print(f"\n【参与匹配统计】")
    print(f"  参与匹配的源表数量: {statistics['参与匹配的源表数量']}")
    print(f"  参与匹配的目标表数量: {statistics['参与匹配的目标表数量']}")
    print(f"  参与匹配的源字段数量: {statistics['参与匹配的源字段数量']}")
    print(f"  参与匹配的目标字段数量: {statistics['参与匹配的目标字段数量']}")
    print(f"  实际比较的字段对数量: {statistics['候选匹配对数量']}")
    
    # 多层次匹配结果统计
    print(f"\n【多层次匹配结果统计】")
    total_matches = sum(len(matches) for matches in classified_matches.values())
    print(f"  总匹配数量: {total_matches}")
    print(f"  高置信度匹配 (≥0.8): {len(classified_matches['high_confidence'])}")
    print(f"  中等置信度匹配 (0.6-0.8): {len(classified_matches['medium_confidence'])}")
    print(f"  低置信度匹配 (0.4-0.6): {len(classified_matches['low_confidence'])}")
    print(f"  潜在匹配 (<0.4): {len(classified_matches['potential_matches'])}")
    
    # 匹配率统计
    if statistics['总源字段数量'] > 0:
        total_match_rate = total_matches / statistics['总源字段数量'] * 100
        high_match_rate = len(classified_matches['high_confidence']) / statistics['总源字段数量'] * 100
        print(f"\n【匹配率统计】")
        print(f"  总匹配率: {total_match_rate:.2f}%")
        print(f"  高置信度匹配率: {high_match_rate:.2f}%")
    
    # 效率统计
    total_possible = statistics['总源字段数量'] * statistics['总目标字段数量']
    actual_comparisons = statistics['候选匹配对数量']
    if total_possible > 0:
        efficiency = (1 - actual_comparisons / total_possible) * 100
        print(f"\n【效率统计】")
        print(f"  优化效率: 减少了 {efficiency:.2f}% 的字段比较")
        print(f"  实际比较: {actual_comparisons:,} / {total_possible:,}")
    
    print("="*70)

def interactive_table_selection(source_schemas, target_schemas):
    """交互式选择要匹配的表"""
    print("\n=== 交互式表选择 ===")
    
    print("\n源表列表:")
    for i, schema in enumerate(source_schemas):
        print(f"{i+1}. {schema.table_name} - {schema.table_desc}")
    
    print("\n目标表列表:")
    for i, schema in enumerate(target_schemas):
        print(f"{i+1}. {schema.table_name} - {schema.table_desc}")
    
    print("\n选择匹配方式:")
    print("1. 自动选择潜在匹配的表 (基于相似度)")
    print("2. 手动选择特定表")
    print("3. 处理所有表")
    
    choice = input("请选择 (1/2/3): ")
    
    if choice == "1":
        table_threshold = float(input("请输入表相似度阈值 (0-1, 建议0.2): ") or "0.2")
        max_pairs = int(input("请输入最大表对数量 (建议50): ") or "50")
        return filter_table_pairs(source_schemas, target_schemas, table_threshold, max_pairs), False
    
    elif choice == "2":
        selected_pairs = []
        
        source_indices = input("请输入源表序号 (多个用逗号分隔，如1,3,5): ")
        source_indices = [int(idx.strip()) - 1 for idx in source_indices.split(",") if idx.strip().isdigit()]
        selected_sources = [source_schemas[idx] for idx in source_indices if 0 <= idx < len(source_schemas)]
        
        target_indices = input("请输入目标表序号 (多个用逗号分隔，如1,3,5): ")
        target_indices = [int(idx.strip()) - 1 for idx in target_indices.split(",") if idx.strip().isdigit()]
        selected_targets = [target_schemas[idx] for idx in target_indices if 0 <= idx < len(target_schemas)]
        
        for source, target in product(selected_sources, selected_targets):
            selected_pairs.append((source, target))
        
        return selected_pairs, False
    
    else:
        return [(s, t) for s, t in product(source_schemas, target_schemas)], True

def main():
    """增强版匹配主函数"""
    parser = argparse.ArgumentParser(description='增强版Schema匹配（提升召回率）')
    parser.add_argument('--source', type=str, default='data/源数据字典.xlsx', help='源表数据字典文件路径')
    # parser.add_argument('--target', type=str, default='data/项目匹配字典.xlsx', help='目标表数据字典文件路径')
    parser.add_argument('--target', type=str, default='data/项目匹配字典_列类型注释.xlsx', help='目标表数据字典文件路径')
    parser.add_argument('--config', type=str, default='config/config_enhanced.yaml', help='增强配置文件路径')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--auto', action='store_true', help='自动模式，不交互')
    parser.add_argument('--max-pairs', type=int, default=50, help='最大表对数量')
    parser.add_argument('--max-llm', type=int, default=200, help='最大LLM处理候选对数量')
    args = parser.parse_args()
    
    print("=== 增强版Schema匹配系统（提升召回率）===")
    
    # 确保文件存在
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        print(f"请确保使用增强配置文件: config/config_enhanced.yaml")
        sys.exit(1)
    
    if not os.path.exists(args.source) or not os.path.exists(args.target):
        print(f"错误: 数据文件不存在")
        sys.exit(1)
    
    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print(f"成功加载增强配置文件: {args.config}")
    print(f"积极匹配模式: {'启用' if config['matching_strategy']['enable_aggressive_mode'] else '禁用'}")
    
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
    
    # 初始化增强版组件
    preprocessor = MetadataPreprocessor(
        enable_pinyin=config["chinese"]["enable_pinyin"],
        enable_abbreviation=config["chinese"]["enable_abbreviation"]
    )
    
    # 使用增强版相似度计算器
    similarity_calculator = EnhancedSimilarityCalculator(
        char_weight=config["similarity"]["char_weight"],
        semantic_weight=config["similarity"]["semantic_weight"],
        struct_weight=config["similarity"]["struct_weight"],
        pinyin_boost=config["similarity"]["pinyin_boost"]
    )
    
    # 2. 表对筛选
    if args.auto:
        table_pairs, process_all = filter_table_pairs(source_schemas, target_schemas, max_pairs=args.max_pairs), False
    else:
        table_pairs, process_all = interactive_table_selection(source_schemas, target_schemas)
    
    if process_all:
        print(f"警告: 将处理所有 {len(source_schemas) * len(target_schemas)} 对表，可能需要大量时间")
        confirmation = input("是否继续? (y/n): ")
        if confirmation.lower() != 'y':
            print("已取消操作")
            return
    
    # 3. 并行处理表对（使用增强版）
    print(f"\n3. 开始处理 {len(table_pairs)} 对表（增强版相似度计算）...")
    start_time = time.time()
    
    pair_info_list = [
        (source, target, preprocessor, similarity_calculator, config["thresholds"]["similarity_threshold"])
        for source, target in table_pairs
    ]
    
    with multiprocessing.Pool(processes=min(os.cpu_count(), len(pair_info_list))) as pool:
        results = pool.map(process_table_pair_enhanced, pair_info_list)
    
    print(f"表对处理完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 4. 整合结果
    processed_source_schemas = {}
    processed_target_schemas = {}
    similarity_matrices = {}
    all_candidates = []
    
    for result in results:
        source_schema = result["source_schema"]
        target_schema = result["target_schema"]
        
        processed_source_schemas[source_schema["table_name"]] = source_schema
        processed_target_schemas[target_schema["table_name"]] = target_schema
        
        matrix_key = (source_schema["table_name"], target_schema["table_name"])
        similarity_matrices[matrix_key] = result["matrix"]
        
        all_candidates.extend(result["candidates"])
    
    all_candidates.sort(key=lambda x: x["similarity"], reverse=True)
    
    print(f"找到 {len(all_candidates)} 对候选字段匹配（使用增强相似度）")
    
    # 显示相似度分布
    high_sim = len([c for c in all_candidates if c["similarity"] >= 0.7])
    medium_sim = len([c for c in all_candidates if 0.4 <= c["similarity"] < 0.7])
    low_sim = len([c for c in all_candidates if c["similarity"] < 0.4])
    print(f"  高相似度(≥0.7): {high_sim}")
    print(f"  中等相似度(0.4-0.7): {medium_sim}")
    print(f"  低相似度(<0.4): {low_sim}")
    
    # 5. 应用匹配规则
    print("\n4. 应用增强匹配规则...")
    start_time = time.time()
    
    candidate_filter = CandidateFilter(
        similarity_threshold=config["thresholds"]["similarity_threshold"]
    )
    
    filtered_candidates = candidate_filter.apply_matching_rules(all_candidates)
    
    print(f"规则应用完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"应用规则后保留 {len(filtered_candidates)} 对候选匹配")
    
    # 6. 增强版LLM语义匹配
    print("\n5. 增强版LLM语义匹配...")
    start_time = time.time()
    
    max_llm_candidates = args.max_llm
    if len(filtered_candidates) > max_llm_candidates:
        print(f"候选匹配对数量较多({len(filtered_candidates)}个)，只处理相似度较高的前{max_llm_candidates}个")
        current_candidates = filtered_candidates[:max_llm_candidates]
    else:
        current_candidates = filtered_candidates
    
    try:
        # 使用增强版LLM匹配器
        llm_matcher = EnhancedLLMMatcher(config_path=args.config)
        matching_results = llm_matcher.batch_process_candidates(
            current_candidates,
            processed_source_schemas,
            processed_target_schemas
        )
        
        print(f"增强版LLM匹配完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"处理了 {len(current_candidates)} 对候选匹配")
    except Exception as e:
        print(f"LLM匹配失败: {e}")
        import traceback
        traceback.print_exc()
        
        # 增强的回退方案
        print("使用增强相似度结果作为替代...")
        matching_results = []
        for candidate in current_candidates:
            # 更积极的相似度判断
            match_threshold = 0.6 if candidate["similarity"] > 0.8 else 0.4
            matching_results.append({
                "source_table": candidate["source_table"],
                "source_field": candidate["source_field"],
                "target_table": candidate["target_table"],
                "target_field": candidate["target_field"],
                "match": candidate["similarity"] > match_threshold,
                "confidence": min(0.8, candidate["similarity"] + 0.1),
                "reason": f"基于增强相似度 {candidate['similarity']:.4f}",
                "similarity": candidate["similarity"]
            })
    
    # 7. 多层次结果处理
    print("\n6. 多层次结果处理...")
    start_time = time.time()
    
    # 按置信度分类匹配结果
    classified_matches = classify_matches_by_confidence(matching_results, config)
    
    # 计算统计信息
    result_processor = ResultProcessor(
        confidence_threshold=config["thresholds"]["low_confidence"]  # 使用较低的阈值
    )
    
    statistics = result_processor.calculate_matching_statistics(
        matching_results,
        processed_source_schemas,
        processed_target_schemas,
        all_candidates
    )
    
    # 处理所有匹配结果（不仅仅是高置信度的）
    all_matches = []
    for category, matches in classified_matches.items():
        for match in matches:
            match["confidence_category"] = category
            all_matches.append(match)
    
    # 应用一对一约束到高置信度匹配
    high_confidence_matches = result_processor.process_matching_results(
        classified_matches["high_confidence"],
        processed_source_schemas,
        processed_target_schemas
    )
    
    print(f"结果处理完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 8. 保存增强结果
    os.makedirs(args.output, exist_ok=True)
    
    # 保存分层结果
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    
    # 保存高置信度匹配结果
    high_conf_files = result_processor.save_results(high_confidence_matches, statistics, args.output)
    
    # 保存所有匹配结果
    all_matches_file = os.path.join(args.output, f"all_matches_{timestamp}.xlsx")
    
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
    
    # 输出增强统计信息
    print_enhanced_statistics(statistics, classified_matches)
    
    print(f"\n结果文件:")
    print(f"  高置信度匹配: {high_conf_files['excel']}")
    print(f"  所有分层匹配结果: {all_matches_file}")
    print(f"  统计信息: {high_conf_files['statistics']}")
    
    # 输出分层匹配结果示例
    print(f"\n=== 分层匹配结果示例 ===")
    
    for category, matches in classified_matches.items():
        if matches:
            category_name = {
                "high_confidence": "高置信度匹配",
                "medium_confidence": "中等置信度匹配",
                "low_confidence": "低置信度匹配", 
                "potential_matches": "潜在匹配"
            }[category]
            
            print(f"\n【{category_name}】({len(matches)}个)")
            for i, result in enumerate(matches[:3]):  # 每类显示3个
                print(f"{i+1}. {result['source_table']}.{result['source_field']} <-> "
                      f"{result['target_table']}.{result['target_field']}")
                print(f"   置信度: {result['confidence']:.2f}, 相似度: {result.get('similarity', 0):.2f}")
                print(f"   理由: {result.get('reason', '')[:100]}...")
            
            if len(matches) > 3:
                print(f"   ... 还有 {len(matches) - 3} 个匹配")
    
    total_matches = sum(len(matches) for matches in classified_matches.values())
    print(f"\n总计找到 {total_matches} 个匹配，其中高置信度匹配 {len(classified_matches['high_confidence'])} 个")
    print("\n增强版匹配实验完成！")

if __name__ == "__main__":
    main()