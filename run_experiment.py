"""
基于元数据的Schema匹配实验运行脚本
"""
import os
import yaml
import pandas as pd
from src.data.data_loader import DataLoader, SchemaMetadata
from src.data.data_preprocessor import MetadataPreprocessor
from src.features.similarity_calculator import SimilarityCalculator
from src.matching.candidate_filter import CandidateFilter
from src.matching.llm_matcher import LLMMatcher
from src.matching.result_processor import ResultProcessor

def main():
    """运行匹配实验"""
    print("=== 基于元数据的Schema匹配实验 ===")
    
    # 配置路径
    config_path = "config/config.yaml"
    source_file = "data/源数据字典.xlsx"
    target_file = "data/项目匹配字典.xlsx"
    output_dir = "output"
    
    # 确保文件存在
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    if not os.path.exists(source_file):
        raise FileNotFoundError(f"源数据字典文件不存在: {source_file}")
    if not os.path.exists(target_file):
        raise FileNotFoundError(f"项目匹配字典文件不存在: {target_file}")
    
    # 加载配置
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print(f"成功加载配置文件: {config_path}")
    print(f"使用模型: {config['openai']['model']}")
    print(f"API Base URL: {config['openai'].get('api_base_url', 'Default OpenAI API')}")
    
    # 加载数据
    print("\n1. 数据加载...")
    data_loader = DataLoader()
    try:
        source_schemas, target_schemas = data_loader.load_schemas(source_file, target_file)
        print(f"加载完成，源表数量: {len(source_schemas)}, 目标表数量: {len(target_schemas)}")
        
        # 打印部分源表和目标表信息做验证
        if source_schemas:
            print(f"\n源表示例: {source_schemas[0].table_name}")
            print(f"源表字段数: {len(source_schemas[0].fields)}")
            print(f"前3个字段: {', '.join([f['name'] for f in source_schemas[0].fields[:3]])}")
        
        if target_schemas:
            print(f"\n目标表示例: {target_schemas[0].table_name}")
            print(f"目标表字段数: {len(target_schemas[0].fields)}")
            print(f"前3个字段: {', '.join([f['name'] for f in target_schemas[0].fields[:3]])}")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return
    
    # 元数据预处理
    print("\n2. 元数据预处理...")
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
    
    print(f"预处理完成，处理了 {len(processed_source_schemas)} 个源表和 {len(processed_target_schemas)} 个目标表")
    
    # 计算相似度
    print("\n3. 计算字段相似度...")
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
    
    print(f"相似度计算完成，共计算了 {total_pairs} 对字段相似度")
    
    # 候选对筛选
    print("\n4. 候选对筛选...")
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
    
    print(f"筛选完成，从 {total_pairs} 对字段中筛选出 {len(candidates)} 对候选匹配，" 
          f"应用规则后保留 {len(filtered_candidates)} 对")
    
    if filtered_candidates:
        print("\n候选匹配对示例:")
        for i, candidate in enumerate(filtered_candidates[:3]):
            print(f"{i+1}. 源表: {candidate['source_table']}, 源字段: {candidate['source_field']}")
            print(f"   目标表: {candidate['target_table']}, 目标字段: {candidate['target_field']}")
            print(f"   相似度: {candidate['similarity']:.4f}")
    
    # LLM语义匹配
    print("\n5. LLM语义匹配...")
    try:
        llm_matcher = LLMMatcher(config_path=config_path)
        matching_results = llm_matcher.batch_process_candidates(
            filtered_candidates,
            processed_source_schemas,
            processed_target_schemas
        )
        print(f"LLM匹配完成，处理了 {len(filtered_candidates)} 对候选匹配")
    except Exception as e:
        print(f"LLM匹配失败: {e}")
        return
    
    # 结果处理
    print("\n6. 结果处理...")
    result_processor = ResultProcessor(
        confidence_threshold=config["thresholds"]["confidence_threshold"]
    )
    
    final_results = result_processor.process_matching_results(
        matching_results,
        processed_source_schemas,
        processed_target_schemas
    )
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_files = result_processor.save_results(final_results, output_dir)
    
    print(f"结果处理完成，找到 {len(final_results)} 对匹配字段")
    print(f"结果已保存至: {output_files['excel']}")
    
    # 输出匹配结果示例
    print("\n=== 匹配结果示例 ===")
    for i, result in enumerate(final_results[:5]):
        print(f"{i+1}. 源表: {result['source_table']}, 源字段: {result['source_field']}")
        print(f"   目标表: {result['target_table']}, 目标字段: {result['target_field']}")
        print(f"   匹配置信度: {result['confidence']:.2f}")
        print(f"   匹配依据: {result.get('matching_basis', '')}")
        print()
    
    if len(final_results) > 5:
        print(f"... 共 {len(final_results)} 条匹配结果，请查看完整输出文件")

if __name__ == "__main__":
    main()