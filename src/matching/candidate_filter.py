"""
候选对筛选模块
"""
import numpy as np
from typing import Dict, List, Tuple, Any


class CandidateFilter:
    """候选匹配对筛选类"""
    
    def __init__(self, similarity_threshold: float = 0.5):
        """
        初始化候选对筛选器
        
        Args:
            similarity_threshold: 相似度阈值
        """
        self.similarity_threshold = similarity_threshold
    
    def filter_candidates(self, 
                        source_schemas: List[Dict],
                        target_schemas: List[Dict], 
                        similarity_matrices: Dict[Tuple[str, str], np.ndarray]) -> List[Dict]:
        """
        筛选候选匹配对
        
        Args:
            source_schemas: 源表元数据列表
            target_schemas: 目标表元数据列表
            similarity_matrices: 相似度矩阵字典，键为(源表名, 目标表名)
                                元素为形状为(len(source_fields), len(target_fields))的矩阵
            
        Returns:
            候选匹配对列表
        """
        candidates = []
        
        # 处理每对表
        for source_schema in source_schemas:
            source_table_name = source_schema["table_name"]
            source_fields = source_schema["fields"]
            
            for target_schema in target_schemas:
                target_table_name = target_schema["table_name"]
                target_fields = target_schema["fields"]
                
                # 获取相似度矩阵
                matrix_key = (source_table_name, target_table_name)
                if matrix_key not in similarity_matrices:
                    continue
                
                similarity_matrix = similarity_matrices[matrix_key]
                
                # 筛选候选对
                for i, source_field in enumerate(source_fields):
                    for j, target_field in enumerate(target_fields):
                        similarity = similarity_matrix[i, j]
                        
                        # 只保留相似度大于阈值的候选对
                        if similarity >= self.similarity_threshold:
                            candidates.append({
                                "source_table": source_table_name,
                                "source_field": source_field["name"],
                                "target_table": target_table_name,
                                "target_field": target_field["name"],
                                "similarity": float(similarity)
                            })
        
        # 按相似度降序排序
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        
        return candidates
    
    def apply_matching_rules(self, candidates: List[Dict]) -> List[Dict]:
        """
        应用匹配规则进一步筛选候选对
        
        Args:
            candidates: 初步筛选的候选匹配对列表
        
        Returns:
            应用规则后的候选匹配对列表
        """
        enhanced_candidates = []
        
        # 应用规则1: 字段名完全相同的优先级更高
        exact_name_matches = []
        for candidate in candidates:
            if candidate["source_field"].lower() == candidate["target_field"].lower():
                exact_name_matches.append(candidate)
        
        # 将相同名称的候选对添加到结果中
        enhanced_candidates.extend(exact_name_matches)
        
        # 应用规则2: 处理剩余的高相似度候选对，避免冲突
        remaining = [c for c in candidates if c not in exact_name_matches]
        
        # 已使用的源字段和目标字段
        used_source_fields = set(c["source_field"] for c in enhanced_candidates)
        used_target_fields = set(c["target_field"] for c in enhanced_candidates)
        
        # 按相似度降序遍历
        for candidate in remaining:
            source_field = candidate["source_field"]
            target_field = candidate["target_field"]
            
            # 检查冲突
            if source_field in used_source_fields or target_field in used_target_fields:
                continue
            
            # 添加到结果
            enhanced_candidates.append(candidate)
            used_source_fields.add(source_field)
            used_target_fields.add(target_field)
        
        return enhanced_candidates