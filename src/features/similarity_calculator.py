"""
相似度计算模块
"""
import difflib
import numpy as np
from typing import Dict, List, Tuple, Any

from Levenshtein import ratio as levenshtein_ratio
from ..utils.pinyin_utils import pinyin_similarity


class SimilarityCalculator:
    """相似度计算类"""
    
    def __init__(self, 
                 char_weight: float = 0.4, 
                 semantic_weight: float = 0.5, 
                 struct_weight: float = 0.1):
        """
        初始化相似度计算器
        
        Args:
            char_weight: 字符相似度权重
            semantic_weight: 语义相似度权重
            struct_weight: 结构相似度权重
        """
        self.char_weight = char_weight
        self.semantic_weight = semantic_weight
        self.struct_weight = struct_weight
        
        # 验证权重总和为1
        total_weight = char_weight + semantic_weight + struct_weight
        if not (0.99 <= total_weight <= 1.01):  # 允许浮点误差
            raise ValueError(f"权重总和必须为1.0，当前为{total_weight}")
    
    def calculate_similarity(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算两个字段的相似度
        
        Args:
            source_field: 源字段元数据
            target_field: 目标字段元数据
        
        Returns:
            相似度得分 (0-1)
        """
        # 字符级相似度
        char_sim = self.calculate_char_similarity(source_field, target_field)
        
        # 语义相似度
        semantic_sim = self.calculate_semantic_similarity(source_field, target_field)
        
        # 结构相似度
        struct_sim = self.calculate_structural_similarity(source_field, target_field)
        
        # 加权融合
        final_sim = (
            self.char_weight * char_sim + 
            self.semantic_weight * semantic_sim + 
            self.struct_weight * struct_sim
        )
        
        return final_sim
    
    def calculate_char_similarity(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算字符级相似度
        
        Args:
            source_field: 源字段元数据
            target_field: 目标字段元数据
        
        Returns:
            字符级相似度 (0-1)
        """
        s_name = source_field.get("name", "")
        t_name = target_field.get("name", "")
        
        # Levenshtein距离
        lev_sim = levenshtein_ratio(s_name.lower(), t_name.lower())
        
        # 序列匹配相似度
        seq_sim = difflib.SequenceMatcher(None, s_name.lower(), t_name.lower()).ratio()
        
        # Jaccard系数
        s_tokens = set(source_field.get("tokenized_name", []))
        t_tokens = set(target_field.get("tokenized_name", []))
        
        if s_tokens and t_tokens:
            jaccard_sim = len(s_tokens.intersection(t_tokens)) / len(s_tokens.union(t_tokens))
        else:
            jaccard_sim = 0
        
        # 拼音相似度 (如果字段包含中文)
        pinyin_sim = 0
        
        # 从中文字段描述到英文字段名的映射
        s_desc = source_field.get("desc", "")
        if s_desc:
            pinyin_sim = max(pinyin_sim, pinyin_similarity(s_desc, t_name))
        
        # 从英文字段名到中文字段描述的映射
        t_desc = target_field.get("desc", "")
        if t_desc:
            pinyin_sim = max(pinyin_sim, pinyin_similarity(t_desc, s_name))
        
        # 组合各种相似度
        alpha1, alpha2, alpha3, alpha4 = 0.3, 0.2, 0.2, 0.3  # 权重
        char_sim = alpha1 * lev_sim + alpha2 * seq_sim + alpha3 * jaccard_sim + alpha4 * pinyin_sim
        
        return char_sim
    
    def calculate_semantic_similarity(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算语义相似度
        
        Args:
            source_field: 源字段元数据
            target_field: 目标字段元数据
        
        Returns:
            语义相似度 (0-1)
        """
        # 获取字段注释/描述
        s_desc = source_field.get("desc", "")
        t_desc = target_field.get("desc", "")
        
        # 如果两个字段都有描述，计算描述间的相似度
        if s_desc and t_desc:
            desc_sim = difflib.SequenceMatcher(None, s_desc, t_desc).ratio()
        else:
            desc_sim = 0
        
        # 如果字段名包含在对方的描述中，增加相似度
        s_name = source_field.get("name", "").lower()
        t_name = target_field.get("name", "").lower()
        
        name_in_desc_sim = 0
        if s_name and t_desc and s_name in t_desc.lower():
            name_in_desc_sim += 0.5
        if t_name and s_desc and t_name in s_desc.lower():
            name_in_desc_sim += 0.5
        
        # 检查语义相等的关键词
        equiv_keywords = {
            "id": ["编号", "号码", "代码", "标识", "key"],
            "name": ["名称", "名字", "姓名"],
            "code": ["代码", "编码", "编号"],
            "type": ["类型", "种类", "分类"],
            "status": ["状态", "状况", "情况"],
            "time": ["时间", "日期", "date", "日", "年月日"],
            "date": ["日期", "时间", "time", "日", "年月日"],
            "desc": ["描述", "说明", "备注", "注释", "注解"],
            "amount": ["金额", "数量", "数目", "金钱", "价格", "价值"],
            "address": ["地址", "住址", "位置", "地点"],
            "phone": ["电话", "手机", "联系方式", "电话号码", "联系电话"]
        }
        
        keyword_sim = 0
        # 检查源字段是否匹配目标字段的等价关键词
        for keyword, equiv_words in equiv_keywords.items():
            if keyword in s_name or any(word in s_name for word in equiv_words):
                if keyword in t_name or any(word in t_name for word in equiv_words):
                    keyword_sim = 1.0
                    break
        
        # 组合各种相似度
        beta1, beta2, beta3 = 0.4, 0.3, 0.3  # 权重
        semantic_sim = beta1 * desc_sim + beta2 * name_in_desc_sim + beta3 * keyword_sim
        
        return semantic_sim
    
    def calculate_structural_similarity(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算结构相似度
        
        Args:
            source_field: 源字段元数据
            target_field: 目标字段元数据
        
        Returns:
            结构相似度 (0-1)
        """
        # 字段类型相似度
        type_sim = 0
        s_type = source_field.get("type", "").lower()
        t_type = target_field.get("type", "").lower()
        
        # 类型映射表，列出常见类型的等价关系
        type_mapping = {
            "int": ["integer", "int", "bigint", "smallint", "tinyint", "number", "numeric", "整数", "数字"],
            "float": ["double", "decimal", "numeric", "real", "float", "浮点", "小数"],
            "char": ["varchar", "character", "nvarchar", "text", "string", "字符", "字符串"],
            "date": ["datetime", "timestamp", "time", "日期", "时间"],
            "bool": ["boolean", "bit", "布尔", "是否"]
        }
        
        # 检查两个字段的类型是否属于同一类别
        for type_group, equiv_types in type_mapping.items():
            if s_type in equiv_types and t_type in equiv_types:
                type_sim = 1.0
                break
        
        # 字段位置相似度 (暂不考虑，因为位置信息不在输入元数据中)
        
        # 字段名模式相似度
        pattern_sim = 0
        
        # 检查常见的字段命名模式
        patterns = {
            r'id$|_id$': 0.8,  # ID字段
            r'^code|_code': 0.7,  # 代码字段
            r'^name|_name': 0.7,  # 名称字段
            r'^type|_type': 0.6,  # 类型字段
            r'^date|_date|time|_time': 0.6,  # 日期时间字段
            r'^desc|_desc|description|_description': 0.5,  # 描述字段
            r'^status|_status': 0.6,  # 状态字段
            r'^create|_create|^update|_update': 0.5,  # 创建更新字段
        }
        
        import re
        s_name = source_field.get("name", "").lower()
        t_name = target_field.get("name", "").lower()
        
        for pattern, score in patterns.items():
            if re.search(pattern, s_name) and re.search(pattern, t_name):
                pattern_sim = max(pattern_sim, score)
        
        # 组合各种相似度
        gamma1, gamma2 = 0.6, 0.4  # 权重
        struct_sim = gamma1 * type_sim + gamma2 * pattern_sim
        
        return struct_sim
    
    def calculate_similarity_matrix(self, 
                                    source_fields: List[Dict], 
                                    target_fields: List[Dict]) -> np.ndarray:
        """
        计算相似度矩阵
        
        Args:
            source_fields: 源字段列表
            target_fields: 目标字段列表
        
        Returns:
            相似度矩阵
        """
        # 初始化相似度矩阵
        sim_matrix = np.zeros((len(source_fields), len(target_fields)))
        
        # 计算每对字段间的相似度
        for i, s_field in enumerate(source_fields):
            for j, t_field in enumerate(target_fields):
                sim_matrix[i, j] = self.calculate_similarity(s_field, t_field)
        
        return sim_matrix