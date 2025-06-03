"""
增强版相似度计算模块 - 提升匹配召回率
"""
import difflib
import numpy as np
from typing import Dict, List, Tuple, Any
import re

from Levenshtein import ratio as levenshtein_ratio
from ..utils.pinyin_utils import pinyin_similarity


class EnhancedSimilarityCalculator:
    """增强版相似度计算类"""
    
    def __init__(self, 
                 char_weight: float = 0.3, 
                 semantic_weight: float = 0.6, 
                 struct_weight: float = 0.1,
                 pinyin_boost: float = 1.5):
        """
        初始化增强版相似度计算器
        
        Args:
            char_weight: 字符相似度权重
            semantic_weight: 语义相似度权重
            struct_weight: 结构相似度权重
            pinyin_boost: 拼音匹配加权系数
        """
        self.char_weight = char_weight
        self.semantic_weight = semantic_weight
        self.struct_weight = struct_weight
        self.pinyin_boost = pinyin_boost
        
        # 验证权重总和
        total_weight = char_weight + semantic_weight + struct_weight
        if not (0.99 <= total_weight <= 1.01):
            print(f"警告: 权重总和为{total_weight}，将自动标准化")
            self.char_weight = char_weight / total_weight
            self.semantic_weight = semantic_weight / total_weight
            self.struct_weight = struct_weight / total_weight
        
        # 初始化同义词词典
        self.synonyms_dict = self._build_synonyms_dict()
        
        # 初始化常见缩写词典
        self.abbreviations_dict = self._build_abbreviations_dict()
    
    def calculate_similarity(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算两个字段的增强相似度
        
        Args:
            source_field: 源字段元数据
            target_field: 目标字段元数据
        
        Returns:
            相似度得分 (0-1)
        """
        # 字符级相似度
        char_sim = self.calculate_enhanced_char_similarity(source_field, target_field)
        
        # 语义相似度
        semantic_sim = self.calculate_enhanced_semantic_similarity(source_field, target_field)
        
        # 结构相似度
        struct_sim = self.calculate_enhanced_structural_similarity(source_field, target_field)
        
        # 基础加权融合
        base_sim = (
            self.char_weight * char_sim + 
            self.semantic_weight * semantic_sim + 
            self.struct_weight * struct_sim
        )
        
        # 拼音匹配加权
        pinyin_bonus = self._calculate_pinyin_bonus(source_field, target_field)
        
        # 同义词匹配加权
        synonym_bonus = self._calculate_synonym_bonus(source_field, target_field)
        
        # 最终相似度（确保不超过1.0）
        final_sim = min(1.0, base_sim + pinyin_bonus + synonym_bonus)
        
        return final_sim
    
    def calculate_enhanced_char_similarity(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算增强字符级相似度
        """
        s_name = source_field.get("name", "").lower()
        t_name = target_field.get("name", "").lower()
        
        if not s_name or not t_name:
            return 0.0
        
        # 1. 完全匹配加权
        if s_name == t_name:
            return 1.0
        
        # 2. Levenshtein距离
        lev_sim = levenshtein_ratio(s_name, t_name)
        
        # 3. 序列匹配相似度
        seq_sim = difflib.SequenceMatcher(None, s_name, t_name).ratio()
        
        # 4. Jaccard系数（基于token）
        s_tokens = set(source_field.get("tokenized_name", []))
        t_tokens = set(target_field.get("tokenized_name", []))
        
        if s_tokens and t_tokens:
            jaccard_sim = len(s_tokens.intersection(t_tokens)) / len(s_tokens.union(t_tokens))
        else:
            jaccard_sim = 0
        
        # 5. 包含关系检查
        contain_sim = 0
        if s_name in t_name or t_name in s_name:
            contain_sim = 0.8
        
        # 6. 前缀/后缀匹配
        prefix_sim = self._calculate_prefix_suffix_similarity(s_name, t_name)
        
        # 组合各种相似度
        weights = [0.25, 0.2, 0.2, 0.15, 0.1, 0.1]
        similarities = [lev_sim, seq_sim, jaccard_sim, contain_sim, prefix_sim, 0]
        
        char_sim = sum(w * s for w, s in zip(weights, similarities))
        
        return char_sim
    
    def calculate_enhanced_semantic_similarity(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算增强语义相似度
        """
        # 获取字段注释/描述
        s_desc = source_field.get("desc", "")
        t_desc = target_field.get("desc", "")
        
        semantic_scores = []
        
        # 1. 描述文本相似度
        if s_desc and t_desc:
            desc_sim = difflib.SequenceMatcher(None, s_desc.lower(), t_desc.lower()).ratio()
            semantic_scores.append(("desc_similarity", desc_sim, 0.3))
        
        # 2. 字段名与对方描述的包含关系
        name_desc_sim = self._calculate_name_desc_relationship(source_field, target_field)
        semantic_scores.append(("name_desc_relation", name_desc_sim, 0.25))
        
        # 3. 语义关键词匹配（增强版）
        keyword_sim = self._calculate_enhanced_keyword_similarity(source_field, target_field)
        semantic_scores.append(("keyword_match", keyword_sim, 0.25))
        
        # 4. 业务概念匹配
        concept_sim = self._calculate_business_concept_similarity(source_field, target_field)
        semantic_scores.append(("concept_match", concept_sim, 0.2))
        
        # 加权计算语义相似度
        total_weight = sum(weight for _, _, weight in semantic_scores)
        if total_weight > 0:
            semantic_sim = sum(score * weight for _, score, weight in semantic_scores) / total_weight
        else:
            semantic_sim = 0
        
        return semantic_sim
    
    def calculate_enhanced_structural_similarity(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算增强结构相似度
        """
        # 字段类型相似度（增强版）
        type_sim = self._calculate_enhanced_type_similarity(source_field, target_field)
        
        # 字段命名模式相似度
        pattern_sim = self._calculate_enhanced_pattern_similarity(source_field, target_field)
        
        # 字段位置相似度（如果有位置信息）
        position_sim = self._calculate_position_similarity(source_field, target_field)
        
        # 组合结构相似度
        weights = [0.5, 0.3, 0.2]
        similarities = [type_sim, pattern_sim, position_sim]
        
        struct_sim = sum(w * s for w, s in zip(weights, similarities))
        
        return struct_sim
    
    def _calculate_pinyin_bonus(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算拼音匹配加权分数
        """
        s_name = source_field.get("name", "")
        t_name = target_field.get("name", "")
        s_desc = source_field.get("desc", "")
        t_desc = target_field.get("desc", "")
        
        pinyin_scores = []
        
        # 1. 中文描述 -> 英文字段名
        if s_desc and t_name:
            pinyin_sim1 = pinyin_similarity(s_desc, t_name)
            pinyin_scores.append(pinyin_sim1)
        
        if t_desc and s_name:
            pinyin_sim2 = pinyin_similarity(t_desc, s_name)
            pinyin_scores.append(pinyin_sim2)
        
        # 2. 检查拼音缩写匹配
        pinyin_abbr_sim = self._check_pinyin_abbreviation_match(source_field, target_field)
        pinyin_scores.append(pinyin_abbr_sim)
        
        # 计算拼音加权分数
        if pinyin_scores:
            max_pinyin_score = max(pinyin_scores)
            # 只有当拼音相似度较高时才给加权
            if max_pinyin_score > 0.7:
                bonus = (max_pinyin_score - 0.7) * 0.3 * self.pinyin_boost
                return min(0.3, bonus)  # 最大加权0.3
        
        return 0.0
    
    def _calculate_synonym_bonus(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算同义词匹配加权分数
        """
        s_name = source_field.get("name", "").lower()
        t_name = target_field.get("name", "").lower()
        s_desc = source_field.get("desc", "").lower()
        t_desc = target_field.get("desc", "").lower()
        
        # 检查字段名的同义词匹配
        name_synonym_score = self._check_synonym_match(s_name, t_name)
        
        # 检查描述的同义词匹配
        desc_synonym_score = 0
        if s_desc and t_desc:
            desc_synonym_score = self._check_synonym_match(s_desc, t_desc)
        
        # 计算同义词加权分数
        max_synonym_score = max(name_synonym_score, desc_synonym_score)
        if max_synonym_score > 0.5:
            bonus = (max_synonym_score - 0.5) * 0.2
            return min(0.2, bonus)  # 最大加权0.2
        
        return 0.0
    
    def _build_synonyms_dict(self) -> Dict[str, List[str]]:
        """
        构建同义词词典
        """
        return {
            "id": ["编号", "号码", "代码", "标识", "key", "code", "identifier", "主键"],
            "name": ["名称", "名字", "姓名", "title", "label"],
            "code": ["代码", "编码", "编号", "代号", "id"],
            "type": ["类型", "种类", "分类", "category", "kind"],
            "status": ["状态", "状况", "情况", "condition", "state"],
            "time": ["时间", "日期", "date", "日", "年月日", "datetime", "timestamp"],
            "date": ["日期", "时间", "time", "日", "年月日", "datetime"],
            "desc": ["描述", "说明", "备注", "注释", "注解", "description", "remark"],
            "amount": ["金额", "数量", "数目", "金钱", "价格", "价值", "money", "price"],
            "address": ["地址", "住址", "位置", "地点", "location"],
            "phone": ["电话", "手机", "联系方式", "电话号码", "联系电话", "mobile"],
            "user": ["用户", "使用者", "人员", "person", "people"],
            "student": ["学生", "学员", "pupil", "learner"],
            "teacher": ["老师", "教师", "instructor", "educator"],
            "course": ["课程", "科目", "subject", "lesson"],
            "grade": ["成绩", "分数", "score", "mark"],
            "class": ["班级", "班", "class", "grade"],
            "school": ["学校", "院校", "institution"],
            "department": ["部门", "科室", "dept", "division"],
            "create": ["创建", "新建", "添加", "insert", "add"],
            "update": ["更新", "修改", "编辑", "modify", "edit"],
            "delete": ["删除", "移除", "remove"],
            "login": ["登录", "登入", "signin"],
            "logout": ["登出", "退出", "signout"]
        }
    
    def _build_abbreviations_dict(self) -> Dict[str, List[str]]:
        """
        构建缩写词典
        """
        return {
            "id": ["identifier", "identification"],
            "num": ["number"],
            "addr": ["address"],
            "tel": ["telephone"],
            "dept": ["department"],
            "mgr": ["manager"],
            "admin": ["administrator"],
            "info": ["information"],
            "desc": ["description"],
            "std": ["student"],
            "tch": ["teacher"],
            "cls": ["class"],
            "crs": ["course"],
            "sch": ["school"]
        }
    
    def _calculate_name_desc_relationship(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算字段名与描述的关系匹配度
        """
        s_name = source_field.get("name", "").lower()
        t_name = target_field.get("name", "").lower()
        s_desc = source_field.get("desc", "").lower()
        t_desc = target_field.get("desc", "").lower()
        
        scores = []
        
        # 源字段名包含在目标字段描述中
        if s_name and t_desc and s_name in t_desc:
            scores.append(0.8)
        
        # 目标字段名包含在源字段描述中
        if t_name and s_desc and t_name in s_desc:
            scores.append(0.8)
        
        # 字段名的token在对方描述中
        s_tokens = source_field.get("tokenized_name", [])
        t_tokens = target_field.get("tokenized_name", [])
        
        for token in s_tokens:
            if token.lower() in t_desc:
                scores.append(0.6)
        
        for token in t_tokens:
            if token.lower() in s_desc:
                scores.append(0.6)
        
        return max(scores) if scores else 0.0
    
    def _calculate_enhanced_keyword_similarity(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算增强的关键词相似度
        """
        s_name = source_field.get("name", "").lower()
        t_name = target_field.get("name", "").lower()
        
        max_similarity = 0
        
        for keyword, equiv_words in self.synonyms_dict.items():
            s_match = any(word in s_name for word in [keyword] + equiv_words)
            t_match = any(word in t_name for word in [keyword] + equiv_words)
            
            if s_match and t_match:
                # 计算具体匹配程度
                s_scores = [1.0 if word in s_name else 0.0 for word in [keyword] + equiv_words]
                t_scores = [1.0 if word in t_name else 0.0 for word in [keyword] + equiv_words]
                
                match_score = max(max(s_scores), max(t_scores))
                max_similarity = max(max_similarity, match_score)
        
        return max_similarity
    
    def _calculate_business_concept_similarity(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算业务概念相似度
        """
        # 业务概念映射
        business_concepts = {
            "person_info": ["姓名", "name", "人员", "用户", "学生", "老师", "员工"],
            "identifier": ["编号", "id", "code", "标识", "主键", "key"],
            "temporal": ["时间", "日期", "time", "date", "创建", "更新"],
            "contact": ["电话", "手机", "邮箱", "地址", "联系"],
            "academic": ["成绩", "分数", "课程", "班级", "学校", "专业"],
            "status": ["状态", "状况", "类型", "种类", "标志"],
            "description": ["描述", "说明", "备注", "注释", "详情"]
        }
        
        s_text = (source_field.get("name", "") + " " + source_field.get("desc", "")).lower()
        t_text = (target_field.get("name", "") + " " + target_field.get("desc", "")).lower()
        
        for concept, keywords in business_concepts.items():
            s_match = any(keyword in s_text for keyword in keywords)
            t_match = any(keyword in t_text for keyword in keywords)
            
            if s_match and t_match:
                return 0.8
        
        return 0.0
    
    def _calculate_enhanced_type_similarity(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算增强的类型相似度
        """
        s_type = source_field.get("type", "").lower()
        t_type = target_field.get("type", "").lower()
        
        if not s_type or not t_type:
            return 0.3  # 给予一定的默认分数
        
        # 增强的类型映射表
        type_groups = {
            "integer": ["int", "integer", "bigint", "smallint", "tinyint", "number", "numeric", "整数", "数字"],
            "float": ["double", "decimal", "numeric", "real", "float", "money", "浮点", "小数", "金额"],
            "string": ["varchar", "char", "character", "nvarchar", "text", "string", "字符", "字符串", "文本"],
            "datetime": ["datetime", "timestamp", "time", "date", "日期", "时间"],
            "boolean": ["boolean", "bit", "bool", "布尔", "是否"],
            "binary": ["blob", "binary", "image", "二进制"]
        }
        
        # 检查类型是否属于同一组
        for group_name, type_list in type_groups.items():
            s_in_group = any(t in s_type for t in type_list)
            t_in_group = any(t in t_type for t in type_list)
            
            if s_in_group and t_in_group:
                return 1.0
        
        # 字符串相似度作为后备
        return difflib.SequenceMatcher(None, s_type, t_type).ratio()
    
    def _calculate_enhanced_pattern_similarity(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算增强的命名模式相似度
        """
        s_name = source_field.get("name", "").lower()
        t_name = target_field.get("name", "").lower()
        
        # 增强的命名模式
        patterns = {
            r'(id|_id|编号)$': 0.9,  # ID字段
            r'^(code|编码|代码)': 0.8,  # 代码字段
            r'^(name|名称|姓名)': 0.8,  # 名称字段
            r'(type|类型|种类)': 0.7,  # 类型字段
            r'(date|time|时间|日期)': 0.7,  # 日期时间字段
            r'(desc|description|说明|描述)': 0.6,  # 描述字段
            r'(status|状态|状况)': 0.7,  # 状态字段
            r'(create|add|新增|创建)': 0.6,  # 创建字段
            r'(update|modify|更新|修改)': 0.6,  # 更新字段
            r'(num|number|数量|序号)': 0.6,  # 数字字段
        }
        
        max_score = 0
        for pattern, score in patterns.items():
            if re.search(pattern, s_name) and re.search(pattern, t_name):
                max_score = max(max_score, score)
        
        return max_score
    
    def _calculate_position_similarity(self, source_field: Dict, target_field: Dict) -> float:
        """
        计算字段位置相似度（如果有位置信息）
        """
        # 这里可以根据字段在表中的位置计算相似度
        # 目前返回默认值
        return 0.5
    
    def _calculate_prefix_suffix_similarity(self, s_name: str, t_name: str) -> float:
        """
        计算前缀后缀相似度
        """
        if len(s_name) < 2 or len(t_name) < 2:
            return 0
        
        scores = []
        
        # 前缀匹配
        for i in range(2, min(len(s_name), len(t_name)) + 1):
            if s_name[:i] == t_name[:i]:
                scores.append(i / max(len(s_name), len(t_name)))
        
        # 后缀匹配
        for i in range(2, min(len(s_name), len(t_name)) + 1):
            if s_name[-i:] == t_name[-i:]:
                scores.append(i / max(len(s_name), len(t_name)))
        
        return max(scores) if scores else 0
    
    def _check_pinyin_abbreviation_match(self, source_field: Dict, target_field: Dict) -> float:
        """
        检查拼音缩写匹配
        """
        # 获取拼音缩写信息
        s_name = source_field.get("name", "")
        t_name = target_field.get("name", "")
        s_desc = source_field.get("desc", "")
        t_desc = target_field.get("desc", "")
        
        # 检查是否有拼音缩写字段
        s_pinyin_abbr = source_field.get("name_pinyin_abbr", "")
        t_pinyin_abbr = target_field.get("name_pinyin_abbr", "")
        s_desc_pinyin_abbr = source_field.get("desc_pinyin_abbr", "")
        t_desc_pinyin_abbr = target_field.get("desc_pinyin_abbr", "")
        
        scores = []
        
        # 检查各种拼音缩写组合
        if s_pinyin_abbr and t_name and s_pinyin_abbr.lower() == t_name.lower():
            scores.append(0.9)
        
        if t_pinyin_abbr and s_name and t_pinyin_abbr.lower() == s_name.lower():
            scores.append(0.9)
        
        if s_desc_pinyin_abbr and t_name and s_desc_pinyin_abbr.lower() == t_name.lower():
            scores.append(0.95)
        
        if t_desc_pinyin_abbr and s_name and t_desc_pinyin_abbr.lower() == s_name.lower():
            scores.append(0.95)
        
        return max(scores) if scores else 0.0
    
    def _check_synonym_match(self, text1: str, text2: str) -> float:
        """
        检查同义词匹配
        """
        words1 = text1.split()
        words2 = text2.split()
        
        for word1 in words1:
            for word2 in words2:
                # 检查是否为同义词
                for key, synonyms in self.synonyms_dict.items():
                    if (word1 in synonyms or word1 == key) and (word2 in synonyms or word2 == key):
                        return 0.8
                
                # 检查缩写匹配
                if word1 in self.abbreviations_dict and word2 in self.abbreviations_dict[word1]:
                    return 0.7
                if word2 in self.abbreviations_dict and word1 in self.abbreviations_dict[word2]:
                    return 0.7
        
        return 0.0
    
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