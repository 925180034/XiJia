"""
增强版LLM语义匹配模块 - 提升匹配召回率
"""
import os
import time
import json
import re
import requests
from typing import Dict, List, Tuple, Any, Optional
import yaml
from tqdm import tqdm
from src.matching.robust_llm_matcher import RobustLLMMatcher as EnhancedLLMMatcher

class EnhancedLLMMatcher:
    """增强版LLM语义匹配类"""
    
    def __init__(self, config_path: str = "config/config_enhanced.yaml"):
        """
        初始化增强版LLM匹配器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 配置OpenAI
        self.api_key = config["openai"]["api_key"]
        self.api_base_url = config["openai"].get("api_base_url")
        self.model = config["openai"]["model"]
        self.temperature = config["openai"]["temperature"]
        self.max_tokens = config["openai"]["max_tokens"]
        
        # 批处理配置
        self.batch_size = config["system"]["batch_size"]
        self.cache_enabled = config["system"]["cache_enabled"]
        
        # 多层次匹配配置
        self.enable_aggressive_mode = config["matching_strategy"]["enable_aggressive_mode"]
        self.include_potential_matches = config["matching_strategy"]["include_potential_matches"]
        
        # 缓存
        self._cache = {}
        self._cache_path = "cache/enhanced_llm_responses.json"
        if self.cache_enabled:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            if os.path.exists(self._cache_path):
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
    
    def match_field_pair(self, 
                        source_schema: Dict, 
                        source_field: Dict, 
                        target_schema: Dict, 
                        target_field: Dict,
                        similarity: float) -> Dict:
        """
        匹配单对字段（增强版）
        
        Args:
            source_schema: 源表元数据
            source_field: 源字段元数据
            target_schema: 目标表元数据
            target_field: 目标字段元数据
            similarity: 计算的相似度得分
            
        Returns:
            匹配结果，包含是否匹配、置信度和理由
        """
        # 构建增强提示模板
        prompt = self._create_enhanced_prompt(source_schema, source_field, target_schema, target_field, similarity)
        
        # 缓存Key
        cache_key = f"enhanced_{source_schema['table_name']}_{source_field['name']}_{target_schema['table_name']}_{target_field['name']}"
        
        # 检查缓存
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 调用LLM
        try:
            response = self._call_llm(prompt)
            
            # 解析响应（增强版）
            result = self._parse_enhanced_response(response, similarity)
            
            # 添加相似度信息
            result["similarity"] = similarity
            
            # 缓存结果
            if self.cache_enabled:
                self._cache[cache_key] = result
                # 定期保存缓存
                if len(self._cache) % 10 == 0:
                    with open(self._cache_path, "w", encoding="utf-8") as f:
                        json.dump(self._cache, f, ensure_ascii=False, indent=2)
            
            return result
            
        except Exception as e:
            print(f"LLM调用失败: {e}")
            # 增强的默认结果：基于相似度给出判断
            return self._generate_fallback_result(source_field, target_field, similarity, str(e))
    
    def batch_process_candidates(self, 
                                candidate_pairs: List[Dict],
                                source_schemas: Dict[str, Dict],
                                target_schemas: Dict[str, Dict]) -> List[Dict]:
        """
        批量处理候选匹配对（增强版）
        
        Args:
            candidate_pairs: 候选匹配对列表
            source_schemas: 源表元数据字典
            target_schemas: 目标表元数据字典
            
        Returns:
            处理结果列表
        """
        results = []
        
        # 按相似度分层处理
        high_sim_candidates = [c for c in candidate_pairs if c["similarity"] >= 0.7]
        medium_sim_candidates = [c for c in candidate_pairs if 0.4 <= c["similarity"] < 0.7]
        low_sim_candidates = [c for c in candidate_pairs if c["similarity"] < 0.4]
        
        print(f"高相似度候选对: {len(high_sim_candidates)}")
        print(f"中等相似度候选对: {len(medium_sim_candidates)}")
        print(f"低相似度候选对: {len(low_sim_candidates)}")
        
        # 优先处理高相似度候选对
        for candidates, desc in [(high_sim_candidates, "高相似度"), 
                                (medium_sim_candidates, "中等相似度"),
                                (low_sim_candidates, "低相似度")]:
            if candidates:
                print(f"处理{desc}候选对...")
                batch_results = self._process_candidate_batch(
                    candidates, source_schemas, target_schemas, desc
                )
                results.extend(batch_results)
        
        # 保存缓存
        if self.cache_enabled:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        
        return results
    
    def _process_candidate_batch(self, 
                                candidates: List[Dict],
                                source_schemas: Dict[str, Dict],
                                target_schemas: Dict[str, Dict],
                                batch_desc: str) -> List[Dict]:
        """
        处理一批候选对
        """
        results = []
        
        # 分批处理
        for i in tqdm(range(0, len(candidates), self.batch_size), desc=f"批量处理{batch_desc}候选匹配对"):
            batch = candidates[i:i+self.batch_size]
            
            batch_results = []
            for candidate in batch:
                source_schema = source_schemas[candidate["source_table"]]
                source_field = next(f for f in source_schema["fields"] if f["name"] == candidate["source_field"])
                
                target_schema = target_schemas[candidate["target_table"]]
                target_field = next(f for f in target_schema["fields"] if f["name"] == candidate["target_field"])
                
                result = self.match_field_pair(
                    source_schema, 
                    source_field, 
                    target_schema, 
                    target_field,
                    candidate["similarity"]
                )
                
                # 添加源表和目标表信息
                result["source_table"] = candidate["source_table"]
                result["source_field"] = candidate["source_field"]
                result["target_table"] = candidate["target_table"]
                result["target_field"] = candidate["target_field"]
                
                batch_results.append(result)
            
            results.extend(batch_results)
            
            # 防止API限制
            time.sleep(0.3)
        
        return results
    
    def _create_enhanced_prompt(self, 
                               source_schema: Dict, 
                               source_field: Dict, 
                               target_schema: Dict, 
                               target_field: Dict,
                               similarity: float) -> str:
        """
        创建增强版提示模板（更鼓励匹配）
        
        Args:
            source_schema: 源表元数据
            source_field: 源字段元数据
            target_schema: 目标表元数据
            target_field: 目标字段元数据
            similarity: 计算的相似度得分
            
        Returns:
            增强提示模板
        """
        # 根据相似度选择不同的提示策略
        if similarity >= 0.7:
            prompt_style = "high_similarity"
        elif similarity >= 0.4:
            prompt_style = "medium_similarity"
        else:
            prompt_style = "low_similarity"
        
        if prompt_style == "high_similarity":
            prompt = f"""系统角色：您是数据集成专家，擅长识别字段间的语义等价关系。

任务描述：判断两个字段是否语义等价。这两个字段已经通过初步筛选，具有较高的相似度({similarity:.2f})，请仔细分析它们是否表示相同的业务概念。

源字段：
- 表名：{source_schema['table_name']}
- 表描述：{source_schema.get('table_desc', '(无)')}
- 字段名：{source_field['name']}
- 字段描述：{source_field.get('desc', '(无)')}
- 字段类型：{source_field.get('type', '(未知)')}

目标字段：
- 表名：{target_schema['table_name']}
- 表描述：{target_schema.get('table_desc', '(无)')}
- 字段名：{target_field['name']}
- 字段描述：{target_field.get('desc', '(无)')}
- 字段类型：{target_field.get('type', '(未知)')}

分析要点：
1. 字段名语义关系：考虑中英文对应、拼音缩写（如"XSBH"="学生编号"对应"XH"="学号"）
2. 业务概念匹配：分析字段在业务流程中的作用是否相同
3. 描述语义：比较字段描述的语义相似度
4. 数据类型兼容性：检查数据类型是否兼容

常见等价关系示例：
- ID/编号/代码/标识 → 通常表示唯一标识符
- 姓名/名称/名字 → 通常表示人或物的名称
- 时间/日期/创建时间/更新时间 → 通常表示时间概念
- 状态/类型/种类 → 通常表示分类信息

判断标准：
- 如果两个字段表示相同的业务概念，即使名称不完全相同，也应该判断为匹配
- 中英文字段如果语义相同，应判断为匹配
- 拼音缩写与原词如果对应，应判断为匹配

请给出判断和置信度：
格式：
判断：[是/否]
置信度：[0-1之间的数字]
理由：[详细说明]"""

        elif prompt_style == "medium_similarity":
            prompt = f"""系统角色：您是数据集成专家，擅长在复杂情况下识别字段间的潜在等价关系。

任务描述：判断两个字段是否可能语义等价。这两个字段具有中等相似度({similarity:.2f})，需要深入分析业务语义。

源字段：
- 表名：{source_schema['table_name']}
- 表描述：{source_schema.get('table_desc', '(无)')}
- 字段名：{source_field['name']}
- 字段描述：{source_field.get('desc', '(无)')}
- 字段类型：{source_field.get('type', '(未知)')}

目标字段：
- 表名：{target_schema['table_name']}
- 表描述：{target_schema.get('table_desc', '(无)')}
- 字段名：{target_field['name']}
- 字段描述：{target_field.get('desc', '(无)')}
- 字段类型：{target_field.get('type', '(未知)')}

深度分析要点：
1. 语义挖掘：即使名称不同，分析是否表示相同业务概念
2. 上下文理解：结合表名和表描述理解字段的业务作用
3. 模糊匹配：考虑同义词、近义词、缩写形式
4. 领域知识：运用教育、管理等领域的常识判断

匹配倾向：
- 优先考虑业务语义相似性
- 对于可能的匹配，给予积极的判断
- 即使不确定，如果有合理的匹配可能性，倾向于匹配

请给出判断和置信度：
格式：
判断：[是/否]
置信度：[0-1之间的数字]
理由：[详细说明]"""

        else:  # low_similarity
            prompt = f"""系统角色：您是数据集成专家，擅长发现隐含的字段等价关系。

任务描述：虽然这两个字段的表面相似度较低({similarity:.2f})，但请深入分析是否存在隐含的语义等价关系。

源字段：
- 表名：{source_schema['table_name']}
- 表描述：{source_schema.get('table_desc', '(无)')}
- 字段名：{source_field['name']}
- 字段描述：{source_field.get('desc', '(无)')}
- 字段类型：{source_field.get('type', '(未知)')}

目标字段：
- 表名：{target_schema['table_name']}
- 表描述：{target_schema.get('table_desc', '(无)')}
- 字段名：{target_field['name']}
- 字段描述：{target_field.get('desc', '(无)')}
- 字段类型：{target_field.get('type', '(未知)')}

隐含关系发现：
1. 业务本质：抛开表面形式，分析字段的业务本质
2. 功能等价：即使名称完全不同，如果功能相同也可能匹配
3. 数据流向：考虑数据在业务流程中的作用
4. 潜在对应：寻找可能的对应关系

探索性判断：
- 即使表面相似度低，如果发现合理的业务关联，给予匹配判断
- 注重业务逻辑而非字面意思
- 对潜在匹配保持开放态度

请给出判断和置信度：
格式：
判断：[是/否]
置信度：[0-1之间的数字]  
理由：[详细说明]"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        调用LLM
        
        Args:
            prompt: 提示模板
            
        Returns:
            LLM响应
        """
        try:
            api_url = f"{self.api_base_url}/chat/completions" if self.api_base_url else "https://api.openai.com/v1/chat/completions"
            
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
            
            response = requests.post(api_url, headers=headers, json=data, timeout=60)
            
            if response.status_code != 200:
                error_msg = f"API调用失败，状态码：{response.status_code}，响应：{response.text}"
                return f"错误：{error_msg}"
            
            resp_json = response.json()
            
            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                if "message" in resp_json["choices"][0] and "content" in resp_json["choices"][0]["message"]:
                    return resp_json["choices"][0]["message"]["content"]
                elif "text" in resp_json["choices"][0]:
                    return resp_json["choices"][0]["text"]
            
            return str(resp_json)
            
        except Exception as e:
            error_msg = f"API调用异常: {str(e)}"
            return f"错误：{error_msg}"
    
    def _parse_enhanced_response(self, response: str, similarity: float) -> Dict:
        """
        解析LLM响应（增强版）
        
        Args:
            response: LLM响应
            similarity: 原始相似度
            
        Returns:
            解析结果
        """
        try:
            if response.startswith("错误："):
                return self._generate_fallback_result_from_similarity(similarity, response)
            
            # 提取判断结果（更宽松的匹配）
            match = False
            response_lower = response.lower()
            
            # 多种判断模式
            positive_indicators = ["判断：是", "判断:是", "判断: 是", "结论：是", "结论:是", "结论: 是"]
            negative_indicators = ["判断：否", "判断:否", "判断: 否", "结论：否", "结论:否", "结论: 否"]
            
            for indicator in positive_indicators:
                if indicator in response:
                    match = True
                    break
            
            # 如果没有明确的否定，检查是否有积极的表述
            if not match and not any(indicator in response for indicator in negative_indicators):
                # 检查积极的关键词
                positive_keywords = ["匹配", "等价", "相同", "一致", "对应", "相似", "可以匹配"]
                if any(keyword in response for keyword in positive_keywords):
                    match = True
            
            # 提取置信度（增强版）
            confidence = self._extract_confidence(response, match, similarity)
            
            # 提取理由
            reason = self._extract_reason(response)
            
            # 如果LLM倾向于匹配但置信度很低，适当提升置信度
            if match and confidence < 0.5:
                confidence = max(0.6, confidence + 0.2)
            
            return {
                "match": match,
                "confidence": confidence,
                "reason": reason,
                "llm_response": response
            }
            
        except Exception as e:
            return self._generate_fallback_result_from_similarity(similarity, f"解析响应失败: {e}")
    
    def _extract_confidence(self, response: str, match: bool, similarity: float) -> float:
        """
        提取置信度（增强版）
        """
        confidence = 0.0
        
        # 尝试从响应中提取置信度
        confidence_patterns = [
            r"置信度：([0-9.]+)",
            r"置信度:([0-9.]+)", 
            r"置信度: ([0-9.]+)",
            r"confidence:([0-9.]+)",
            r"confidence：([0-9.]+)",
            r"confidence: ([0-9.]+)"
        ]
        
        for pattern in confidence_patterns:
            match_obj = re.search(pattern, response)
            if match_obj:
                try:
                    confidence = float(match_obj.group(1))
                    break
                except ValueError:
                    pass
        
        # 如果没有提取到置信度，基于相似度和匹配结果推算
        if confidence == 0.0:
            if match:
                # 匹配的情况下，基于相似度给予较高置信度
                confidence = max(0.7, similarity + 0.2)
            else:
                # 不匹配的情况下，给予较低置信度
                confidence = max(0.3, similarity)
        
        # 确保置信度在合理范围内
        confidence = max(0.1, min(1.0, confidence))
        
        return confidence
    
    def _extract_reason(self, response: str) -> str:
        """
        提取理由
        """
        reason_patterns = [
            r"理由：(.*?)(?=\n|$)",
            r"理由:(.*?)(?=\n|$)",
            r"理由: (.*?)(?=\n|$)",
            r"原因：(.*?)(?=\n|$)",
            r"原因:(.*?)(?=\n|$)",
            r"原因: (.*?)(?=\n|$)",
            r"说明：(.*?)(?=\n|$)",
            r"说明:(.*?)(?=\n|$)",
            r"说明: (.*?)(?=\n|$)"
        ]
        
        for pattern in reason_patterns:
            match_obj = re.search(pattern, response, re.DOTALL)
            if match_obj:
                reason = match_obj.group(1).strip()
                if reason:
                    return reason[:200]  # 限制长度
        
        # 如果没有找到特定的理由模式，返回整个响应的摘要
        lines = response.split('\n')
        reason_lines = [line.strip() for line in lines if line.strip() and not line.startswith('判断：') and not line.startswith('置信度：')]
        if reason_lines:
            return ' '.join(reason_lines)[:200]
        
        return response[:200]
    
    def _generate_fallback_result(self, source_field: Dict, target_field: Dict, similarity: float, error_msg: str) -> Dict:
        """
        生成增强的回退结果
        """
        # 基于相似度和字段特征做判断
        match = False
        confidence = similarity
        
        s_name = source_field.get("name", "").lower()
        t_name = target_field.get("name", "").lower()
        
        # 检查明显的匹配情况
        if s_name == t_name:
            match = True
            confidence = 0.95
        elif similarity > 0.8:
            match = True
            confidence = max(0.7, similarity)
        elif similarity > 0.6:
            # 检查是否有明显的业务关联
            if any(keyword in s_name for keyword in ["id", "code", "name", "time", "date"]) and \
               any(keyword in t_name for keyword in ["id", "code", "name", "time", "date"]):
                match = True
                confidence = max(0.6, similarity)
        
        reason = f"基于相似度{similarity:.2f}和字段特征的自动判断。LLM调用失败：{error_msg}"
        
        return {
            "match": match,
            "confidence": confidence,
            "reason": reason,
            "fallback": True
        }
    
    def _generate_fallback_result_from_similarity(self, similarity: float, error_msg: str) -> Dict:
        """
        基于相似度生成回退结果
        """
        # 更积极的回退策略
        if similarity >= 0.7:
            match = True
            confidence = min(0.8, similarity + 0.1)
        elif similarity >= 0.5:
            match = True
            confidence = min(0.7, similarity + 0.1)
        elif similarity >= 0.3:
            match = True
            confidence = min(0.6, similarity + 0.2)
        else:
            match = False
            confidence = similarity
        
        reason = f"基于相似度{similarity:.2f}的积极匹配策略。{error_msg}"
        
        return {
            "match": match,
            "confidence": confidence,
            "reason": reason,
            "fallback": True
        }