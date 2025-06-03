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


class EnhancedLLMMatcher:
    """增强版LLM语义匹配类"""
    
    def __init__(self, config_path: str = "config/config_enhanced.yaml"):
        """
        初始化增强版LLM匹配器
        
        Args:
            config_path: 配置文件路径
        """
        print(f"加载增强配置文件: {config_path}")
        
        # 确保配置文件存在
        if not os.path.exists(config_path):
            print(f"配置文件不存在: {config_path}")
            # 尝试备选配置文件
            alternative_configs = [
                "config/config.yaml",
                "config/config_enhanced.yaml"
            ]
            for alt_config in alternative_configs:
                if os.path.exists(alt_config):
                    print(f"使用备选配置文件: {alt_config}")
                    config_path = alt_config
                    break
            else:
                raise FileNotFoundError(f"找不到有效的配置文件")
        
        # 加载配置
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        # 配置OpenAI
        self.api_key = config["openai"]["api_key"]
        self.api_base_url = config["openai"].get("api_base_url")
        self.model = config["openai"]["model"]
        self.temperature = config["openai"]["temperature"]
        self.max_tokens = config["openai"]["max_tokens"]
        
        print(f"API Key: {self.api_key[:10]}...{self.api_key[-5:]}")
        print(f"API Base URL: {self.api_base_url}")
        print(f"Model: {self.model}")
        
        # 批处理配置
        self.batch_size = config["system"]["batch_size"]
        self.cache_enabled = config["system"]["cache_enabled"]
        
        # 多层次匹配配置
        self.enable_aggressive_mode = config["matching_strategy"]["enable_aggressive_mode"]
        self.include_potential_matches = config["matching_strategy"]["include_potential_matches"]
        
        print(f"积极匹配模式: {'启用' if self.enable_aggressive_mode else '禁用'}")
        
        # 缓存
        self._cache = {}
        self._cache_path = "cache/enhanced_llm_responses.json"
        if self.cache_enabled:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            if os.path.exists(self._cache_path):
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
        
        # 统计
        self.success_count = 0
        self.fail_count = 0
        self.cache_hit_count = 0
    
    def match_field_pair(self, 
                        source_schema: Dict, 
                        source_field: Dict, 
                        target_schema: Dict, 
                        target_field: Dict,
                        similarity: float) -> Dict:
        """
        匹配单对字段（增强版）
        """
        # 构建增强提示模板
        prompt = self._create_enhanced_prompt(source_schema, source_field, target_schema, target_field, similarity)
        
        # 缓存Key
        cache_key = f"enhanced_{source_schema['table_name']}_{source_field['name']}_{target_schema['table_name']}_{target_field['name']}"
        
        # 检查缓存
        if self.cache_enabled and cache_key in self._cache:
            self.cache_hit_count += 1
            return self._cache[cache_key]
        
        # 调用LLM
        response = self._call_llm_like_test(prompt)
        
        if response.startswith("错误："):
            self.fail_count += 1
            # 使用增强回退策略
            result = self._generate_enhanced_fallback_result(source_field, target_field, similarity, response)
        else:
            self.success_count += 1
            # 解析响应（增强版）
            result = self._parse_enhanced_response(response, similarity)
        
        # 添加相似度信息
        result["similarity"] = similarity
        
        # 缓存结果
        if self.cache_enabled:
            self._cache[cache_key] = result
            if len(self._cache) % 10 == 0:
                self._save_cache()
        
        return result
    
    def _call_llm_like_test(self, prompt: str) -> str:
        """
        完全模仿测试脚本的API调用方法
        """
        try:
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
            
            response = requests.post(api_url, headers=headers, json=data, timeout=60)
            
            if response.status_code != 200:
                error_msg = f"API调用失败，状态码：{response.status_code}，响应：{response.text}"
                return f"错误：{error_msg}"
            
            try:
                resp_json = response.json()
            except json.JSONDecodeError:
                return f"错误：API返回无效JSON格式，响应：{response.text[:200]}"
            
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
                    return f"错误：API返回格式异常，choices结构：{choice}"
            else:
                return f"错误：API返回格式异常，无choices字段：{resp_json}"
            
        except requests.exceptions.Timeout:
            return "错误：API请求超时"
        except requests.exceptions.ConnectionError:
            return "错误：网络连接失败"
        except requests.exceptions.RequestException as e:
            return f"错误：请求异常 - {str(e)}"
        except Exception as e:
            return f"错误：未知异常 - {str(e)}"
    
    def batch_process_candidates(self, 
                                candidate_pairs: List[Dict],
                                source_schemas: Dict[str, Dict],
                                target_schemas: Dict[str, Dict]) -> List[Dict]:
        """
        批量处理候选匹配对（增强版）
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
            self._save_cache()
        
        # 打印统计信息
        print(f"\n=== LLM调用统计 ===")
        print(f"成功调用: {self.success_count}")
        print(f"失败调用: {self.fail_count}")
        print(f"缓存命中: {self.cache_hit_count}")
        total_calls = self.success_count + self.fail_count
        if total_calls > 0:
            success_rate = self.success_count / total_calls * 100
            print(f"成功率: {success_rate:.1f}%")
        
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
                try:
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
                    
                    # 在批次内添加间隔
                    time.sleep(0.5)
                    
                except Exception as e:
                    print(f"处理候选对失败: {candidate}, 错误: {e}")
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
                    batch_results.append(error_result)
            
            results.extend(batch_results)
            
            # 批次间的等待时间
            if i + self.batch_size < len(candidates):
                time.sleep(1)
        
        return results
    
    def _create_enhanced_prompt(self, 
                               source_schema: Dict, 
                               source_field: Dict, 
                               target_schema: Dict, 
                               target_field: Dict,
                               similarity: float) -> str:
        """
        创建增强版提示模板（更鼓励匹配）
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
- 字段名：{source_field['name']}
- 字段描述：{source_field.get('desc', '(无)')}
- 字段类型：{source_field.get('type', '(未知)')}

目标字段：
- 字段名：{target_field['name']}
- 字段描述：{target_field.get('desc', '(无)')}
- 字段类型：{target_field.get('type', '(未知)')}

分析要点：
1. 字段名语义关系：考虑中英文对应、拼音缩写（如"XSBH"="学生编号"对应"XH"="学号"）
2. 业务概念匹配：分析字段在业务流程中的作用是否相同
3. 描述语义：比较字段描述的语义相似度

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
- 字段名：{source_field['name']}
- 字段描述：{source_field.get('desc', '(无)')}

目标字段：
- 字段名：{target_field['name']}
- 字段描述：{target_field.get('desc', '(无)')}

深度分析要点：
1. 语义挖掘：即使名称不同，分析是否表示相同业务概念
2. 模糊匹配：考虑同义词、近义词、缩写形式
3. 领域知识：运用教育、管理等领域的常识判断

匹配倾向：
- 优先考虑业务语义相似性
- 对于可能的匹配，给予积极的判断

请给出判断和置信度：
格式：
判断：[是/否]
置信度：[0-1之间的数字]
理由：[详细说明]"""

        else:  # low_similarity
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
        
        return prompt
    
    def _parse_enhanced_response(self, response: str, similarity: float) -> Dict:
        """
        解析LLM响应（增强版）
        """
        try:
            # 提取判断结果（更宽松的匹配）
            match = False
            response_lower = response.lower()
            
            # 多种判断模式
            positive_indicators = ["判断：是", "判断:是", "判断: 是", "结论：是", "结论:是", "结论: 是"]
            
            for indicator in positive_indicators:
                if indicator in response:
                    match = True
                    break
            
            # 如果没有明确判断，检查是否有积极的表述
            if not match:
                positive_keywords = ["匹配", "等价", "相同", "一致", "对应", "相似"]
                if any(keyword in response for keyword in positive_keywords):
                    match = True
            
            # 提取置信度
            confidence = self._extract_confidence(response, match, similarity)
            
            # 提取理由
            reason = self._extract_reason(response)
            
            # 如果LLM倾向于匹配但置信度很低，适当提升置信度
            if match and confidence < 0.5 and self.enable_aggressive_mode:
                confidence = max(0.6, confidence + 0.2)
            
            return {
                "match": match,
                "confidence": confidence,
                "reason": reason,
                "llm_response": response
            }
            
        except Exception as e:
            return self._generate_enhanced_fallback_result_from_similarity(similarity, f"解析响应失败: {e}")
    
    def _extract_confidence(self, response: str, match: bool, similarity: float) -> float:
        """提取置信度"""
        confidence = 0.0
        
        confidence_patterns = [
            r"置信度：([0-9.]+)",
            r"置信度:([0-9.]+)", 
            r"置信度: ([0-9.]+)"
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
                confidence = max(0.7, similarity + 0.2)
            else:
                confidence = max(0.3, similarity)
        
        return max(0.1, min(1.0, confidence))
    
    def _extract_reason(self, response: str) -> str:
        """提取理由"""
        reason_patterns = [
            r"理由：(.*?)(?=\n|$)",
            r"理由:(.*?)(?=\n|$)",
            r"理由: (.*?)(?=\n|$)"
        ]
        
        for pattern in reason_patterns:
            match_obj = re.search(pattern, response, re.DOTALL)
            if match_obj:
                reason = match_obj.group(1).strip()
                if reason:
                    return reason[:200]
        
        return response[:200]
    
    def _generate_enhanced_fallback_result(self, source_field: Dict, target_field: Dict, similarity: float, error_msg: str) -> Dict:
        """生成增强的回退结果"""
        s_name = source_field.get("name", "").lower()
        t_name = target_field.get("name", "").lower()
        
        # 更积极的回退策略
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
        
        reason = f"LLM调用失败，基于相似度{similarity:.2f}的增强回退判断。错误：{error_msg[:100]}"
        
        return {
            "match": match,
            "confidence": confidence,
            "reason": reason,
            "fallback": True
        }
    
    def _generate_enhanced_fallback_result_from_similarity(self, similarity: float, error_msg: str) -> Dict:
        """基于相似度生成增强回退结果"""
        if self.enable_aggressive_mode:
            # 积极模式：更容易判断为匹配
            if similarity >= 0.5:
                match = True
                confidence = min(0.8, similarity + 0.2)
            elif similarity >= 0.3:
                match = True
                confidence = min(0.7, similarity + 0.3)
            else:
                match = False
                confidence = similarity
        else:
            # 保守模式
            if similarity >= 0.7:
                match = True
                confidence = similarity
            else:
                match = False
                confidence = similarity
        
        reason = f"基于相似度{similarity:.2f}的增强回退策略。{error_msg}"
        
        return {
            "match": match,
            "confidence": confidence,
            "reason": reason,
            "fallback": True
        }
    
    def _save_cache(self):
        """保存缓存"""
        try:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存失败: {e}")