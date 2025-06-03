"""
鲁棒性增强的LLM匹配器 - 带有重试机制和错误处理
"""
import os
import time
import json
import re
import requests
from typing import Dict, List, Tuple, Any, Optional
import yaml
from tqdm import tqdm
import random


class RobustLLMMatcher:
    """鲁棒性增强的LLM匹配类"""
    
    def __init__(self, config_path: str = "config/config_enhanced.yaml"):
        """
        初始化鲁棒性LLM匹配器
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
        
        # 重试配置
        self.max_retries = 3
        self.retry_delay = 2  # 秒
        self.exponential_backoff = True
        
        # 批处理配置
        self.batch_size = config["system"]["batch_size"]
        self.cache_enabled = config["system"]["cache_enabled"]
        
        # 缓存
        self._cache = {}
        self._cache_path = "cache/robust_llm_responses.json"
        if self.cache_enabled:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            if os.path.exists(self._cache_path):
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
        
        # 统计信息
        self.api_call_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "cached_hits": 0,
            "retry_counts": 0
        }
    
    def match_field_pair(self, 
                        source_schema: Dict, 
                        source_field: Dict, 
                        target_schema: Dict, 
                        target_field: Dict,
                        similarity: float) -> Dict:
        """
        匹配单对字段（带重试机制）
        """
        # 构建提示模板
        prompt = self._create_enhanced_prompt(source_schema, source_field, target_schema, target_field, similarity)
        
        # 缓存Key
        cache_key = f"robust_{source_schema['table_name']}_{source_field['name']}_{target_schema['table_name']}_{target_field['name']}"
        
        # 检查缓存
        if self.cache_enabled and cache_key in self._cache:
            self.api_call_stats["cached_hits"] += 1
            return self._cache[cache_key]
        
        # 调用LLM（带重试）
        self.api_call_stats["total_calls"] += 1
        
        result = self._call_llm_with_retry(prompt, similarity, source_field, target_field)
        
        # 添加相似度信息
        result["similarity"] = similarity
        
        # 缓存结果
        if self.cache_enabled and result.get("confidence", 0) > 0:
            self._cache[cache_key] = result
            # 定期保存缓存
            if len(self._cache) % 5 == 0:
                self._save_cache()
        
        return result
    
    def _call_llm_with_retry(self, prompt: str, similarity: float, source_field: Dict, target_field: Dict) -> Dict:
        """
        带重试机制的LLM调用
        """
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    self.api_call_stats["retry_counts"] += 1
                    # 指数退避
                    delay = self.retry_delay * (2 ** (attempt - 1)) if self.exponential_backoff else self.retry_delay
                    # 添加随机因子避免雷群效应
                    delay += random.uniform(0, 1)
                    print(f"第 {attempt} 次重试，等待 {delay:.1f} 秒...")
                    time.sleep(delay)
                
                response = self._call_llm(prompt)
                
                if not response.startswith("错误："):
                    # 成功调用
                    result = self._parse_enhanced_response(response, similarity)
                    self.api_call_stats["successful_calls"] += 1
                    return result
                else:
                    last_error = response
                    print(f"第 {attempt + 1} 次尝试失败: {response}")
                    
            except Exception as e:
                last_error = str(e)
                print(f"第 {attempt + 1} 次尝试异常: {e}")
        
        # 所有重试都失败，使用回退策略
        print(f"API调用完全失败，使用回退策略")
        self.api_call_stats["failed_calls"] += 1
        return self._generate_enhanced_fallback_result(source_field, target_field, similarity, last_error)
    
    def _call_llm(self, prompt: str) -> str:
        """
        调用LLM API
        """
        try:
            api_url = f"{self.api_base_url}/chat/completions" if self.api_base_url else "https://api.openai.com/v1/chat/completions"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "User-Agent": "Schema-Matching-Tool/1.0"
            }
            
            data = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            # 增加超时时间
            response = requests.post(api_url, headers=headers, json=data, timeout=120)
            
            # 详细的错误处理
            if response.status_code == 401:
                return "错误：API认证失败，请检查API密钥是否正确"
            elif response.status_code == 403:
                return "错误：API访问被禁止，可能是权限问题"
            elif response.status_code == 429:
                return "错误：API请求频率过高，请稍后重试"
            elif response.status_code == 500:
                return "错误：API服务器内部错误"
            elif response.status_code == 502 or response.status_code == 503:
                return "错误：API服务暂时不可用"
            elif response.status_code != 200:
                return f"错误：API调用失败，状态码：{response.status_code}，响应：{response.text[:200]}"
            
            # 解析响应
            try:
                resp_json = response.json()
            except json.JSONDecodeError:
                return f"错误：API返回无效JSON格式，响应：{response.text[:200]}"
            
            # 提取内容
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
    
    def _generate_enhanced_fallback_result(self, source_field: Dict, target_field: Dict, similarity: float, error_msg: str) -> Dict:
        """
        生成增强的回退结果
        """
        s_name = source_field.get("name", "").lower()
        t_name = target_field.get("name", "").lower()
        s_desc = source_field.get("desc", "").lower()
        t_desc = target_field.get("desc", "").lower()
        
        # 多层次回退策略
        
        # 1. 完全匹配
        if s_name == t_name:
            return {
                "match": True,
                "confidence": 0.95,
                "reason": f"字段名完全匹配 (回退策略)",
                "fallback": True,
                "error": error_msg
            }
        
        # 2. 高相似度 + 明显的业务关联
        if similarity >= 0.8:
            # 检查常见的业务概念
            business_keywords = {
                "id": ["id", "编号", "号码", "代码"],
                "name": ["name", "名称", "姓名", "名字"],
                "time": ["time", "日期", "时间", "date"],
                "code": ["code", "编码", "代码", "编号"],
                "type": ["type", "类型", "种类"],
                "status": ["status", "状态", "情况"]
            }
            
            for concept, keywords in business_keywords.items():
                s_match = any(kw in s_name or kw in s_desc for kw in keywords)
                t_match = any(kw in t_name or kw in t_desc for kw in keywords)
                
                if s_match and t_match:
                    return {
                        "match": True,
                        "confidence": min(0.85, similarity + 0.1),
                        "reason": f"高相似度({similarity:.2f})+业务概念匹配({concept}) (回退策略)",
                        "fallback": True,
                        "error": error_msg
                    }
        
        # 3. 中等相似度判断
        if similarity >= 0.6:
            return {
                "match": True,
                "confidence": min(0.75, similarity + 0.05),
                "reason": f"中等相似度匹配({similarity:.2f}) (回退策略)",
                "fallback": True,
                "error": error_msg
            }
        
        # 4. 拼音或包含关系检查
        if similarity >= 0.4:
            # 检查包含关系
            if (s_name and t_name and (s_name in t_name or t_name in s_name)) or \
               (s_desc and t_desc and (s_desc in t_desc or t_desc in s_desc)):
                return {
                    "match": True,
                    "confidence": min(0.7, similarity + 0.1),
                    "reason": f"字段包含关系+相似度({similarity:.2f}) (回退策略)",
                    "fallback": True,
                    "error": error_msg
                }
        
        # 5. 默认不匹配
        return {
            "match": False,
            "confidence": similarity * 0.8,  # 降低置信度
            "reason": f"相似度较低({similarity:.2f})，判断为不匹配 (回退策略)",
            "fallback": True,
            "error": error_msg
        }
    
    def batch_process_candidates(self, 
                                candidate_pairs: List[Dict],
                                source_schemas: Dict[str, Dict],
                                target_schemas: Dict[str, Dict]) -> List[Dict]:
        """
        批量处理候选匹配对（鲁棒版）
        """
        results = []
        
        print(f"开始处理 {len(candidate_pairs)} 对候选匹配...")
        print(f"批处理大小: {self.batch_size}")
        
        # 分批处理，减少并发压力
        for i in tqdm(range(0, len(candidate_pairs), self.batch_size), desc="鲁棒批量处理"):
            batch = candidate_pairs[i:i+self.batch_size]
            
            batch_results = []
            for j, candidate in enumerate(batch):
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
                    
                    # 在批次内也要有间隔
                    if j < len(batch) - 1:
                        time.sleep(0.5)
                        
                except Exception as e:
                    print(f"处理候选对失败: {candidate}, 错误: {e}")
                    # 创建一个错误结果
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
            if i + self.batch_size < len(candidate_pairs):
                time.sleep(1)
        
        # 保存最终缓存
        if self.cache_enabled:
            self._save_cache()
        
        # 打印统计信息
        self._print_api_stats()
        
        return results
    
    def _save_cache(self):
        """保存缓存"""
        try:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存失败: {e}")
    
    def _print_api_stats(self):
        """打印API调用统计"""
        stats = self.api_call_stats
        print(f"\n=== API调用统计 ===")
        print(f"总调用次数: {stats['total_calls']}")
        print(f"成功调用: {stats['successful_calls']}")
        print(f"失败调用: {stats['failed_calls']}")
        print(f"缓存命中: {stats['cached_hits']}")
        print(f"重试次数: {stats['retry_counts']}")
        
        if stats['total_calls'] > 0:
            success_rate = stats['successful_calls'] / stats['total_calls'] * 100
            print(f"成功率: {success_rate:.1f}%")
    
    def _create_enhanced_prompt(self, source_schema: Dict, source_field: Dict, target_schema: Dict, target_field: Dict, similarity: float) -> str:
        """创建增强提示（简化版，减少token消耗）"""
        
        # 为了减少API调用失败，使用更简洁的提示
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
        """解析LLM响应（增强版）"""
        try:
            # 提取判断结果
            match = False
            if any(indicator in response for indicator in ["判断：是", "判断:是", "判断: 是"]):
                match = True
            
            # 提取置信度
            confidence = 0.0
            confidence_patterns = [r"置信度：([0-9.]+)", r"置信度:([0-9.]+)", r"置信度: ([0-9.]+)"]
            
            for pattern in confidence_patterns:
                match_obj = re.search(pattern, response)
                if match_obj:
                    try:
                        confidence = float(match_obj.group(1))
                        break
                    except ValueError:
                        pass
            
            # 如果没有提取到置信度，基于匹配结果和相似度推算
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
                reason = response[:100] + "..." if len(response) > 100 else response
            
            return {
                "match": match,
                "confidence": confidence,
                "reason": reason,
                "llm_response": response
            }
            
        except Exception as e:
            # 解析失败，返回基于相似度的结果
            return {
                "match": similarity > 0.6,
                "confidence": similarity,
                "reason": f"解析LLM响应失败: {e}，基于相似度判断",
                "parse_error": True
            }