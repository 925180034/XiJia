"""
修复版LLM匹配器 - 与测试脚本完全一致的实现
"""
import os
import time
import json
import re
import requests
from typing import Dict, List, Tuple, Any, Optional
import yaml
from tqdm import tqdm


class FixedLLMMatcher:
    """修复版LLM匹配类 - 与测试脚本逻辑完全一致"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化修复版LLM匹配器
        """
        print(f"加载配置文件: {config_path}")
        
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
        
        # 配置OpenAI - 与测试脚本完全一致
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
        
        # 缓存
        self._cache = {}
        self._cache_path = "cache/fixed_llm_responses.json"
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
        匹配单对字段
        """
        # 缓存Key
        cache_key = f"fixed_{source_schema['table_name']}_{source_field['name']}_{target_schema['table_name']}_{target_field['name']}"
        
        # 检查缓存
        if self.cache_enabled and cache_key in self._cache:
            self.cache_hit_count += 1
            return self._cache[cache_key]
        
        # 创建简化的提示
        prompt = self._create_simple_prompt(source_schema, source_field, target_schema, target_field, similarity)
        
        # 调用LLM - 与测试脚本完全一致的方法
        response = self._call_llm_like_test(prompt)
        
        if response.startswith("错误："):
            self.fail_count += 1
            # 使用回退策略
            result = self._generate_fallback_result(source_field, target_field, similarity, response)
        else:
            self.success_count += 1
            # 解析响应
            result = self._parse_response(response, similarity)
        
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
            # 与debug_api_calls.py完全一致的实现
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
            
            # 与测试脚本一致的状态码检查
            if response.status_code != 200:
                error_msg = f"API调用失败，状态码：{response.status_code}，响应：{response.text}"
                return f"错误：{error_msg}"
            
            # 与测试脚本一致的响应解析
            try:
                resp_json = response.json()
            except json.JSONDecodeError:
                return f"错误：API返回无效JSON格式，响应：{response.text[:200]}"
            
            # 提取内容 - 与测试脚本完全一致
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
    
    def _create_simple_prompt(self, source_schema: Dict, source_field: Dict, target_schema: Dict, target_field: Dict, similarity: float) -> str:
        """
        创建简化的提示模板
        """
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
    
    def _parse_response(self, response: str, similarity: float) -> Dict:
        """
        解析LLM响应
        """
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
    
    def _generate_fallback_result(self, source_field: Dict, target_field: Dict, similarity: float, error_msg: str) -> Dict:
        """
        生成回退结果
        """
        s_name = source_field.get("name", "").lower()
        t_name = target_field.get("name", "").lower()
        
        # 简单的回退逻辑
        if s_name == t_name:
            match = True
            confidence = 0.95
        elif similarity > 0.8:
            match = True
            confidence = 0.8
        elif similarity > 0.6:
            match = True
            confidence = 0.7
        else:
            match = False
            confidence = similarity
        
        reason = f"LLM调用失败，基于相似度{similarity:.2f}的回退判断。错误：{error_msg[:100]}"
        
        return {
            "match": match,
            "confidence": confidence,
            "reason": reason,
            "fallback": True
        }
    
    def batch_process_candidates(self, 
                                candidate_pairs: List[Dict],
                                source_schemas: Dict[str, Dict],
                                target_schemas: Dict[str, Dict]) -> List[Dict]:
        """
        批量处理候选匹配对
        """
        results = []
        
        print(f"开始处理 {len(candidate_pairs)} 对候选匹配...")
        print(f"批处理大小: {self.batch_size}")
        
        # 分批处理
        for i in tqdm(range(0, len(candidate_pairs), self.batch_size), desc="修复版批量处理"):
            batch = candidate_pairs[i:i+self.batch_size]
            
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
            if i + self.batch_size < len(candidate_pairs):
                time.sleep(1)
        
        # 保存最终缓存
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
    
    def _save_cache(self):
        """保存缓存"""
        try:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存失败: {e}")