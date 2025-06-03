"""
LLM语义匹配模块
"""
import os
import time
import json
import re
import requests
from typing import Dict, List, Tuple, Any, Optional
import yaml
from tqdm import tqdm


class LLMMatcher:
    """基于LLM的语义匹配类"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化LLM匹配器
        
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
        
        # 缓存
        self._cache = {}
        self._cache_path = "cache/llm_responses.json"
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
        匹配单对字段
        
        Args:
            source_schema: 源表元数据
            source_field: 源字段元数据
            target_schema: 目标表元数据
            target_field: 目标字段元数据
            similarity: 计算的相似度得分
            
        Returns:
            匹配结果，包含是否匹配、置信度和理由
        """
        # 构建提示模板
        prompt = self._create_prompt(source_schema, source_field, target_schema, target_field, similarity)
        
        # 缓存Key
        cache_key = f"{source_schema['table_name']}_{source_field['name']}_{target_schema['table_name']}_{target_field['name']}"
        
        # 检查缓存
        if self.cache_enabled and cache_key in self._cache:
            return self._cache[cache_key]
        
        # 调用LLM
        try:
            response = self._call_llm(prompt)
            
            # 解析响应
            result = self._parse_response(response)
            
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
            # 返回默认结果
            return {
                "match": False,
                "confidence": 0.0,
                "reason": f"LLM调用失败: {e}",
                "similarity": similarity
            }
    
    def batch_process_candidates(self, 
                                candidate_pairs: List[Dict],
                                source_schemas: Dict[str, Dict],
                                target_schemas: Dict[str, Dict]) -> List[Dict]:
        """
        批量处理候选匹配对
        
        Args:
            candidate_pairs: 候选匹配对列表
            source_schemas: 源表元数据字典
            target_schemas: 目标表元数据字典
            
        Returns:
            处理结果列表
        """
        results = []
        
        # 分批处理
        for i in tqdm(range(0, len(candidate_pairs), self.batch_size), desc="批量处理候选匹配对"):
            batch = candidate_pairs[i:i+self.batch_size]
            
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
            time.sleep(0.5)
        
        # 保存缓存
        if self.cache_enabled:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        
        return results
    
    def _create_prompt(self, 
                      source_schema: Dict, 
                      source_field: Dict, 
                      target_schema: Dict, 
                      target_field: Dict,
                      similarity: float) -> str:
        """
        创建提示模板
        
        Args:
            source_schema: 源表元数据
            source_field: 源字段元数据
            target_schema: 目标表元数据
            target_field: 目标字段元数据
            similarity: 计算的相似度得分
            
        Returns:
            提示模板
        """
        prompt = f"""系统角色：您是数据集成和Schema匹配专家，擅长分析表结构和字段关系。

        任务描述：判断两个字段是否语义等价。每个字段有名称和描述（注释）。

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

        计算的相似度：{similarity:.2f}

        分析问题：
        1. 分析字段名称的语义关系（考虑缩写、拼音转换等）
        2. 比较字段描述的语义相似度
        3. 考虑中英文专业术语对应关系
        4. 分析字段在各自表中的作用是否相同

        源字段和目标字段是否语义等价？
        1. 回答[是/否]
        2. 给出判断的置信度（0-1之间的数字）
        3. 简要解释理由

        请按照以下格式回答：
        判断：[是/否]
        置信度：[0-1之间的数字]
        理由：[简要解释]
        """
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """
        调用LLM - 直接使用HTTP请求，避免openai库的兼容性问题
        
        Args:
            prompt: 提示模板
            
        Returns:
            LLM响应
        """
        try:
            # 如果没有指定base_url，使用默认的OpenAI API URL
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
            
            # 检查响应状态
            if response.status_code != 200:
                error_msg = f"API调用失败，状态码：{response.status_code}，响应：{response.text}"
                print(error_msg)
                return f"错误：{error_msg}"
            
            # 解析响应
            resp_json = response.json()
            
            # 从响应中提取文本内容
            if "choices" in resp_json and len(resp_json["choices"]) > 0:
                if "message" in resp_json["choices"][0] and "content" in resp_json["choices"][0]["message"]:
                    return resp_json["choices"][0]["message"]["content"]
                elif "text" in resp_json["choices"][0]:
                    return resp_json["choices"][0]["text"]
            
            # 如果找不到预期的结构，返回原始响应
            return str(resp_json)
            
        except Exception as e:
            error_msg = f"API调用异常: {str(e)}"
            print(error_msg)
            return f"错误：{error_msg}"
    
    def _parse_response(self, response: str) -> Dict:
        """
        解析LLM响应
        
        Args:
            response: LLM响应
            
        Returns:
            解析结果
        """
        try:
            # 如果响应以"错误："开头，表示调用失败
            if response.startswith("错误："):
                return {
                    "match": False,
                    "confidence": 0.0,
                    "reason": response
                }
            
            # 提取判断结果
            match = False
            if "判断：是" in response or "判断:是" in response or "判断: 是" in response:
                match = True
            
            # 提取置信度
            confidence = 0.0
            
            # 尝试不同的置信度表达方式
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
            
            # 提取理由
            reason = ""
            reason_patterns = [
                r"理由：(.*?)(?=$|\n\n)",
                r"理由:(.*?)(?=$|\n\n)",
                r"理由: (.*?)(?=$|\n\n)",
                r"reason:(.*?)(?=$|\n\n)",
                r"reason：(.*?)(?=$|\n\n)",
                r"reason: (.*?)(?=$|\n\n)"
            ]
            
            for pattern in reason_patterns:
                match_obj = re.search(pattern, response, re.DOTALL)
                if match_obj:
                    reason = match_obj.group(1).strip()
                    break
            
            # 如果没有找到理由，使用整个响应
            if not reason:
                reason = response
            
            # 如果置信度为0但判断为是，设置一个默认值
            if match and confidence < 0.1:
                confidence = 0.8  # 设置一个默认的较高置信度
            
            return {
                "match": match,
                "confidence": confidence,
                "reason": reason
            }
            
        except Exception as e:
            print(f"解析LLM响应失败: {e}")
            # 返回默认结果
            return {
                "match": False,
                "confidence": 0.0,
                "reason": f"解析LLM响应失败: {e}\n原始响应: {response}"
            }