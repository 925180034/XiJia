# """
# 交互式增强Schema匹配脚本 - 支持动态参数调整
# """
# import os
# import sys
# import yaml
# import pandas as pd
# import numpy as np
# from typing import Dict, List, Tuple, Any, Optional
# import time
# import argparse
# import json
# import re
# import requests
# from tqdm import tqdm

# from src.data.data_loader import DataLoader, SchemaMetadata
# from src.data.data_preprocessor import MetadataPreprocessor
# from src.features.enhanced_similarity_calculator import EnhancedSimilarityCalculator
# from src.matching.candidate_filter import CandidateFilter
# from src.matching.result_processor import ResultProcessor


# class InteractiveEnhancedLLMMatcher:
#     """交互式增强LLM匹配器"""
    
#     def __init__(self, config_path: str = "config/config_enhanced.yaml"):
#         print(f"加载配置文件: {config_path}")
        
#         with open(config_path, "r", encoding="utf-8") as f:
#             self.config = yaml.safe_load(f)
        
#         self.api_key = self.config["openai"]["api_key"]
#         self.api_base_url = self.config["openai"]["api_base_url"]
#         self.model = self.config["openai"]["model"]
#         self.temperature = self.config["openai"]["temperature"]
#         self.max_tokens = self.config["openai"]["max_tokens"]
        
#         # 可动态调整的参数
#         self.batch_size = self.config["system"]["batch_size"]
#         self.cache_enabled = self.config["system"]["cache_enabled"]
#         self.enable_aggressive_mode = self.config["matching_strategy"]["enable_aggressive_mode"]
        
#         # 阈值参数（可动态调整）
#         self.similarity_threshold = self.config["thresholds"]["similarity_threshold"]
#         self.high_confidence_threshold = self.config["thresholds"]["high_confidence"]
#         self.medium_confidence_threshold = self.config["thresholds"]["medium_confidence"]
#         self.low_confidence_threshold = self.config["thresholds"]["low_confidence"]
        
#         print(f"API配置: {self.api_base_url}, 模型: {self.model}")
#         print(f"当前阈值配置:")
#         print(f"  相似度阈值: {self.similarity_threshold}")
#         print(f"  高置信度阈值: {self.high_confidence_threshold}")
#         print(f"  中等置信度阈值: {self.medium_confidence_threshold}")
#         print(f"  低置信度阈值: {self.low_confidence_threshold}")
        
#         # 缓存
#         self._cache = {}
#         self._cache_path = "cache/interactive_enhanced_responses.json"
#         if self.cache_enabled:
#             os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
#             if os.path.exists(self._cache_path):
#                 with open(self._cache_path, "r", encoding="utf-8") as f:
#                     self._cache = json.load(f)
        
#         self.success_count = 0
#         self.fail_count = 0
#         self.cache_hit_count = 0
    
#     def update_thresholds(self, similarity_threshold=None, high_conf=None, 
#                          medium_conf=None, low_conf=None):
#         """动态更新阈值"""
#         if similarity_threshold is not None:
#             self.similarity_threshold = similarity_threshold
#             print(f"✅ 相似度阈值更新为: {self.similarity_threshold}")
        
#         if high_conf is not None:
#             self.high_confidence_threshold = high_conf
#             print(f"✅ 高置信度阈值更新为: {self.high_confidence_threshold}")
        
#         if medium_conf is not None:
#             self.medium_confidence_threshold = medium_conf
#             print(f"✅ 中等置信度阈值更新为: {self.medium_confidence_threshold}")
        
#         if low_conf is not None:
#             self.low_confidence_threshold = low_conf
#             print(f"✅ 低置信度阈值更新为: {self.low_confidence_threshold}")
    
#     def update_matching_settings(self, batch_size=None, aggressive_mode=None):
#         """动态更新匹配设置"""
#         if batch_size is not None:
#             self.batch_size = batch_size
#             print(f"✅ 批处理大小更新为: {self.batch_size}")
        
#         if aggressive_mode is not None:
#             self.enable_aggressive_mode = aggressive_mode
#             print(f"✅ 积极匹配模式: {'启用' if self.enable_aggressive_mode else '禁用'}")
    
#     def _call_api_like_test(self, prompt: str) -> str:
#         """API调用方法"""
#         try:
#             api_url = f"{self.api_base_url}/chat/completions"
            
#             headers = {
#                 "Content-Type": "application/json",
#                 "Authorization": f"Bearer {self.api_key}"
#             }
            
#             data = {
#                 "model": self.model,
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature": self.temperature,
#                 "max_tokens": self.max_tokens
#             }
            
#             response = requests.post(api_url, headers=headers, json=data, timeout=30)
            
#             if response.status_code != 200:
#                 return f"错误：API调用失败，状态码：{response.status_code}"
            
#             resp_json = response.json()
            
#             if "choices" in resp_json and len(resp_json["choices"]) > 0:
#                 choice = resp_json["choices"][0]
#                 if "message" in choice and "content" in choice["message"]:
#                     content = choice["message"]["content"]
#                     if content and content.strip():
#                         return content
#                     else:
#                         return "错误：API返回空内容"
#                 elif "text" in choice:
#                     return choice["text"]
#                 else:
#                     return f"错误：响应格式异常"
#             else:
#                 return f"错误：响应中没有choices字段"
                
#         except requests.exceptions.Timeout:
#             return "错误：请求超时"
#         except requests.exceptions.ConnectionError:
#             return "错误：连接错误"
#         except Exception as e:
#             return f"错误：{str(e)}"
    
#     def match_field_pair(self, source_field: Dict, target_field: Dict, similarity: float) -> Dict:
#         """匹配单对字段"""
#         cache_key = f"interactive_{source_field['name']}_{target_field['name']}_{similarity:.3f}"
        
#         if self.cache_enabled and cache_key in self._cache:
#             self.cache_hit_count += 1
#             return self._cache[cache_key]
        
#         # 根据相似度调整提示策略
#         if similarity >= 0.8:
#             prompt_type = "high_similarity"
#         elif similarity >= 0.5:
#             prompt_type = "medium_similarity"
#         else:
#             prompt_type = "low_similarity"
        
#         prompt = self._create_adaptive_prompt(source_field, target_field, similarity, prompt_type)
        
#         response = self._call_api_like_test(prompt)
        
#         if response.startswith("错误："):
#             self.fail_count += 1
#             result = self._generate_fallback_result(source_field, target_field, similarity, response)
#         else:
#             self.success_count += 1
#             result = self._parse_response(response, similarity)
        
#         result["similarity"] = similarity
        
#         if self.cache_enabled:
#             self._cache[cache_key] = result
#             if len(self._cache) % 10 == 0:
#                 self._save_cache()
        
#         return result
    
#     def _create_adaptive_prompt(self, source_field: Dict, target_field: Dict, 
#                                similarity: float, prompt_type: str) -> str:
#         """根据相似度创建自适应提示"""
        
#         if prompt_type == "high_similarity":
#             # 高相似度：详细分析
#             prompt = f"""系统角色：您是数据集成专家，擅长识别字段间的语义等价关系。

# 任务描述：判断两个字段是否语义等价。这两个字段具有较高的相似度({similarity:.2f})，请仔细分析它们是否表示相同的业务概念。

# 源字段：
# - 字段名：{source_field['name']}
# - 字段描述：{source_field.get('desc', '(无)')}
# - 字段类型：{source_field.get('type', '(未知)')}

# 目标字段：
# - 字段名：{target_field['name']}
# - 字段描述：{target_field.get('desc', '(无)')}
# - 字段类型：{target_field.get('type', '(未知)')}

# 分析要点：
# 1. 字段名语义关系：考虑中英文对应、拼音缩写（如"XSBH"="学生编号"对应"XH"="学号"）
# 2. 业务概念匹配：分析字段在业务流程中的作用是否相同
# 3. 描述语义：比较字段描述的语义相似度
# 4. 数据类型兼容性：检查数据类型是否兼容

# 判断标准：
# - 如果两个字段表示相同的业务概念，即使名称不完全相同，也应该判断为匹配
# - 中英文字段如果语义相同，应判断为匹配
# - 拼音缩写与原词如果对应，应判断为匹配

# 请给出判断和置信度：
# 格式：
# 判断：[是/否]
# 置信度：[0-1之间的数字]
# 理由：[详细说明]"""

#         elif prompt_type == "medium_similarity":
#             # 中等相似度：重点分析
#             prompt = f"""系统角色：您是数据集成专家，擅长在复杂情况下识别字段间的潜在等价关系。

# 任务描述：判断两个字段是否可能语义等价。这两个字段具有中等相似度({similarity:.2f})，需要深入分析业务语义。

# 源字段：{source_field['name']} ({source_field.get('desc', '无描述')})
# 目标字段：{target_field['name']} ({target_field.get('desc', '无描述')})

# 深度分析要点：
# 1. 语义挖掘：即使名称不同，分析是否表示相同业务概念
# 2. 模糊匹配：考虑同义词、近义词、缩写形式
# 3. 领域知识：运用教育、管理等领域的常识判断

# 匹配倾向：
# - 优先考虑业务语义相似性
# - 对于可能的匹配，给予积极的判断

# 请给出判断和置信度：
# 格式：
# 判断：[是/否]
# 置信度：[0-1之间的数字]
# 理由：[详细说明]"""

#         else:
#             # 低相似度：简化分析
#             prompt = f"""判断两个数据库字段是否语义等价：

# 源字段：{source_field['name']} ({source_field.get('desc', '无描述')})
# 目标字段：{target_field['name']} ({target_field.get('desc', '无描述')})
# 相似度：{similarity:.2f}

# 分析要点：
# 1. 字段名语义关系（考虑中英文对应、拼音缩写）
# 2. 业务概念是否相同
# 3. 描述内容匹配度

# 回答格式：
# 判断：是/否
# 置信度：0-1数字
# 理由：简要说明"""
        
#         return prompt
    
#     def _parse_response(self, response: str, similarity: float) -> Dict:
#         """解析LLM响应"""
#         try:
#             match = False
#             if any(indicator in response for indicator in ["判断：是", "判断:是", "判断: 是"]):
#                 match = True
            
#             confidence = 0.0
#             patterns = [r"置信度：([0-9.]+)", r"置信度:([0-9.]+)", r"置信度: ([0-9.]+)"]
            
#             for pattern in patterns:
#                 match_obj = re.search(pattern, response)
#                 if match_obj:
#                     try:
#                         confidence = float(match_obj.group(1))
#                         break
#                     except ValueError:
#                         pass
            
#             if confidence == 0.0:
#                 if match:
#                     confidence = max(0.7, similarity + 0.1)
#                 else:
#                     confidence = min(0.4, similarity)
            
#             reason_patterns = [r"理由：(.*?)(?=\n|$)", r"理由:(.*?)(?=\n|$)", r"理由: (.*?)(?=\n|$)"]
#             reason = ""
            
#             for pattern in reason_patterns:
#                 match_obj = re.search(pattern, response, re.DOTALL)
#                 if match_obj:
#                     reason = match_obj.group(1).strip()
#                     break
            
#             if not reason:
#                 reason = response[:100] + "..."
            
#             return {
#                 "match": match,
#                 "confidence": confidence,
#                 "reason": reason,
#                 "llm_response": response
#             }
            
#         except Exception as e:
#             return {
#                 "match": similarity > 0.6,
#                 "confidence": similarity,
#                 "reason": f"解析失败: {e}，基于相似度判断",
#                 "parse_error": True
#             }
    
#     def _generate_fallback_result(self, source_field: Dict, target_field: Dict, 
#                                  similarity: float, error_msg: str) -> Dict:
#         """生成回退结果"""
#         s_name = source_field.get("name", "").lower()
#         t_name = target_field.get("name", "").lower()
        
#         if s_name == t_name:
#             match = True
#             confidence = 0.95
#         elif similarity >= 0.8:
#             match = True
#             confidence = 0.8
#         elif similarity >= 0.6:
#             match = True
#             confidence = 0.7
#         elif similarity >= 0.4 and self.enable_aggressive_mode:
#             match = True
#             confidence = 0.6
#         else:
#             match = False
#             confidence = similarity
        
#         reason = f"API调用失败，基于相似度{similarity:.2f}的智能回退"
        
#         return {
#             "match": match,
#             "confidence": confidence,
#             "reason": reason,
#             "fallback": True
#         }
    
#     def batch_process_candidates(self, candidate_pairs: List[Dict], 
#                                 source_schemas: Dict[str, Dict],
#                                 target_schemas: Dict[str, Dict],
#                                 max_candidates: int = None) -> List[Dict]:
#         """批量处理候选匹配对"""
        
#         if max_candidates and len(candidate_pairs) > max_candidates:
#             print(f"候选对数量({len(candidate_pairs)})超过限制({max_candidates})，只处理前{max_candidates}个")
#             candidate_pairs = candidate_pairs[:max_candidates]
        
#         results = []
        
#         print(f"开始处理 {len(candidate_pairs)} 对候选匹配...")
        
#         # 分层处理
#         high_sim = [c for c in candidate_pairs if c["similarity"] >= 0.7]
#         medium_sim = [c for c in candidate_pairs if 0.4 <= c["similarity"] < 0.7]
#         low_sim = [c for c in candidate_pairs if c["similarity"] < 0.4]
        
#         print(f"高相似度: {len(high_sim)}, 中等相似度: {len(medium_sim)}, 低相似度: {len(low_sim)}")
        
#         for candidates, desc in [(high_sim, "高相似度"), (medium_sim, "中等相似度"), (low_sim, "低相似度")]:
#             if not candidates:
#                 continue
                
#             print(f"处理{desc}候选对...")
            
#             for candidate in tqdm(candidates, desc=f"处理{desc}"):
#                 try:
#                     source_schema = source_schemas[candidate["source_table"]]
#                     source_field = next(f for f in source_schema["fields"] if f["name"] == candidate["source_field"])
                    
#                     target_schema = target_schemas[candidate["target_table"]]
#                     target_field = next(f for f in target_schema["fields"] if f["name"] == candidate["target_field"])
                    
#                     result = self.match_field_pair(source_field, target_field, candidate["similarity"])
                    
#                     result["source_table"] = candidate["source_table"]
#                     result["source_field"] = candidate["source_field"]
#                     result["target_table"] = candidate["target_table"]
#                     result["target_field"] = candidate["target_field"]
                    
#                     results.append(result)
                    
#                     # API调用间隔
#                     if not result.get("fallback", False):
#                         time.sleep(1)
                        
#                 except Exception as e:
#                     print(f"处理失败: {candidate}, 错误: {e}")
        
#         if self.cache_enabled:
#             self._save_cache()
        
#         print(f"\n=== API调用统计 ===")
#         print(f"成功调用: {self.success_count}")
#         print(f"失败调用: {self.fail_count}")
#         print(f"缓存命中: {self.cache_hit_count}")
        
#         return results
    
#     def _save_cache(self):
#         """保存缓存"""
#         try:
#             with open(self._cache_path, "w", encoding="utf-8") as f:
#                 json.dump(self._cache, f, ensure_ascii=False, indent=2)
#         except Exception as e:
#             print(f"保存缓存失败: {e}")


# def interactive_parameter_setup():
#     """交互式参数设置"""
#     print("\n" + "="*60)
#     print("                 交互式参数设置")
#     print("="*60)
    
#     params = {}
    
#     # 数据文件选择
#     print("\n1. 数据文件选择")
#     source_options = [
#         "data/源数据字典.xlsx",
#         "data/自定义源文件.xlsx"
#     ]
#     target_options = [
#         "data/项目匹配字典.xlsx", 
#         "data/项目匹配字典_列类型注释.xlsx",
#         "data/自定义目标文件.xlsx"
#     ]
    
#     print("源文件选项:")
#     for i, option in enumerate(source_options):
#         exists = "✅" if os.path.exists(option) else "❌"
#         print(f"  {i+1}. {option} {exists}")
    
#     while True:
#         try:
#             choice = int(input("选择源文件 (1-2): ")) - 1
#             if 0 <= choice < len(source_options):
#                 params["source_file"] = source_options[choice]
#                 break
#         except ValueError:
#             pass
#         print("无效选择，请重新输入")
    
#     print("\n目标文件选项:")
#     for i, option in enumerate(target_options):
#         exists = "✅" if os.path.exists(option) else "❌"
#         print(f"  {i+1}. {option} {exists}")
    
#     while True:
#         try:
#             choice = int(input("选择目标文件 (1-3): ")) - 1
#             if 0 <= choice < len(target_options):
#                 params["target_file"] = target_options[choice]
#                 break
#         except ValueError:
#             pass
#         print("无效选择，请重新输入")
    
#     # 阈值设置
#     print("\n2. 阈值参数设置")
#     print("当前默认值：")
#     print("  相似度阈值: 0.2 (初筛候选对)")
#     print("  高置信度阈值: 0.8")
#     print("  中等置信度阈值: 0.6")
#     print("  低置信度阈值: 0.4")
    
#     if input("\n是否修改阈值设置? (y/n): ").lower() == 'y':
#         params["similarity_threshold"] = float(input("相似度阈值 (0-1, 建议0.2): ") or "0.2")
#         params["high_confidence"] = float(input("高置信度阈值 (0-1, 建议0.8): ") or "0.8")
#         params["medium_confidence"] = float(input("中等置信度阈值 (0-1, 建议0.6): ") or "0.6")
#         params["low_confidence"] = float(input("低置信度阈值 (0-1, 建议0.4): ") or "0.4")
    
#     # 处理数量设置
#     print("\n3. 处理数量设置")
#     params["max_table_pairs"] = int(input("最大表对数量 (建议20): ") or "20")
#     params["max_llm_candidates"] = int(input("最大LLM处理候选对数量 (建议50): ") or "50")
    
#     # 匹配策略设置
#     print("\n4. 匹配策略设置")
#     params["aggressive_mode"] = input("启用积极匹配模式? (y/n, 建议y): ").lower() == 'y'
#     params["batch_size"] = int(input("批处理大小 (建议3): ") or "3")
    
#     return params


# def calculate_table_similarity(source_schema, target_schema):
#     """计算表级别相似度"""
#     source_name = source_schema.table_name.lower()
#     target_name = target_schema.table_name.lower()
    
#     import difflib
#     name_sim = difflib.SequenceMatcher(None, source_name, target_name).ratio()
    
#     source_desc = source_schema.table_desc.lower() if source_schema.table_desc else ""
#     target_desc = target_schema.table_desc.lower() if target_schema.table_desc else ""
    
#     if source_desc and target_desc:
#         desc_sim = difflib.SequenceMatcher(None, source_desc, target_desc).ratio()
#     else:
#         desc_sim = 0
    
#     return 0.7 * name_sim + 0.3 * desc_sim


# def filter_table_pairs(source_schemas, target_schemas, table_threshold=0.2, max_pairs=20):
#     """筛选表对"""
#     table_similarities = []
    
#     for source_schema in source_schemas:
#         for target_schema in target_schemas:
#             similarity = calculate_table_similarity(source_schema, target_schema)
#             if similarity >= table_threshold:
#                 table_similarities.append((source_schema, target_schema, similarity))
    
#     table_similarities.sort(key=lambda x: x[2], reverse=True)
    
#     if len(table_similarities) > max_pairs:
#         table_similarities = table_similarities[:max_pairs]
    
#     selected_pairs = [(s, t) for s, t, _ in table_similarities]
    
#     print(f"筛选出 {len(selected_pairs)} 对潜在匹配的表")
#     for i, (source, target, sim) in enumerate(table_similarities[:10]):  # 显示前10个
#         print(f"  {i+1}. {source.table_name} <-> {target.table_name} (相似度: {sim:.3f})")
    
#     if len(table_similarities) > 10:
#         print(f"  ... 还有 {len(table_similarities) - 10} 对表")
    
#     return selected_pairs


# def classify_matches_by_confidence(results: List[Dict], matcher) -> Dict[str, List[Dict]]:
#     """根据置信度分类匹配结果"""
#     classified = {
#         "high_confidence": [],
#         "medium_confidence": [],
#         "low_confidence": [],
#         "potential_matches": []
#     }
    
#     for result in results:
#         if not result.get("match", False):
#             continue
            
#         confidence = result.get("confidence", 0)
        
#         if confidence >= matcher.high_confidence_threshold:
#             classified["high_confidence"].append(result)
#         elif confidence >= matcher.medium_confidence_threshold:
#             classified["medium_confidence"].append(result)
#         elif confidence >= matcher.low_confidence_threshold:
#             classified["low_confidence"].append(result)
#         else:
#             classified["potential_matches"].append(result)
    
#     return classified


# def main():
#     """主函数"""
#     parser = argparse.ArgumentParser(description='交互式增强Schema匹配')
#     parser.add_argument('--config', type=str, default='config/config_enhanced.yaml', help='配置文件路径')
#     parser.add_argument('--output', type=str, default='output', help='输出目录')
#     parser.add_argument('--auto', action='store_true', help='使用默认参数，跳过交互式设置')
#     args = parser.parse_args()
    
#     print("=== 交互式增强Schema匹配系统 ===")
#     print("支持动态参数调整和实时阈值修改")
    
#     # 检查配置文件
#     if not os.path.exists(args.config):
#         print(f"错误: 配置文件不存在: {args.config}")
#         sys.exit(1)
    
#     # 交互式参数设置或使用默认值
#     if args.auto:
#         print("使用默认参数运行...")
#         params = {
#             "source_file": "data/源数据字典.xlsx",
#             "target_file": "data/项目匹配字典_列类型注释.xlsx",
#             "max_table_pairs": 20,
#             "max_llm_candidates": 50,
#             "aggressive_mode": True,
#             "batch_size": 3
#         }
#     else:
#         params = interactive_parameter_setup()
    
#     # 检查数据文件
#     for file_key in ["source_file", "target_file"]:
#         if not os.path.exists(params[file_key]):
#             print(f"错误: {file_key}不存在: {params[file_key]}")
#             sys.exit(1)
    
#     # 加载配置
#     with open(args.config, "r", encoding="utf-8") as f:
#         config = yaml.safe_load(f)
    
#     print(f"\n使用配置文件: {args.config}")
#     print(f"源文件: {params['source_file']}")
#     print(f"目标文件: {params['target_file']}")
    
#     # 1. 数据加载
#     print("\n" + "="*50)
#     print("1. 数据加载")
#     print("="*50)
#     start_time = time.time()
#     data_loader = DataLoader()
    
#     try:
#         source_schemas = data_loader.load_excel_dictionary(params["source_file"])
#         target_schemas = data_loader.load_excel_dictionary(params["target_file"])
        
#         print(f"✅ 数据加载完成，耗时: {time.time() - start_time:.2f}秒")
#         print(f"源表数量: {len(source_schemas)}, 目标表数量: {len(target_schemas)}")
        
#     except Exception as e:
#         print(f"❌ 数据加载失败: {e}")
#         return
    
#     # 2. 元数据预处理
#     print("\n" + "="*50)
#     print("2. 元数据预处理")
#     print("="*50)
#     start_time = time.time()
    
#     preprocessor = MetadataPreprocessor(
#         enable_pinyin=config["chinese"]["enable_pinyin"],
#         enable_abbreviation=config["chinese"]["enable_abbreviation"]
#     )
    
#     processed_source_schemas = {}
#     for schema in source_schemas:
#         processed = preprocessor.preprocess_schema(schema)
#         processed_source_schemas[schema.table_name] = processed
    
#     processed_target_schemas = {}
#     for schema in target_schemas:
#         processed = preprocessor.preprocess_schema(schema)
#         processed_target_schemas[schema.table_name] = processed
    
#     print(f"✅ 预处理完成，耗时: {time.time() - start_time:.2f}秒")
    
#     # 3. 表对筛选
#     print("\n" + "="*50)
#     print("3. 表对筛选")
#     print("="*50)
    
#     table_pairs = filter_table_pairs(
#         source_schemas, 
#         target_schemas, 
#         max_pairs=params["max_table_pairs"]
#     )
    
#     # 4. 增强相似度计算
#     print("\n" + "="*50)
#     print("4. 增强相似度计算")
#     print("="*50)
#     start_time = time.time()
    
#     similarity_calculator = EnhancedSimilarityCalculator(
#         char_weight=config["similarity"]["char_weight"],
#         semantic_weight=config["similarity"]["semantic_weight"],
#         struct_weight=config["similarity"]["struct_weight"],
#         pinyin_boost=config["similarity"]["pinyin_boost"]
#     )
    
#     all_candidates = []
    
#     for source_schema, target_schema in table_pairs:
#         print(f"计算表 {source_schema.table_name} 和 {target_schema.table_name} 的相似度...")
        
#         source_processed = processed_source_schemas[source_schema.table_name]
#         target_processed = processed_target_schemas[target_schema.table_name]
        
#         matrix = similarity_calculator.calculate_similarity_matrix(
#             source_processed["fields"],
#             target_processed["fields"]
#         )
        
#         # 使用动态阈值
#         similarity_threshold = params.get("similarity_threshold", config["thresholds"]["similarity_threshold"])
        
#         for i, s_field in enumerate(source_processed["fields"]):
#             for j, t_field in enumerate(target_processed["fields"]):
#                 sim = matrix[i, j]
#                 if sim >= similarity_threshold:
#                     all_candidates.append({
#                         "source_table": source_schema.table_name,
#                         "source_field": s_field["name"],
#                         "target_table": target_schema.table_name,
#                         "target_field": t_field["name"],
#                         "similarity": float(sim)
#                     })
    
#     all_candidates.sort(key=lambda x: x["similarity"], reverse=True)
    
#     print(f"✅ 相似度计算完成，耗时: {time.time() - start_time:.2f}秒")
#     print(f"找到 {len(all_candidates)} 对候选字段匹配（阈值: {similarity_threshold}）")
    
#     # 显示相似度分布
#     high_sim = len([c for c in all_candidates if c["similarity"] >= 0.7])
#     medium_sim = len([c for c in all_candidates if 0.4 <= c["similarity"] < 0.7])
#     low_sim = len([c for c in all_candidates if c["similarity"] < 0.4])
#     print(f"  高相似度(≥0.7): {high_sim}")
#     print(f"  中等相似度(0.4-0.7): {medium_sim}")
#     print(f"  低相似度(<0.4): {low_sim}")
    
#     # 5. 应用匹配规则
#     print("\n" + "="*50)
#     print("5. 应用匹配规则")
#     print("="*50)
    
#     candidate_filter = CandidateFilter(similarity_threshold=similarity_threshold)
#     filtered_candidates = candidate_filter.apply_matching_rules(all_candidates)
    
#     print(f"✅ 应用规则后保留 {len(filtered_candidates)} 对候选匹配")
    
#     # 6. 初始化交互式LLM匹配器
#     print("\n" + "="*50)
#     print("6. 初始化交互式LLM匹配器")
#     print("="*50)
    
#     llm_matcher = InteractiveEnhancedLLMMatcher(config_path=args.config)
    
#     # 应用用户设置的参数
#     if "similarity_threshold" in params:
#         llm_matcher.update_thresholds(similarity_threshold=params["similarity_threshold"])
#     if "high_confidence" in params:
#         llm_matcher.update_thresholds(high_conf=params["high_confidence"])
#     if "medium_confidence" in params:
#         llm_matcher.update_thresholds(medium_conf=params["medium_confidence"])
#     if "low_confidence" in params:
#         llm_matcher.update_thresholds(low_conf=params["low_confidence"])
    
#     llm_matcher.update_matching_settings(
#         batch_size=params["batch_size"],
#         aggressive_mode=params["aggressive_mode"]
#     )
    
#     # 7. LLM匹配
#     print("\n" + "="*50)
#     print("7. 交互式LLM匹配")
#     print("="*50)
#     start_time = time.time()
    
#     matching_results = llm_matcher.batch_process_candidates(
#         filtered_candidates,
#         processed_source_schemas,
#         processed_target_schemas,
#         max_candidates=params["max_llm_candidates"]
#     )
    
#     print(f"✅ LLM匹配完成，耗时: {time.time() - start_time:.2f}秒")
    
#     # 8. 结果分类和处理
#     print("\n" + "="*50)
#     print("8. 结果分类和处理")
#     print("="*50)
    
#     classified_matches = classify_matches_by_confidence(matching_results, llm_matcher)
    
#     result_processor = ResultProcessor(
#         confidence_threshold=llm_matcher.low_confidence_threshold
#     )
    
#     statistics = result_processor.calculate_matching_statistics(
#         matching_results,
#         processed_source_schemas,
#         processed_target_schemas,
#         all_candidates
#     )
    
#     high_confidence_matches = result_processor.process_matching_results(
#         classified_matches["high_confidence"],
#         processed_source_schemas,
#         processed_target_schemas
#     )
    
#     # 9. 保存结果
#     print("\n" + "="*50)
#     print("9. 保存结果")
#     print("="*50)
    
#     os.makedirs(args.output, exist_ok=True)
    
#     high_conf_files = result_processor.save_results(high_confidence_matches, statistics, args.output)
    
#     timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
#     all_matches_file = os.path.join(args.output, f"interactive_enhanced_matches_{timestamp}.xlsx")
    
#     with pd.ExcelWriter(all_matches_file, engine='openpyxl') as writer:
#         for category, matches in classified_matches.items():
#             if matches:
#                 df = pd.DataFrame(matches)
#                 columns = {
#                     "source_table": "源表名",
#                     "source_field": "源字段名", 
#                     "target_table": "目标表名",
#                     "target_field": "目标字段名",
#                     "confidence": "匹配置信度",
#                     "similarity": "特征相似度",
#                     "reason": "匹配理由"
#                 }
#                 existing_columns = {k: v for k, v in columns.items() if k in df.columns}
#                 df_output = df[list(existing_columns.keys())].copy()
#                 df_output.rename(columns=existing_columns, inplace=True)
                
#                 sheet_name = {
#                     "high_confidence": "高置信度匹配",
#                     "medium_confidence": "中等置信度匹配", 
#                     "low_confidence": "低置信度匹配",
#                     "potential_matches": "潜在匹配"
#                 }[category]
                
#                 df_output.to_excel(writer, sheet_name=sheet_name, index=False)
    
#     # 10. 输出结果总结
#     print("\n" + "="*60)
#     print("                  交互式匹配结果总结")
#     print("="*60)
    
#     total_matches = sum(len(matches) for matches in classified_matches.values())
    
#     print(f"\n【使用参数】")
#     print(f"  相似度阈值: {llm_matcher.similarity_threshold}")
#     print(f"  高置信度阈值: {llm_matcher.high_confidence_threshold}")
#     print(f"  中等置信度阈值: {llm_matcher.medium_confidence_threshold}")
#     print(f"  低置信度阈值: {llm_matcher.low_confidence_threshold}")
#     print(f"  积极匹配模式: {'启用' if llm_matcher.enable_aggressive_mode else '禁用'}")
    
#     print(f"\n【匹配结果】")
#     print(f"  总匹配数量: {total_matches}")
#     print(f"  高置信度匹配: {len(classified_matches['high_confidence'])}")
#     print(f"  中等置信度匹配: {len(classified_matches['medium_confidence'])}")
#     print(f"  低置信度匹配: {len(classified_matches['low_confidence'])}")
#     print(f"  潜在匹配: {len(classified_matches['potential_matches'])}")
    
#     print(f"\n【输出文件】")
#     print(f"  高置信度匹配: {high_conf_files['excel']}")
#     print(f"  所有分层匹配结果: {all_matches_file}")
#     print(f"  统计信息: {high_conf_files['statistics']}")
    
#     # 显示匹配结果示例
#     if total_matches > 0:
#         print(f"\n【匹配结果示例】")
#         for category, matches in classified_matches.items():
#             if matches:
#                 category_name = {
#                     "high_confidence": "高置信度匹配",
#                     "medium_confidence": "中等置信度匹配",
#                     "low_confidence": "低置信度匹配", 
#                     "potential_matches": "潜在匹配"
#                 }[category]
                
#                 print(f"\n{category_name} ({len(matches)}个):")
#                 for i, result in enumerate(matches[:2]):
#                     fallback_mark = " [回退]" if result.get("fallback", False) else ""
#                     print(f"  {i+1}. {result['source_table']}.{result['source_field']} <-> "
#                           f"{result['target_table']}.{result['target_field']}{fallback_mark}")
#                     print(f"     置信度: {result['confidence']:.2f}, 相似度: {result.get('similarity', 0):.2f}")
                
#                 if len(matches) > 2:
#                     print(f"     ... 还有 {len(matches) - 2} 个匹配")
#     else:
#         print(f"\n未找到匹配结果")
    
#     print("\n🎉 交互式增强匹配实验完成！")


# if __name__ == "__main__":
#     main()
"""
超稳定Schema匹配脚本 - 解决API频率限制问题
"""
import os
import sys
import yaml
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import time
import argparse
import json
import re
import requests
from tqdm import tqdm
import random
import threading

from src.data.data_loader import DataLoader, SchemaMetadata
from src.data.data_preprocessor import MetadataPreprocessor
from src.features.enhanced_similarity_calculator import EnhancedSimilarityCalculator
from src.matching.candidate_filter import CandidateFilter
from src.matching.result_processor import ResultProcessor


class UltraStableLLMMatcher:
    """超稳定LLM匹配器 - 具有智能重试和速率控制"""
    
    def __init__(self, config_path: str = "config/config_enhanced.yaml"):
        print(f"初始化超稳定LLM匹配器...")
        
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)
        
        self.api_key = self.config["openai"]["api_key"]
        self.api_base_url = self.config["openai"]["api_base_url"]
        self.model = self.config["openai"]["model"]
        self.temperature = self.config["openai"]["temperature"]
        self.max_tokens = self.config["openai"]["max_tokens"]
        
        # 智能速率控制参数
        self.base_delay = 2.0  # 基础延迟（秒）
        self.max_delay = 30.0  # 最大延迟
        self.current_delay = self.base_delay
        self.success_count_for_speedup = 5  # 连续成功次数后加速
        self.consecutive_successes = 0
        
        # 重试机制参数
        self.max_retries = 3
        self.retry_delays = [2, 5, 10]  # 重试间隔
        
        # 缓存
        self.cache_enabled = self.config["system"]["cache_enabled"]
        self._cache = {}
        self._cache_path = "cache/ultra_stable_responses.json"
        if self.cache_enabled:
            os.makedirs(os.path.dirname(self._cache_path), exist_ok=True)
            if os.path.exists(self._cache_path):
                with open(self._cache_path, "r", encoding="utf-8") as f:
                    self._cache = json.load(f)
        
        # 统计信息
        self.total_calls = 0
        self.success_count = 0
        self.fail_count = 0
        self.cache_hit_count = 0
        self.retry_count = 0
        
        # 线程锁，确保API调用串行
        self.api_lock = threading.Lock()
        
        print(f"✅ 超稳定匹配器初始化完成")
        print(f"API配置: {self.api_base_url}")
        print(f"基础延迟: {self.base_delay}秒")
        print(f"最大重试次数: {self.max_retries}")
    
    def _call_api_with_smart_retry(self, prompt: str, field_info: str = "") -> str:
        """智能重试的API调用"""
        with self.api_lock:  # 确保串行调用
            self.total_calls += 1
            
            for attempt in range(self.max_retries + 1):
                try:
                    if attempt > 0:
                        self.retry_count += 1
                        retry_delay = self.retry_delays[min(attempt - 1, len(self.retry_delays) - 1)]
                        # 添加随机因子，避免雷群效应
                        actual_delay = retry_delay + random.uniform(0, 2)
                        print(f"  🔄 第{attempt}次重试 {field_info}，等待{actual_delay:.1f}秒...")
                        time.sleep(actual_delay)
                    
                    # 调用API
                    response = self._make_api_request(prompt)
                    
                    if not response.startswith("错误："):
                        # 成功！
                        self.success_count += 1
                        self.consecutive_successes += 1
                        
                        # 动态调整延迟：连续成功时适度加速
                        if self.consecutive_successes >= self.success_count_for_speedup:
                            self.current_delay = max(1.0, self.current_delay * 0.8)
                            self.consecutive_successes = 0
                            print(f"  ⚡ 调整延迟为: {self.current_delay:.1f}秒")
                        
                        # 成功调用后的标准延迟
                        time.sleep(self.current_delay)
                        return response
                    else:
                        # 失败，重置连续成功计数
                        self.consecutive_successes = 0
                        
                        # 检查是否是频率限制错误
                        if "429" in response or "频率" in response or "rate" in response.lower():
                            # 频率限制，增加延迟
                            self.current_delay = min(self.max_delay, self.current_delay * 2)
                            print(f"  ⚠️  检测到频率限制，增加延迟至: {self.current_delay:.1f}秒")
                        
                        print(f"  ❌ API调用失败 {field_info}: {response[:100]}")
                        
                        if attempt == self.max_retries:
                            # 最后一次重试也失败
                            self.fail_count += 1
                            return response
                            
                except Exception as e:
                    print(f"  ❌ API调用异常 {field_info}: {e}")
                    if attempt == self.max_retries:
                        self.fail_count += 1
                        return f"错误：API调用异常 - {str(e)}"
            
            return "错误：所有重试都失败"
    
    def _make_api_request(self, prompt: str) -> str:
        """实际的API请求"""
        try:
            api_url = f"{self.api_base_url}/chat/completions"
            
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
            
            # 适当增加超时时间
            response = requests.post(api_url, headers=headers, json=data, timeout=45)
            
            if response.status_code == 429:
                return "错误：API频率限制(429)"
            elif response.status_code == 401:
                return "错误：API认证失败(401)"
            elif response.status_code == 403:
                return "错误：API访问被禁止(403)"
            elif response.status_code == 500:
                return "错误：API服务器错误(500)"
            elif response.status_code != 200:
                return f"错误：API调用失败，状态码：{response.status_code}"
            
            try:
                resp_json = response.json()
            except json.JSONDecodeError:
                return "错误：JSON解析失败"
            
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
                    return "错误：响应格式异常"
            else:
                return "错误：响应中没有choices字段"
                
        except requests.exceptions.Timeout:
            return "错误：请求超时"
        except requests.exceptions.ConnectionError:
            return "错误：网络连接失败"
        except Exception as e:
            return f"错误：{str(e)}"
    
    def match_field_pair(self, source_field: Dict, target_field: Dict, similarity: float) -> Dict:
        """匹配单对字段"""
        field_info = f"{source_field['name']} <-> {target_field['name']}"
        
        # 检查缓存
        cache_key = f"ultra_{source_field['name']}_{target_field['name']}_{similarity:.3f}"
        if self.cache_enabled and cache_key in self._cache:
            self.cache_hit_count += 1
            print(f"  💾 缓存命中: {field_info}")
            return self._cache[cache_key]
        
        # 创建简化提示
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
        
        print(f"  🔄 调用API: {field_info}")
        
        # 智能重试API调用
        response = self._call_api_with_smart_retry(prompt, field_info)
        
        if response.startswith("错误："):
            print(f"  🔄 使用智能回退: {field_info}")
            result = self._generate_enhanced_fallback_result(source_field, target_field, similarity, response)
        else:
            print(f"  ✅ API成功: {field_info}")
            result = self._parse_response(response, similarity)
        
        result["similarity"] = similarity
        
        # 缓存结果
        if self.cache_enabled:
            self._cache[cache_key] = result
            if len(self._cache) % 5 == 0:
                self._save_cache()
        
        return result
    
    def _parse_response(self, response: str, similarity: float) -> Dict:
        """解析LLM响应"""
        try:
            match = False
            if any(indicator in response for indicator in ["判断：是", "判断:是", "判断: 是"]):
                match = True
            
            confidence = 0.0
            patterns = [r"置信度：([0-9.]+)", r"置信度:([0-9.]+)", r"置信度: ([0-9.]+)"]
            
            for pattern in patterns:
                match_obj = re.search(pattern, response)
                if match_obj:
                    try:
                        confidence = float(match_obj.group(1))
                        break
                    except ValueError:
                        pass
            
            if confidence == 0.0:
                if match:
                    confidence = max(0.7, similarity + 0.1)
                else:
                    confidence = min(0.4, similarity)
            
            reason_patterns = [r"理由：(.*?)(?=\n|$)", r"理由:(.*?)(?=\n|$)", r"理由: (.*?)(?=\n|$)"]
            reason = ""
            
            for pattern in reason_patterns:
                match_obj = re.search(pattern, response, re.DOTALL)
                if match_obj:
                    reason = match_obj.group(1).strip()
                    break
            
            if not reason:
                reason = response[:100] + "..."
            
            return {
                "match": match,
                "confidence": confidence,
                "reason": reason,
                "llm_response": response
            }
            
        except Exception as e:
            return {
                "match": similarity > 0.6,
                "confidence": similarity,
                "reason": f"解析失败: {e}，基于相似度判断",
                "parse_error": True
            }
    
    def _generate_enhanced_fallback_result(self, source_field: Dict, target_field: Dict, 
                                          similarity: float, error_msg: str) -> Dict:
        """生成增强回退结果"""
        s_name = source_field.get("name", "").lower()
        t_name = target_field.get("name", "").lower()
        s_desc = source_field.get("desc", "").lower()
        t_desc = target_field.get("desc", "").lower()
        
        # 多层次智能回退策略
        
        # 1. 完全匹配
        if s_name == t_name:
            match = True
            confidence = 0.95
            reason = f"字段名完全匹配 (智能回退)"
        
        # 2. 高相似度 + 业务概念匹配
        elif similarity >= 0.8:
            # 检查业务概念
            business_concepts = {
                "id": ["id", "编号", "号码", "代码"],
                "name": ["name", "名称", "姓名", "名字"],
                "time": ["time", "日期", "时间", "date"],
                "create": ["create", "创建", "新建"],
                "update": ["update", "更新", "修改"]
            }
            
            concept_match = False
            for concept, keywords in business_concepts.items():
                s_match = any(kw in s_name or kw in s_desc for kw in keywords)
                t_match = any(kw in t_name or kw in t_desc for kw in keywords)
                if s_match and t_match:
                    concept_match = True
                    break
            
            if concept_match:
                match = True
                confidence = min(0.85, similarity + 0.05)
                reason = f"高相似度({similarity:.2f})+业务概念匹配 (智能回退)"
            else:
                match = True
                confidence = min(0.8, similarity)
                reason = f"高相似度({similarity:.2f})匹配 (智能回退)"
        
        # 3. 中等相似度判断
        elif similarity >= 0.6:
            match = True
            confidence = min(0.75, similarity + 0.05)
            reason = f"中等相似度({similarity:.2f})匹配 (智能回退)"
        
        # 4. 包含关系检查
        elif similarity >= 0.4:
            contain_match = False
            if (s_name and t_name and (s_name in t_name or t_name in s_name)) or \
               (s_desc and t_desc and (s_desc in t_desc or t_desc in s_desc)):
                contain_match = True
            
            if contain_match:
                match = True
                confidence = min(0.7, similarity + 0.1)
                reason = f"字段包含关系+相似度({similarity:.2f}) (智能回退)"
            else:
                match = False
                confidence = similarity
                reason = f"相似度较低({similarity:.2f})，判断为不匹配 (智能回退)"
        
        # 5. 低相似度
        else:
            match = False
            confidence = similarity
            reason = f"相似度过低({similarity:.2f})，判断为不匹配 (智能回退)"
        
        return {
            "match": match,
            "confidence": confidence,
            "reason": reason,
            "fallback": True,
            "api_error": error_msg[:100]
        }
    
    def batch_process_candidates(self, candidate_pairs: List[Dict], 
                                source_schemas: Dict[str, Dict],
                                target_schemas: Dict[str, Dict],
                                max_candidates: int = None) -> List[Dict]:
        """批量处理候选匹配对"""
        
        if max_candidates and len(candidate_pairs) > max_candidates:
            print(f"⚠️  候选对数量({len(candidate_pairs)})超过限制({max_candidates})，只处理前{max_candidates}个")
            candidate_pairs = candidate_pairs[:max_candidates]
        
        results = []
        
        print(f"🚀 开始超稳定批量处理 {len(candidate_pairs)} 对候选匹配...")
        print(f"📊 当前延迟设置: {self.current_delay:.1f}秒")
        
        # 分层处理，优先处理高相似度
        high_sim = [c for c in candidate_pairs if c["similarity"] >= 0.7]
        medium_sim = [c for c in candidate_pairs if 0.4 <= c["similarity"] < 0.7]
        low_sim = [c for c in candidate_pairs if c["similarity"] < 0.4]
        
        print(f"📈 分层统计: 高相似度({len(high_sim)}) | 中等相似度({len(medium_sim)}) | 低相似度({len(low_sim)})")
        
        layer_count = 0
        for candidates, desc in [(high_sim, "高相似度"), (medium_sim, "中等相似度"), (low_sim, "低相似度")]:
            if not candidates:
                continue
            
            layer_count += 1
            print(f"\n🔄 处理第{layer_count}层: {desc}候选对 ({len(candidates)}个)")
            
            for i, candidate in enumerate(candidates):
                try:
                    print(f"  📝 进度: {i+1}/{len(candidates)} - {desc}")
                    
                    source_schema = source_schemas[candidate["source_table"]]
                    source_field = next(f for f in source_schema["fields"] if f["name"] == candidate["source_field"])
                    
                    target_schema = target_schemas[candidate["target_table"]]
                    target_field = next(f for f in target_schema["fields"] if f["name"] == candidate["target_field"])
                    
                    result = self.match_field_pair(source_field, target_field, candidate["similarity"])
                    
                    result["source_table"] = candidate["source_table"]
                    result["source_field"] = candidate["source_field"]
                    result["target_table"] = candidate["target_table"]
                    result["target_field"] = candidate["target_field"]
                    
                    results.append(result)
                    
                    # 显示实时统计
                    if (i + 1) % 5 == 0:
                        current_success_rate = self.success_count / max(1, self.total_calls) * 100
                        print(f"  📊 当前成功率: {current_success_rate:.1f}%, 当前延迟: {self.current_delay:.1f}秒")
                        
                except Exception as e:
                    print(f"  ❌ 处理失败: {candidate}, 错误: {e}")
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
                    results.append(error_result)
        
        # 保存最终缓存
        if self.cache_enabled:
            self._save_cache()
        
        # 打印详细统计
        self._print_detailed_stats()
        
        return results
    
    def _print_detailed_stats(self):
        """打印详细统计信息"""
        print(f"\n" + "="*60)
        print("              超稳定API调用统计")
        print("="*60)
        
        total_processed = self.total_calls + self.cache_hit_count
        success_rate = self.success_count / max(1, self.total_calls) * 100
        cache_rate = self.cache_hit_count / max(1, total_processed) * 100
        
        print(f"📞 总API调用: {self.total_calls}")
        print(f"✅ 成功调用: {self.success_count}")
        print(f"❌ 失败调用: {self.fail_count}")
        print(f"🔄 重试次数: {self.retry_count}")
        print(f"💾 缓存命中: {self.cache_hit_count}")
        print(f"")
        print(f"📊 API成功率: {success_rate:.1f}%")
        print(f"💾 缓存命中率: {cache_rate:.1f}%")
        print(f"⚡ 最终延迟: {self.current_delay:.1f}秒")
        
        if self.fail_count > 0:
            print(f"")
            print(f"⚠️  失败调用使用了智能回退策略")
            print(f"🔄 建议：如需提高成功率，可增加基础延迟时间")
        
        print("="*60)
    
    def _save_cache(self):
        """保存缓存"""
        try:
            with open(self._cache_path, "w", encoding="utf-8") as f:
                json.dump(self._cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存失败: {e}")


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
    
    return 0.7 * name_sim + 0.3 * desc_sim


def filter_table_pairs(source_schemas, target_schemas, table_threshold=0.2, max_pairs=20):
    """筛选表对"""
    table_similarities = []
    
    for source_schema in source_schemas:
        for target_schema in target_schemas:
            similarity = calculate_table_similarity(source_schema, target_schema)
            if similarity >= table_threshold:
                table_similarities.append((source_schema, target_schema, similarity))
    
    table_similarities.sort(key=lambda x: x[2], reverse=True)
    
    if len(table_similarities) > max_pairs:
        table_similarities = table_similarities[:max_pairs]
    
    selected_pairs = [(s, t) for s, t, _ in table_similarities]
    
    print(f"筛选出 {len(selected_pairs)} 对潜在匹配的表")
    
    return selected_pairs


def classify_matches_by_confidence(results: List[Dict]) -> Dict[str, List[Dict]]:
    """根据置信度分类匹配结果"""
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
        
        if confidence >= 0.8:
            classified["high_confidence"].append(result)
        elif confidence >= 0.6:
            classified["medium_confidence"].append(result)
        elif confidence >= 0.4:
            classified["low_confidence"].append(result)
        else:
            classified["potential_matches"].append(result)
    
    return classified


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='超稳定Schema匹配')
    parser.add_argument('--source', type=str, default='data/源数据字典.xlsx', help='源表数据字典文件路径')
    parser.add_argument('--target', type=str, default='data/项目匹配字典_列类型注释.xlsx', help='目标表数据字典文件路径')
    parser.add_argument('--config', type=str, default='config/config_enhanced.yaml', help='配置文件路径')
    parser.add_argument('--output', type=str, default='output', help='输出目录')
    parser.add_argument('--max-pairs', type=int, default=15, help='最大表对数量')
    parser.add_argument('--max-llm', type=int, default=30, help='最大LLM处理候选对数量')
    parser.add_argument('--base-delay', type=float, default=3.0, help='基础API调用延迟（秒）')
    args = parser.parse_args()
    
    print("🚀 === 超稳定Schema匹配系统 ===")
    print("✨ 具有智能重试和动态速率控制")
    print(f"⏱️  基础延迟: {args.base_delay}秒")
    
    # 检查文件
    for file_path, desc in [(args.config, "配置文件"), (args.source, "源数据文件"), (args.target, "目标数据文件")]:
        if not os.path.exists(file_path):
            print(f"❌ 错误: {desc}不存在: {file_path}")
            sys.exit(1)
    
    # 加载配置
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    print(f"📄 使用配置文件: {args.config}")
    
    # 1. 数据加载
    print(f"\n📂 1. 数据加载...")
    start_time = time.time()
    data_loader = DataLoader()
    
    try:
        source_schemas = data_loader.load_excel_dictionary(args.source)
        target_schemas = data_loader.load_excel_dictionary(args.target)
        
        print(f"✅ 数据加载完成，耗时: {time.time() - start_time:.2f}秒")
        print(f"📊 源表数量: {len(source_schemas)}, 目标表数量: {len(target_schemas)}")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 2. 元数据预处理
    print(f"\n🔧 2. 元数据预处理...")
    start_time = time.time()
    
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
    
    print(f"✅ 预处理完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 3. 表对筛选
    print(f"\n📋 3. 智能表对筛选...")
    table_pairs = filter_table_pairs(
        source_schemas, 
        target_schemas, 
        max_pairs=args.max_pairs
    )
    
    # 4. 增强相似度计算
    print(f"\n🧮 4. 增强相似度计算...")
    start_time = time.time()
    
    similarity_calculator = EnhancedSimilarityCalculator(
        char_weight=config["similarity"]["char_weight"],
        semantic_weight=config["similarity"]["semantic_weight"],
        struct_weight=config["similarity"]["struct_weight"],
        pinyin_boost=config["similarity"]["pinyin_boost"]
    )
    
    all_candidates = []
    
    for source_schema, target_schema in table_pairs:
        print(f"  🔄 计算表 {source_schema.table_name} <-> {target_schema.table_name}")
        
        source_processed = processed_source_schemas[source_schema.table_name]
        target_processed = processed_target_schemas[target_schema.table_name]
        
        matrix = similarity_calculator.calculate_similarity_matrix(
            source_processed["fields"],
            target_processed["fields"]
        )
        
        similarity_threshold = config["thresholds"]["similarity_threshold"]
        
        for i, s_field in enumerate(source_processed["fields"]):
            for j, t_field in enumerate(target_processed["fields"]):
                sim = matrix[i, j]
                if sim >= similarity_threshold:
                    all_candidates.append({
                        "source_table": source_schema.table_name,
                        "source_field": s_field["name"],
                        "target_table": target_schema.table_name,
                        "target_field": t_field["name"],
                        "similarity": float(sim)
                    })
    
    all_candidates.sort(key=lambda x: x["similarity"], reverse=True)
    
    print(f"✅ 相似度计算完成，耗时: {time.time() - start_time:.2f}秒")
    print(f"📊 找到 {len(all_candidates)} 对候选字段匹配")
    
    # 5. 应用匹配规则
    print(f"\n📏 5. 应用匹配规则...")
    candidate_filter = CandidateFilter(similarity_threshold=config["thresholds"]["similarity_threshold"])
    filtered_candidates = candidate_filter.apply_matching_rules(all_candidates)
    print(f"✅ 应用规则后保留 {len(filtered_candidates)} 对候选匹配")
    
    # 限制处理数量
    if len(filtered_candidates) > args.max_llm:
        print(f"⚠️  候选匹配对较多，只处理前 {args.max_llm} 个")
        current_candidates = filtered_candidates[:args.max_llm]
    else:
        current_candidates = filtered_candidates
    
    # 6. 超稳定LLM匹配
    print(f"\n🤖 6. 超稳定LLM匹配...")
    start_time = time.time()
    
    # 初始化超稳定匹配器
    llm_matcher = UltraStableLLMMatcher(config_path=args.config)
    llm_matcher.base_delay = args.base_delay  # 应用用户设置的延迟
    llm_matcher.current_delay = args.base_delay
    
    matching_results = llm_matcher.batch_process_candidates(
        current_candidates,
        processed_source_schemas,
        processed_target_schemas,
        max_candidates=args.max_llm
    )
    
    print(f"✅ 超稳定LLM匹配完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 7. 结果分类和处理
    print(f"\n📊 7. 结果分类和处理...")
    
    classified_matches = classify_matches_by_confidence(matching_results)
    
    result_processor = ResultProcessor(confidence_threshold=0.4)
    
    statistics = result_processor.calculate_matching_statistics(
        matching_results,
        processed_source_schemas,
        processed_target_schemas,
        all_candidates
    )
    
    high_confidence_matches = result_processor.process_matching_results(
        classified_matches["high_confidence"],
        processed_source_schemas,
        processed_target_schemas
    )
    
    # 8. 保存结果
    print(f"\n💾 8. 保存结果...")
    
    os.makedirs(args.output, exist_ok=True)
    
    high_conf_files = result_processor.save_results(high_confidence_matches, statistics, args.output)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    all_matches_file = os.path.join(args.output, f"ultra_stable_matches_{timestamp}.xlsx")
    
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
    
    # 9. 输出最终总结
    print(f"\n" + "="*60)
    print("                  超稳定匹配总结")
    print("="*60)
    
    total_matches = sum(len(matches) for matches in classified_matches.values())
    api_success_rate = llm_matcher.success_count / max(1, llm_matcher.total_calls) * 100
    
    print(f"\n🎯 【匹配成果】")
    print(f"  总匹配数量: {total_matches}")
    print(f"  高置信度匹配: {len(classified_matches['high_confidence'])}")
    print(f"  中等置信度匹配: {len(classified_matches['medium_confidence'])}")
    print(f"  低置信度匹配: {len(classified_matches['low_confidence'])}")
    print(f"  潜在匹配: {len(classified_matches['potential_matches'])}")
    
    print(f"\n🔧 【系统性能】")
    print(f"  API成功率: {api_success_rate:.1f}%")
    print(f"  最终延迟: {llm_matcher.current_delay:.1f}秒")
    print(f"  重试次数: {llm_matcher.retry_count}")
    print(f"  缓存命中: {llm_matcher.cache_hit_count}")
    
    print(f"\n📁 【输出文件】")
    print(f"  高置信度匹配: {high_conf_files['excel']}")
    print(f"  所有分层匹配结果: {all_matches_file}")
    print(f"  统计信息: {high_conf_files['statistics']}")
    
    # 显示成功匹配示例
    if total_matches > 0:
        print(f"\n🎉 【匹配结果示例】")
        
        # 显示API成功的匹配
        api_success_matches = [r for r in matching_results if r.get("match") and not r.get("fallback")]
        if api_success_matches:
            print(f"\n✅ API成功匹配 ({len(api_success_matches)}个):")
            for i, result in enumerate(api_success_matches[:3]):
                print(f"  {i+1}. {result['source_table']}.{result['source_field']} <-> "
                      f"{result['target_table']}.{result['target_field']}")
                print(f"     置信度: {result['confidence']:.2f}, 相似度: {result.get('similarity', 0):.2f}")
        
        # 显示智能回退匹配
        fallback_matches = [r for r in matching_results if r.get("match") and r.get("fallback")]
        if fallback_matches:
            print(f"\n🔄 智能回退匹配 ({len(fallback_matches)}个):")
            for i, result in enumerate(fallback_matches[:3]):
                print(f"  {i+1}. {result['source_table']}.{result['source_field']} <-> "
                      f"{result['target_table']}.{result['target_field']}")
                print(f"     置信度: {result['confidence']:.2f}, 相似度: {result.get('similarity', 0):.2f}")
    else:
        print(f"\n⚠️  未找到匹配结果")
    
    print(f"\n🎊 超稳定匹配实验完成！")


if __name__ == "__main__":
    main()