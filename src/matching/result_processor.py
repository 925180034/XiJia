# """
# 结果处理模块
# """
# import os
# import json
# import pandas as pd
# from typing import Dict, List, Tuple, Any


# class ResultProcessor:
#     """匹配结果处理类"""
    
#     def __init__(self, confidence_threshold: float = 0.7):
#         """
#         初始化结果处理器
        
#         Args:
#             confidence_threshold: 置信度阈值
#         """
#         self.confidence_threshold = confidence_threshold
    
#     def process_matching_results(self, 
#                                 matching_results: List[Dict], 
#                                 source_schemas: Dict[str, Dict],
#                                 target_schemas: Dict[str, Dict]) -> List[Dict]:
#         """
#         处理匹配结果
        
#         Args:
#             matching_results: LLM匹配结果列表
#             source_schemas: 源表元数据字典
#             target_schemas: 目标表元数据字典
            
#         Returns:
#             处理后的最终匹配结果
#         """
#         # 按置信度排序
#         sorted_results = sorted(matching_results, key=lambda x: x["confidence"], reverse=True)
        
#         # 应用置信度阈值
#         filtered_results = [r for r in sorted_results if r["match"] and r["confidence"] >= self.confidence_threshold]
        
#         # 应用一对一约束
#         final_results = self._apply_one_to_one_constraint(filtered_results)
        
#         # 丰富结果信息
#         enriched_results = self._enrich_results(final_results, source_schemas, target_schemas)
        
#         return enriched_results
    
#     def _apply_one_to_one_constraint(self, results: List[Dict]) -> List[Dict]:
#         """
#         应用一对一约束
        
#         Args:
#             results: 匹配结果列表
            
#         Returns:
#             应用一对一约束后的结果
#         """
#         final_results = []
#         used_source_fields = set()
#         used_target_fields = set()
        
#         for result in results:
#             source_key = (result["source_table"], result["source_field"])
#             target_key = (result["target_table"], result["target_field"])
            
#             # 检查是否已使用
#             if source_key in used_source_fields or target_key in used_target_fields:
#                 continue
            
#             # 添加到结果
#             final_results.append(result)
#             used_source_fields.add(source_key)
#             used_target_fields.add(target_key)
        
#         return final_results
    
#     def _enrich_results(self, 
#                        results: List[Dict], 
#                        source_schemas: Dict[str, Dict],
#                        target_schemas: Dict[str, Dict]) -> List[Dict]:
#         """
#         丰富结果信息
        
#         Args:
#             results: 匹配结果列表
#             source_schemas: 源表元数据字典
#             target_schemas: 目标表元数据字典
            
#         Returns:
#             丰富后的结果
#         """
#         enriched_results = []
        
#         for result in results:
#             enriched = result.copy()
            
#             # 源表信息
#             source_schema = source_schemas.get(result["source_table"])
#             if source_schema:
#                 source_field = next((f for f in source_schema["fields"] if f["name"] == result["source_field"]), None)
#                 if source_field:
#                     enriched["source_field_desc"] = source_field.get("desc", "")
#                     enriched["source_field_type"] = source_field.get("type", "")
            
#             # 目标表信息
#             target_schema = target_schemas.get(result["target_table"])
#             if target_schema:
#                 target_field = next((f for f in target_schema["fields"] if f["name"] == result["target_field"]), None)
#                 if target_field:
#                     enriched["target_field_desc"] = target_field.get("desc", "")
#                     enriched["target_field_type"] = target_field.get("type", "")
            
#             # 计算匹配依据说明
#             enriched["matching_basis"] = self._generate_matching_basis(enriched)
            
#             enriched_results.append(enriched)
        
#         return enriched_results
    
#     def _generate_matching_basis(self, result: Dict) -> str:
#         """
#         生成匹配依据说明
        
#         Args:
#             result: 匹配结果
            
#         Returns:
#             匹配依据说明
#         """
#         basis = []
        
#         # 从理由中提取关键信息
#         reason = result.get("reason", "")
#         if reason:
#             # 简化理由，只保留关键部分
#             import re
#             # 移除多余的空格和换行
#             reason = re.sub(r'\s+', ' ', reason).strip()
#             # 截断过长的理由
#             if len(reason) > 100:
#                 reason = reason[:100] + "..."
#             basis.append(reason)
        
#         # 添加相似度信息
#         similarity = result.get("similarity", 0)
#         basis.append(f"特征相似度: {similarity:.2f}")
        
#         # 添加置信度信息
#         confidence = result.get("confidence", 0)
#         basis.append(f"LLM置信度: {confidence:.2f}")
        
#         return "；".join(basis)
    
#     def save_results(self, results: List[Dict], output_path: str = "output") -> Dict[str, str]:
#         """
#         保存匹配结果
        
#         Args:
#             results: 匹配结果列表
#             output_path: 输出目录
            
#         Returns:
#             输出文件路径字典
#         """
#         os.makedirs(output_path, exist_ok=True)
        
#         # 创建结果文件路径
#         timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
#         json_path = os.path.join(output_path, f"matching_results_{timestamp}.json")
#         excel_path = os.path.join(output_path, f"matching_results_{timestamp}.xlsx")
        
#         # 保存JSON格式
#         with open(json_path, "w", encoding="utf-8") as f:
#             json.dump(results, f, ensure_ascii=False, indent=2)
        
#         # 创建DataFrame
#         df = pd.DataFrame(results)
        
#         # 选择要保存的列并重命名
#         columns = {
#             "source_table": "源表名",
#             "source_field": "源字段名",
#             "source_field_desc": "源字段描述",
#             "target_table": "目标表名",
#             "target_field": "目标字段名",
#             "target_field_desc": "目标字段描述",
#             "confidence": "匹配置信度",
#             "matching_basis": "匹配依据"
#         }
        
#         # 选择存在的列
#         existing_columns = {k: v for k, v in columns.items() if k in df.columns}
        
#         # 选择并重命名列
#         df_output = df[list(existing_columns.keys())].copy()
#         df_output.rename(columns=existing_columns, inplace=True)
        
#         # 保存Excel格式
#         df_output.to_excel(excel_path, index=False)
        
#         return {
#             "json": json_path,
#             "excel": excel_path
#         }
"""
结果处理模块 - 增强统计版
"""
import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Any
from collections import defaultdict


class ResultProcessor:
    """匹配结果处理类"""
    
    def __init__(self, confidence_threshold: float = 0.7):
        """
        初始化结果处理器
        
        Args:
            confidence_threshold: 置信度阈值
        """
        self.confidence_threshold = confidence_threshold
    
    def process_matching_results(self, 
                                matching_results: List[Dict], 
                                source_schemas: Dict[str, Dict],
                                target_schemas: Dict[str, Dict]) -> List[Dict]:
        """
        处理匹配结果
        
        Args:
            matching_results: LLM匹配结果列表
            source_schemas: 源表元数据字典
            target_schemas: 目标表元数据字典
            
        Returns:
            处理后的最终匹配结果
        """
        # 按置信度排序
        sorted_results = sorted(matching_results, key=lambda x: x["confidence"], reverse=True)
        
        # 应用置信度阈值
        filtered_results = [r for r in sorted_results if r["match"] and r["confidence"] >= self.confidence_threshold]
        
        # 应用一对一约束
        final_results = self._apply_one_to_one_constraint(filtered_results)
        
        # 丰富结果信息
        enriched_results = self._enrich_results(final_results, source_schemas, target_schemas)
        
        return enriched_results
    
    def calculate_matching_statistics(self, 
                                    matching_results: List[Dict],
                                    source_schemas: Dict[str, Dict],
                                    target_schemas: Dict[str, Dict],
                                    candidate_pairs: List[Dict] = None) -> Dict[str, Any]:
        """
        计算匹配统计信息
        
        Args:
            matching_results: 匹配结果列表
            source_schemas: 源表元数据字典
            target_schemas: 目标表元数据字典
            candidate_pairs: 候选匹配对列表
            
        Returns:
            统计信息字典
        """
        statistics = {}
        
        # 基础统计
        statistics["总源表数量"] = len(source_schemas)
        statistics["总目标表数量"] = len(target_schemas)
        
        # 计算总字段数
        total_source_fields = sum(len(schema["fields"]) for schema in source_schemas.values())
        total_target_fields = sum(len(schema["fields"]) for schema in target_schemas.values())
        statistics["总源字段数量"] = total_source_fields
        statistics["总目标字段数量"] = total_target_fields
        
        # 参与匹配的表统计
        involved_source_tables = set()
        involved_target_tables = set()
        
        if candidate_pairs:
            for candidate in candidate_pairs:
                involved_source_tables.add(candidate["source_table"])
                involved_target_tables.add(candidate["target_table"])
        elif matching_results:
            for result in matching_results:
                involved_source_tables.add(result["source_table"])
                involved_target_tables.add(result["target_table"])
        
        statistics["参与匹配的源表数量"] = len(involved_source_tables)
        statistics["参与匹配的目标表数量"] = len(involved_target_tables)
        
        # 参与匹配的字段统计
        involved_source_fields = set()
        involved_target_fields = set()
        
        if candidate_pairs:
            for candidate in candidate_pairs:
                involved_source_fields.add((candidate["source_table"], candidate["source_field"]))
                involved_target_fields.add((candidate["target_table"], candidate["target_field"]))
        elif matching_results:
            for result in matching_results:
                involved_source_fields.add((result["source_table"], result["source_field"]))
                involved_target_fields.add((result["target_table"], result["target_field"]))
        
        statistics["参与匹配的源字段数量"] = len(involved_source_fields)
        statistics["参与匹配的目标字段数量"] = len(involved_target_fields)
        
        # 匹配对统计
        if candidate_pairs:
            statistics["候选匹配对数量"] = len(candidate_pairs)
        else:
            statistics["候选匹配对数量"] = len(matching_results)
        
        # 成功匹配统计
        successful_matches = [r for r in matching_results if r.get("match", False)]
        high_confidence_matches = [r for r in successful_matches if r.get("confidence", 0) >= self.confidence_threshold]
        
        statistics["LLM判断为匹配的字段对数量"] = len(successful_matches)
        statistics["高置信度匹配的字段对数量"] = len(high_confidence_matches)
        
        # 匹配率统计
        if total_source_fields > 0:
            statistics["源字段匹配率"] = f"{len(high_confidence_matches) / total_source_fields * 100:.2f}%"
        if total_target_fields > 0:
            statistics["目标字段匹配率"] = f"{len(high_confidence_matches) / total_target_fields * 100:.2f}%"
        
        # 按表统计匹配情况
        table_match_stats = defaultdict(int)
        for result in high_confidence_matches:
            table_pair = f"{result['source_table']} <-> {result['target_table']}"
            table_match_stats[table_pair] += 1
        
        statistics["按表对统计的匹配数量"] = dict(table_match_stats)
        
        # 置信度分布统计
        confidence_ranges = {
            "0.9-1.0": 0,
            "0.8-0.9": 0,
            "0.7-0.8": 0,
            "0.6-0.7": 0,
            "0.5-0.6": 0,
            "0.0-0.5": 0
        }
        
        for result in matching_results:
            confidence = result.get("confidence", 0)
            if confidence >= 0.9:
                confidence_ranges["0.9-1.0"] += 1
            elif confidence >= 0.8:
                confidence_ranges["0.8-0.9"] += 1
            elif confidence >= 0.7:
                confidence_ranges["0.7-0.8"] += 1
            elif confidence >= 0.6:
                confidence_ranges["0.6-0.7"] += 1
            elif confidence >= 0.5:
                confidence_ranges["0.5-0.6"] += 1
            else:
                confidence_ranges["0.0-0.5"] += 1
        
        statistics["置信度分布"] = confidence_ranges
        
        return statistics
    
    def _apply_one_to_one_constraint(self, results: List[Dict]) -> List[Dict]:
        """
        应用一对一约束
        
        Args:
            results: 匹配结果列表
            
        Returns:
            应用一对一约束后的结果
        """
        final_results = []
        used_source_fields = set()
        used_target_fields = set()
        
        for result in results:
            source_key = (result["source_table"], result["source_field"])
            target_key = (result["target_table"], result["target_field"])
            
            # 检查是否已使用
            if source_key in used_source_fields or target_key in used_target_fields:
                continue
            
            # 添加到结果
            final_results.append(result)
            used_source_fields.add(source_key)
            used_target_fields.add(target_key)
        
        return final_results
    
    def _enrich_results(self, 
                       results: List[Dict], 
                       source_schemas: Dict[str, Dict],
                       target_schemas: Dict[str, Dict]) -> List[Dict]:
        """
        丰富结果信息
        
        Args:
            results: 匹配结果列表
            source_schemas: 源表元数据字典
            target_schemas: 目标表元数据字典
            
        Returns:
            丰富后的结果
        """
        enriched_results = []
        
        for result in results:
            enriched = result.copy()
            
            # 源表信息
            source_schema = source_schemas.get(result["source_table"])
            if source_schema:
                source_field = next((f for f in source_schema["fields"] if f["name"] == result["source_field"]), None)
                if source_field:
                    enriched["source_field_desc"] = source_field.get("desc", "")
                    enriched["source_field_type"] = source_field.get("type", "")
                    enriched["source_table_desc"] = source_schema.get("table_desc", "")
            
            # 目标表信息
            target_schema = target_schemas.get(result["target_table"])
            if target_schema:
                target_field = next((f for f in target_schema["fields"] if f["name"] == result["target_field"]), None)
                if target_field:
                    enriched["target_field_desc"] = target_field.get("desc", "")
                    enriched["target_field_type"] = target_field.get("type", "")
                    enriched["target_table_desc"] = target_schema.get("table_desc", "")
            
            # 计算匹配依据说明
            enriched["matching_basis"] = self._generate_matching_basis(enriched)
            
            enriched_results.append(enriched)
        
        return enriched_results
    
    def _generate_matching_basis(self, result: Dict) -> str:
        """
        生成匹配依据说明
        
        Args:
            result: 匹配结果
            
        Returns:
            匹配依据说明
        """
        basis = []
        
        # 从理由中提取关键信息
        reason = result.get("reason", "")
        if reason:
            # 简化理由，只保留关键部分
            import re
            # 移除多余的空格和换行
            reason = re.sub(r'\s+', ' ', reason).strip()
            # 截断过长的理由
            if len(reason) > 100:
                reason = reason[:100] + "..."
            basis.append(reason)
        
        # 添加相似度信息
        similarity = result.get("similarity", 0)
        basis.append(f"特征相似度: {similarity:.2f}")
        
        # 添加置信度信息
        confidence = result.get("confidence", 0)
        basis.append(f"LLM置信度: {confidence:.2f}")
        
        return "；".join(basis)
    
    def save_results(self, results: List[Dict], statistics: Dict[str, Any], output_path: str = "output") -> Dict[str, str]:
        """
        保存匹配结果和统计信息
        
        Args:
            results: 匹配结果列表
            statistics: 统计信息
            output_path: 输出目录
            
        Returns:
            输出文件路径字典
        """
        os.makedirs(output_path, exist_ok=True)
        
        # 创建结果文件路径
        timestamp = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
        json_path = os.path.join(output_path, f"matching_results_{timestamp}.json")
        excel_path = os.path.join(output_path, f"matching_results_{timestamp}.xlsx")
        stats_path = os.path.join(output_path, f"matching_statistics_{timestamp}.json")
        
        # 保存统计信息到JSON
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        
        # 保存结果到JSON格式
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 创建Excel工作簿，包含多个工作表
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # 匹配结果工作表
            if results:
                # 创建DataFrame
                df = pd.DataFrame(results)
                
                # 选择要保存的列并重命名
                columns = {
                    "source_table": "源表名",
                    "source_field": "源字段名",
                    "source_field_desc": "源字段描述",
                    "source_field_type": "源字段类型",
                    "target_table": "目标表名",
                    "target_field": "目标字段名",
                    "target_field_desc": "目标字段描述",
                    "target_field_type": "目标字段类型",
                    "confidence": "匹配置信度",
                    "similarity": "特征相似度",
                    "matching_basis": "匹配依据"
                }
                
                # 选择存在的列
                existing_columns = {k: v for k, v in columns.items() if k in df.columns}
                
                # 选择并重命名列
                df_output = df[list(existing_columns.keys())].copy()
                df_output.rename(columns=existing_columns, inplace=True)
                
                # 保存匹配结果
                df_output.to_excel(writer, sheet_name='匹配结果', index=False)
            
            # 统计信息工作表
            stats_data = []
            for key, value in statistics.items():
                if isinstance(value, dict):
                    # 处理嵌套字典（如按表对统计）
                    for sub_key, sub_value in value.items():
                        stats_data.append({
                            "统计项目": f"{key} - {sub_key}",
                            "数值": sub_value
                        })
                else:
                    stats_data.append({
                        "统计项目": key,
                        "数值": value
                    })
            
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_excel(writer, sheet_name='统计信息', index=False)
        
        return {
            "json": json_path,
            "excel": excel_path,
            "statistics": stats_path
        }