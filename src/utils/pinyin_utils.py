"""
拼音处理工具模块
"""
import re
from pypinyin import pinyin, Style


def convert_to_pinyin(text: str, delimiter: str = '', with_tone: bool = False) -> str:
    """
    将中文文本转换为拼音
    
    Args:
        text: 待转换的文本
        delimiter: 拼音间的分隔符
        with_tone: 是否包含声调
    
    Returns:
        转换后的拼音字符串
    """
    if not text:
        return ""
    
    style = Style.NORMAL if not with_tone else Style.TONE
    
    # 分离英文字符和中文字符
    parts = []
    current_part = []
    is_previous_chinese = None
    
    for char in text:
        is_chinese = bool(re.match(r'[\u4e00-\u9fa5]', char))
        
        # 如果字符类型变化，则切换部分
        if is_previous_chinese is not None and is_chinese != is_previous_chinese:
            if is_previous_chinese:
                # 将中文部分转为拼音
                py_list = pinyin(''.join(current_part), style=style)
                py_text = delimiter.join([item[0] for item in py_list])
                parts.append(py_text)
            else:
                # 将英文部分直接添加
                parts.append(''.join(current_part))
            current_part = []
        
        current_part.append(char)
        is_previous_chinese = is_chinese
    
    # 处理最后一部分
    if current_part:
        if is_previous_chinese:
            py_list = pinyin(''.join(current_part), style=style)
            py_text = delimiter.join([item[0] for item in py_list])
            parts.append(py_text)
        else:
            parts.append(''.join(current_part))
    
    return ''.join(parts)


def get_pinyin_abbr(text: str) -> str:
    """
    获取中文文本的拼音首字母缩写
    
    Args:
        text: 待处理的文本
    
    Returns:
        拼音首字母缩写
    """
    if not text:
        return ""
    
    # 对中文字符，获取拼音首字母；对非中文字符，保持不变
    result = ""
    
    for char in text:
        if re.match(r'[\u4e00-\u9fa5]', char):
            # 中文字符取拼音首字母
            py = pinyin(char, style=Style.FIRST_LETTER)
            result += py[0][0].upper()
        elif re.match(r'[a-zA-Z0-9]', char):
            # 字母数字保持不变
            result += char.upper()
    
    return result


def pinyin_similarity(cn_text: str, en_field: str) -> float:
    """
    计算中文文本与英文字段的拼音相似度
    
    Args:
        cn_text: 中文文本
        en_field: 英文字段名
    
    Returns:
        相似度得分 (0-1)
    """
    if not cn_text or not en_field:
        return 0.0
    
    # 全拼映射
    full_pinyin = convert_to_pinyin(cn_text).lower()
    en_field = en_field.lower()
    
    # 计算字符级相似度
    full_pinyin_sim = string_similarity(full_pinyin, en_field)
    
    # 首字母缩写映射
    abbr_pinyin = get_pinyin_abbr(cn_text).lower()
    abbr_sim = string_similarity(abbr_pinyin, en_field)
    
    return max(full_pinyin_sim, abbr_sim)


def string_similarity(str1: str, str2: str) -> float:
    """
    计算两个字符串的相似度
    
    Args:
        str1: 第一个字符串
        str2: 第二个字符串
    
    Returns:
        相似度得分 (0-1)
    """
    if not str1 or not str2:
        return 0.0
    
    # 转为小写
    str1 = str1.lower()
    str2 = str2.lower()
    
    # 计算Levenshtein距离
    try:
        from Levenshtein import ratio
        return ratio(str1, str2)
    except ImportError:
        # 如果没有Levenshtein库，使用difflib
        import difflib
        return difflib.SequenceMatcher(None, str1, str2).ratio()