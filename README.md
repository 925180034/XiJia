# 基于元数据的Schema匹配系统

本系统实现了一个基于元数据的Schema匹配框架，专注于SMD(仅元数据)场景，结合传统特征相似度计算和OpenAI API的LLM语义理解能力，特别针对中英文混合环境和拼音缩写等复杂等价关系进行了优化。

## 系统特点

- **跨语言匹配**: 支持中英文混合环境下的表结构匹配
- **拼音处理**: 识别中文拼音与英文字段的映射关系，包括全拼和首字母缩写
- **多维度相似度**: 结合字符级、语义级和结构级相似度计算
- **LLM语义匹配**: 利用OpenAI API进行高精度语义等价判断
- **仅元数据匹配**: 在没有实例数据的情况下完成表结构匹配

## 安装与使用

### 环境要求

- Python 3.8+
- 依赖库：详见requirements.txt

### 安装步骤

1. 克隆仓库
2. 安装依赖: `pip install -r requirements.txt`
3. 更新配置: 编辑 `config/config.yaml` 文件，配置OpenAI API密钥和其他参数

### 使用方法

```bash
python main.py --source 源数据字典.xlsx --target 项目匹配字典.xlsx --config config/config.yaml --output output
```

## 配置说明

`config/config.yaml` 文件中包含相似度计算权重、阈值设置、OpenAI API配置等参数。

## 工作流程

1. **数据加载**: 加载源表和目标表的元数据
2. **元数据预处理**: 进行中文分词、拼音转换等处理
3. **特征计算**: 计算字符级、语义级和结构级相似度
4. **候选对筛选**: 基于相似度阈值筛选候选匹配对
5. **LLM语义匹配**: 利用OpenAI API进行精确的语义等价判断
6. **结果处理**: 处理匹配结果，应用一致性约束，输出最终匹配
