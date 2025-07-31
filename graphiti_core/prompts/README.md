# Graphiti 中文提示词库

这是Graphiti项目的中文版本提示词库，专为中文环境和中文大模型优化设计。

## 主要特点

- **完整翻译**: 所有提示词内容已翻译为中文
- **结构兼容**: 保持与原英文版本相同的API接口
- **字段保留**: 所有Pydantic模型字段名称保持英文，确保代码兼容性
- **术语统一**: 专业术语使用一致的中文表达

## 快速开始

### 导入使用

```python
# 使用中文版本提示词
from graphiti.graphiti_core.prompts_zh import prompt_library

# API使用方式与英文版本完全相同
context = {
    'entity_types': '...',
    'previous_episodes': [...],
    'episode_content': '...',
    'custom_prompt': '...'
}

# 调用中文版本的节点提取提示词
messages = prompt_library.extract_nodes.extract_message(context)
```

### 模块说明

| 模块 | 功能 | 说明 |
|------|------|------|
| `extract_nodes` | 实体提取 | 从文本中识别和提取实体，支持对话、JSON、纯文本 |
| `extract_edges` | 关系提取 | 提取实体间的关系和事实 |
| `dedupe_nodes` | 节点去重 | 识别和合并重复的实体节点 |
| `dedupe_edges` | 边去重 | 处理重复的关系和事实 |
| `summarize_nodes` | 节点总结 | 生成实体的摘要信息 |
| `invalidate_edges` | 边失效 | 处理过时或矛盾的关系 |
| `extract_edge_dates` | 时间提取 | 提取关系的时间信息 |
| `eval` | 系统评估 | 评估和验证系统性能 |

## 翻译原则

1. **保留英文字段名**: 确保与现有代码兼容
2. **翻译描述内容**: 提升中文大模型理解效果
3. **统一专业术语**: 保持一致的中文表达
4. **本地化标签**: 将XML标签翻译为中文

## 与英文版本的差异

- 所有用户可见的文本内容都已翻译为中文
- Pydantic模型的字段名称保持英文不变
- 函数名称和类名保持不变
- API接口完全兼容

## 注意事项

- 确保使用支持中文的大语言模型
- 输入的上下文数据可以是中英文混合
- 输出的结构化数据字段名仍为英文
- 建议根据实际使用效果进行fine-tuning