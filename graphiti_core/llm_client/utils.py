"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import json
import logging
import re
from time import time

from graphiti_core.embedder.client import EmbedderClient

logger = logging.getLogger(__name__)


def clean_json_response(raw_output: str) -> str:
    """
    清理LLM返回的JSON响应，处理常见的格式问题：
    1. 移除 ```json 和 ``` 包装
    2. 移除前后空白字符
    3. 处理多余的换行符
    
    Args:
        raw_output: LLM返回的原始文本
        
    Returns:
        清理后的JSON字符串
    """
    if not raw_output or not raw_output.strip():
        return raw_output
    
    # 移除前后空白字符
    cleaned = raw_output.strip()
    
    # 移除 ```json 开头和 ``` 结尾
    if cleaned.startswith('```json'):
        cleaned = cleaned[7:]  # 移除 '```json'
    elif cleaned.startswith('```'):
        cleaned = cleaned[3:]  # 移除 '```'
    
    if cleaned.endswith('```'):
        cleaned = cleaned[:-3]  # 移除结尾的 '```'
    
    # 移除前后空白字符
    cleaned = cleaned.strip()
    
    # 日志记录清理结果
    if cleaned != raw_output.strip():
        logger.debug(f"JSON response cleaned: removed markdown wrapper")
    
    return cleaned


def safe_json_loads(raw_output: str) -> dict:
    """
    安全地解析JSON，包含自动清理功能
    
    Args:
        raw_output: LLM返回的原始文本
        
    Returns:
        解析后的字典对象
        
    Raises:
        json.JSONDecodeError: 如果无法解析JSON
    """
    try:
        # 首先尝试直接解析
        return json.loads(raw_output)
    except json.JSONDecodeError:
        # 如果失败，尝试清理后再解析
        cleaned = clean_json_response(raw_output)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON even after cleaning:")
            logger.error(f"Original: {raw_output[:200]}...")
            logger.error(f"Cleaned: {cleaned[:200]}...")
            raise e


async def generate_embedding(embedder: EmbedderClient, text: str):
    start = time()

    text = text.replace('\n', ' ')
    embedding = await embedder.create(input_data=[text])

    end = time()
    logger.debug(f'embedded text of length {len(text)} in {end - start} ms')

    return embedding
