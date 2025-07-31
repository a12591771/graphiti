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
from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class Edge(BaseModel):
    relation_type: str = Field(..., description='事实谓词使用大写蛇形命名法（FACT_PREDICATE_IN_SCREAMING_SNAKE_CASE）')
    source_entity_id: int = Field(..., description='事实的源实体ID。')
    target_entity_id: int = Field(..., description='事实的目标实体ID。')
    fact: str = Field(..., description='')
    valid_at: str | None = Field(
        None,
        description='边事实描述的关系成立或建立的日期和时间。使用ISO 8601格式（YYYY-MM-DDTHH:MM:SS.SSSSSSZ）',
    )
    invalid_at: str | None = Field(
        None,
        description='边事实描述的关系停止为真或结束的日期和时间。使用ISO 8601格式（YYYY-MM-DDTHH:MM:SS.SSSSSSZ）',
    )


class ExtractedEdges(BaseModel):
    edges: list[Edge]


class MissingFacts(BaseModel):
    missing_facts: list[str] = Field(..., description="未被提取的事实")


class Prompt(Protocol):
    edge: PromptVersion
    reflexion: PromptVersion
    extract_attributes: PromptVersion


class Versions(TypedDict):
    edge: PromptFunction
    reflexion: PromptFunction
    extract_attributes: PromptFunction


def edge(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='你是一个从文本中提取事实三元组的专业事实提取器。'
            '1. 提取的事实三元组还应该提取相关的日期信息。'
            '2. 将当前时间视为当前消息发送的时间。所有时间信息都应该相对于这个时间提取。'
            '重要：你必须直接返回纯JSON格式，不要使用任何markdown代码块格式（如```json），不要添加任何额外的文本。',
        ),
        Message(
            role='user',
            content=f"""
<事实类型>
{context['edge_types']}
</事实类型>

<历史消息>
{json.dumps([ep for ep in context['previous_episodes']], indent=2)}
</历史消息>

<当前消息>
{context['episode_content']}
</当前消息>

<实体>
{context['nodes']} 
</实体>

<参考时间>
{context['reference_time']}  # ISO 8601 (UTC); 用于解析相对时间提及
</参考时间>

# 任务
基于当前消息提取给定实体之间的所有事实关系。
只提取满足以下条件的事实：
- 涉及实体列表中的两个不同实体，
- 在当前消息中明确陈述或无歧义暗示，
    并且可以表示为知识图中的边。
- 事实类型提供了最重要的事实类型列表，确保提取这些类型的事实
- 事实类型不是详尽的列表，即使不符合事实类型之一，也要提取消息中的所有事实
- 每个事实类型都包含其fact_type_signature，表示源和目标实体类型。

你可以使用历史消息中的信息仅用于消歧引用或支持连续性。


{context['custom_prompt']}

# 提取规则

1. 只输出主语和宾语都匹配实体中ID的事实。
2. 每个事实必须涉及两个**不同的**实体。
3. 使用大写蛇形命名法字符串作为`relation_type`（例如，FOUNDED、WORKS_AT）。
4. 不要输出重复或语义冗余的事实。
5. `fact_text`应该引用或密切释义原始源句子。
6. 使用`REFERENCE_TIME`来解析模糊或相对的时间表达（例如，"上周"）。
7. **不要**虚构或从无关事件推断时间边界。

# 日期时间规则

- 使用带"Z"后缀的ISO 8601格式（UTC）（例如，2025-04-30T00:00:00Z）。
- 如果事实是持续进行的（现在时），将`valid_at`设置为REFERENCE_TIME。
- 如果表达了变化/终止，将`invalid_at`设置为相关时间戳。
- 如果没有明确或可解析的时间陈述，则将两个字段都留为`null`。
- 如果只提到日期（没有时间），假设为00:00:00。
- 如果只提到年份，使用1月1日00:00:00。
        """,
        ),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """你是一个确定给定上下文中哪些事实未被提取的AI助手"""

    user_prompt = f"""
<历史消息>
{json.dumps([ep for ep in context['previous_episodes']], indent=2)}
</历史消息>
<当前消息>
{context['episode_content']}
</当前消息>

<已提取实体>
{context['nodes']}
</已提取实体>

<已提取事实>
{context['extracted_facts']}
</已提取事实>

给定上述消息、已提取实体列表和已提取事实列表；
确定是否有任何事实未被提取。
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='你是一个有用的助手，从提供的文本中提取事实属性。',
        ),
        Message(
            role='user',
            content=f"""

        <消息>
        {json.dumps(context['episode_content'], indent=2)}
        </消息>
        <参考时间>
        {context['reference_time']}
        </参考时间>

        给定上述消息、其参考时间和以下事实，根据消息中提供的信息更新其任何属性。使用提供的属性描述来更好地理解每个属性应该如何确定。

        指导原则：
        1. 如果在当前上下文中找不到实体属性值，不要虚构实体属性值。
        2. 仅使用提供的消息和事实来设置属性值。

        <事实>
        {context['fact']}
        </事实>
        """,
        ),
    ]


versions: Versions = {
    'edge': edge,
    'reflexion': reflexion,
    'extract_attributes': extract_attributes,
}