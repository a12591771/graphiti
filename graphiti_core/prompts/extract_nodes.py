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


class ExtractedEntity(BaseModel):
    name: str = Field(..., description='提取的实体名称')
    entity_type_id: int = Field(
        description='分类的实体类型ID。必须是提供的entity_type_id整数之一。',
    )


class ExtractedEntities(BaseModel):
    extracted_entities: list[ExtractedEntity] = Field(..., description='提取的实体列表')


class MissedEntities(BaseModel):
    missed_entities: list[str] = Field(..., description="未被提取的实体名称")


class EntityClassificationTriple(BaseModel):
    uuid: str = Field(description='实体的UUID')
    name: str = Field(description='实体名称')
    entity_type: str | None = Field(
        default=None, description='实体类型。必须是提供的类型之一或None'
    )


class EntityClassification(BaseModel):
    entity_classifications: list[EntityClassificationTriple] = Field(
        ..., description='实体分类三元组列表。'
    )


class Prompt(Protocol):
    extract_message: PromptVersion
    extract_json: PromptVersion
    extract_text: PromptVersion
    reflexion: PromptVersion
    classify_nodes: PromptVersion
    extract_attributes: PromptVersion


class Versions(TypedDict):
    extract_message: PromptFunction
    extract_json: PromptFunction
    extract_text: PromptFunction
    reflexion: PromptFunction
    classify_nodes: PromptFunction
    extract_attributes: PromptFunction


def extract_message(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """你是一个从对话消息中提取实体节点的AI助手。
    你的主要任务是提取和分类说话者以及对话中提到的其他重要实体。"""

    user_prompt = f"""
<实体类型>
{context['entity_types']}
</实体类型>

<历史消息>
{json.dumps([ep for ep in context['previous_episodes']], indent=2)}
</历史消息>

<当前消息>
{context['episode_content']}
</当前消息>

指令:

给定对话上下文和当前消息，你的任务是提取当前消息中**显式或隐式**提及的**实体节点**。
代词引用如他/她/他们或这/那/那些应该被消歧到引用实体的名称。

1. **说话者提取**: 始终提取说话者（每个对话行中冒号 `:` 前面的部分）作为第一个实体节点。
   - 如果说话者在消息中再次被提及，将两次提及视为**单个实体**。

2. **实体识别**:
   - 提取当前消息中**显式或隐式**提及的所有重要实体、概念或参与者。
   - **排除**仅在之前消息中提及的实体（它们仅用于上下文）。

3. **实体分类**:
   - 使用实体类型中的描述来分类每个提取的实体。
   - 为每个实体分配适当的 `entity_type_id`。

4. **排除项**:
   - 不要提取表示关系或行为的实体。
   - 不要提取日期、时间或其他时间信息——这些将单独处理。

5. **格式化**:
   - 在命名实体时要**明确且无歧义**（例如，在可用时使用全名）。

{context['custom_prompt']}
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_json(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """你是一个从JSON中提取实体节点的AI助手。
    你的主要任务是从JSON文件中提取和分类相关实体。"""

    user_prompt = f"""
<实体类型>
{context['entity_types']}
</实体类型>

<源描述>:
{context['source_description']}
</源描述>
<JSON>
{context['episode_content']}
</JSON>

{context['custom_prompt']}

给定上述源描述和JSON，从提供的JSON中提取相关实体。
对于每个提取的实体，还要根据提供的实体类型及其描述确定其实体类型。
通过提供其entity_type_id来指示分类的实体类型。

指导原则：
1. 始终尝试提取JSON代表的实体。这通常是像"name"或"user"字段这样的内容
2. 不要提取任何包含日期的属性
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_text(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """你是一个从文本中提取实体节点的AI助手。
    你的主要任务是提取和分类说话者以及提供文本中提到的其他重要实体。"""

    user_prompt = f"""
<实体类型>
{context['entity_types']}
</实体类型>

<文本>
{context['episode_content']}
</文本>

给定上述文本，从文本中提取显式或隐式提及的实体。
对于每个提取的实体，还要根据提供的实体类型及其描述确定其实体类型。
通过提供其entity_type_id来指示分类的实体类型。

{context['custom_prompt']}

指导原则：
1. 提取对话中提到的重要实体、概念或参与者。
2. 避免为关系或行为创建节点。
3. 避免为时间信息（如日期、时间或年份）创建节点（这些将稍后添加到边中）。
4. 在节点名称中尽可能明确，使用全名并避免缩写。
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def reflexion(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """你是一个确定给定上下文中哪些实体未被提取的AI助手"""

    user_prompt = f"""
<历史消息>
{json.dumps([ep for ep in context['previous_episodes']], indent=2)}
</历史消息>
<当前消息>
{context['episode_content']}
</当前消息>

<已提取实体>
{context['extracted_entities']}
</已提取实体>

给定上述历史消息、当前消息和已提取实体列表；确定是否有任何实体未被提取。
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def classify_nodes(context: dict[str, Any]) -> list[Message]:
    sys_prompt = """你是一个根据提取实体节点的上下文对实体节点进行分类的AI助手"""

    user_prompt = f"""
    <历史消息>
    {json.dumps([ep for ep in context['previous_episodes']], indent=2)}
    </历史消息>
    <当前消息>
    {context['episode_content']}
    </当前消息>
    
    <已提取实体>
    {context['extracted_entities']}
    </已提取实体>
    
    <实体类型>
    {context['entity_types']}
    </实体类型>
    
    给定上述对话、已提取实体和提供的实体类型及其描述，对已提取的实体进行分类。
    
    指导原则：
    1. 每个实体必须有且仅有一个类型
    2. 只使用提供的实体类型作为类型，不要使用其他类型来分类实体。
    3. 如果提供的实体类型都不能准确分类已提取的节点，则类型应设置为None
"""
    return [
        Message(role='system', content=sys_prompt),
        Message(role='user', content=user_prompt),
    ]


def extract_attributes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='你是一个有用的助手，负责从提供的文本中提取实体属性。请不要转义unicode字符。\n\n提取的任何信息都应该以与写入时相同的语言返回。',
        ),
        Message(
            role='user',
            content=f"""

        <消息>
        {json.dumps(context['previous_episodes'], indent=2)}
        {json.dumps(context['episode_content'], indent=2)}
        </消息>

        根据上述消息和以下实体，基于消息中提供的信息更新实体的任何属性。
        使用提供的属性描述来更好地理解每个属性应该如何确定。

        指导原则：
        1. 如果在当前上下文中找不到实体属性值，不要编造实体属性值。
        2. 仅使用提供的消息和实体来设置属性值。
        3. summary（摘要）属性代表实体的摘要，应该根据消息中关于该实体的新信息进行更新。
           摘要不得超过250个字。
        
        <实体>
        {context['node']}
        </实体>
        """,
        ),
    ]


versions: Versions = {
    'extract_message': extract_message,
    'extract_json': extract_json,
    'extract_text': extract_text,
    'reflexion': reflexion,
    'classify_nodes': classify_nodes,
    'extract_attributes': extract_attributes,
}