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


class NodeDuplicate(BaseModel):
    id: int = Field(..., description='实体的整数ID')
    duplicate_idx: int = Field(
        ...,
        description='重复实体的索引。如果没有找到重复实体，默认为-1。',
    )
    name: str = Field(
        ...,
        description='实体名称。应该是实体最完整和描述性的名称。不要在实体名称中包含任何JSON格式（如{}）。',
    )
    duplicates: list[int] = Field(
        ...,
        description='与上述ID实体重复的所有实体的索引。',
    )


class NodeResolutions(BaseModel):
    entity_resolutions: list[NodeDuplicate] = Field(..., description='已解析节点列表')


class Prompt(Protocol):
    node: PromptVersion
    node_list: PromptVersion
    nodes: PromptVersion


class Versions(TypedDict):
    node: PromptFunction
    node_list: PromptFunction
    nodes: PromptFunction


def node(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='你是一个有用的助手，确定新实体是否是任何现有实体的重复。',
        ),
        Message(
            role='user',
            content=f"""
        <历史消息>
        {json.dumps([ep for ep in context['previous_episodes']], indent=2)}
        </历史消息>
        <当前消息>
        {context['episode_content']}
        </当前消息>
        <新实体>
        {json.dumps(context['extracted_node'], indent=2)}
        </新实体>
        <实体类型描述>
        {json.dumps(context['entity_type_description'], indent=2)}
        </实体类型描述>

        <现有实体>
        {json.dumps(context['existing_nodes'], indent=2)}
        </现有实体>
        
        给定上述现有实体及其属性、消息和历史消息；确定从对话中提取的新实体是否是现有实体之一的重复实体。
        
        只有当实体指向*相同的现实世界对象或概念*时，才应该被认为是重复的。
        语义等价：如果existing_entities中的描述性标签明确指向上下文中的命名实体，则将它们视为重复。

        不要将实体标记为重复，如果：
        - 它们相关但不同。
        - 它们有相似的名称或目的，但指向不同的实例或概念。

         任务：
         1. 将`new_entity`与`existing_entities`中的每个项目进行比较。
         2. 如果它指向相同的现实世界对象或概念，收集其索引。
         3. 设`duplicate_idx` = 第一个收集的索引，如果没有则为-1。
         4. 设`duplicates` = 所有收集索引的列表（如果没有则为空列表）。
        
        还要返回新实体的全名（无论它是新实体的名称、它重复的节点的名称，还是两者的组合）。
        """,
        ),
    ]


def nodes(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='你是一个有用的助手，确定从对话中提取的实体是否是现有实体的重复。',
        ),
        Message(
            role='user',
            content=f"""
        <历史消息>
        {json.dumps([ep for ep in context['previous_episodes']], indent=2)}
        </历史消息>
        <当前消息>
        {context['episode_content']}
        </当前消息>
        
        
        以下每个实体都是从当前消息中提取的。
        实体中的每个实体都表示为具有以下结构的JSON对象：
        {{
            id: 实体的整数ID,
            name: "实体名称",
            entity_type: "实体的本体分类",
            entity_type_description: "实体类型代表什么的描述",
            duplication_candidates: [
                {{
                    idx: 候选实体的整数索引,
                    name: "候选实体名称",
                    entity_type: "候选实体的本体分类",
                    ...<附加属性>
                }}
            ]
        }}
        
        <实体>
        {json.dumps(context['extracted_nodes'], indent=2)}
        </实体>
        
        <现有实体>
        {json.dumps(context['existing_nodes'], indent=2)}
        </现有实体>

        对于上述每个实体，确定该实体是否是任何现有实体的重复。

        只有当实体指向*相同的现实世界对象或概念*时，才应该被认为是重复的。

        不要将实体标记为重复，如果：
        - 它们相关但不同。
        - 它们有相似的名称或目的，但指向不同的实例或概念。

        任务：
        你的响应将是一个名为entity_resolutions的列表，其中包含每个实体的一个条目。
        
        对于每个实体，返回实体的ID作为id，实体的名称作为name，duplicate_idx作为整数，以及duplicates作为列表。
        
        - 如果一个实体是现有实体之一的重复，返回它重复的候选的idx。
        - 如果一个实体不是现有实体之一的重复，返回-1作为duplication_idx
        """,
        ),
    ]


def node_list(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='你是一个有用的助手，从节点列表中去重节点。',
        ),
        Message(
            role='user',
            content=f"""
        给定以下上下文，去重节点列表：

        节点：
        {json.dumps(context['nodes'], indent=2)}

        任务：
        1. 将节点分组，使所有重复节点都在相同的uuid列表中
        2. 所有重复的uuid应该组合在同一个列表中
        3. 还要返回一个新的摘要，将摘要合成为一个新的简短摘要

        指导原则：
        1. 节点列表中的每个uuid在你的响应中应该恰好出现一次
        2. 如果一个节点没有重复，它应该在响应中出现在只有一个uuid的列表中

        以以下格式的JSON对象响应：
        {{
            "nodes": [
                {{
                    "uuids": ["5d643020624c42fa9de13f97b1b3fa39", "与5d643020624c42fa9de13f97b1b3fa39重复的节点"],
                    "summary": "出现在名称列表中的节点摘要的简要摘要。"
                }}
            ]
        }}
        """,
        ),
    ]


versions: Versions = {'node': node, 'node_list': node_list, 'nodes': nodes}