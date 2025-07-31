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


class EdgeDuplicate(BaseModel):
    duplicate_facts: list[int] = Field(
        ...,
        description='任何重复事实的ID列表。如果没有找到重复事实，默认为空列表。',
    )
    contradicted_facts: list[int] = Field(
        ...,
        description='应该被无效化的事实ID列表。如果没有事实应该被无效化，列表应该为空。',
    )
    fact_type: str = Field(..., description='提供的事实类型之一或DEFAULT')


class UniqueFact(BaseModel):
    uuid: str = Field(..., description='事实的唯一标识符')
    fact: str = Field(..., description='唯一边的事实')


class UniqueFacts(BaseModel):
    unique_facts: list[UniqueFact]


class Prompt(Protocol):
    edge: PromptVersion
    edge_list: PromptVersion
    resolve_edge: PromptVersion


class Versions(TypedDict):
    edge: PromptFunction
    edge_list: PromptFunction
    resolve_edge: PromptFunction


def edge(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='你是一个有用的助手，从边列表中去重边。',
        ),
        Message(
            role='user',
            content=f"""
        给定以下上下文，确定新边是否代表现有边列表中的任何边。

        <现有边>
        {json.dumps(context['related_edges'], indent=2)}
        </现有边>

        <新边>
        {json.dumps(context['extracted_edges'], indent=2)}
        </新边>
        
        任务：
        如果新边代表与现有边中任何边相同的事实信息，将重复事实的ID作为duplicate_facts列表的一部分返回。
        如果新边不是任何现有边的重复，返回空列表。

        指导原则：
        1. 事实不需要完全相同就是重复，它们只需要表达相同的信息。
        """,
        ),
    ]


def edge_list(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='你是一个有用的助手，从边列表中去重边。',
        ),
        Message(
            role='user',
            content=f"""
        给定以下上下文，在事实列表中找到所有重复项：

        事实：
        {json.dumps(context['edges'], indent=2)}

        任务：
        如果事实中的任何事实是另一个事实的重复，返回一个带有其中一个uuid的新事实。

        指导原则：
        1. 相同或近似相同的事实是重复的
        2. 如果事实由相似的句子表示，它们也是重复的
        3. 事实通常会讨论相同实体之间的相同或相似关系
        4. 最终列表应该只有唯一的事实。如果3个事实都是彼此的重复，响应中应该只有其中一个事实
        """,
        ),
    ]


def resolve_edge(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='你是一个有用的助手，从事实列表中去重事实并确定新事实与哪些现有事实矛盾。',
        ),
        Message(
            role='user',
            content=f"""
        <新事实>
        {context['new_edge']}
        </新事实>
        
        <现有事实>
        {context['existing_edges']}
        </现有事实>
        <事实无效化候选>
        {context['edge_invalidation_candidates']}
        </事实无效化候选>
        
        <事实类型>
        {context['edge_types']}
        </事实类型>
        

        任务：
        如果新事实代表现有事实中一个或多个的相同事实信息，返回重复事实的idx。
        包含关键差异的相似信息的事实不应该被标记为重复。
        如果新事实不是任何现有事实的重复，返回空列表。
        
        给定预定义的事实类型，确定新事实是否应该被分类为这些类型之一。
        返回事实类型作为fact_type，如果新事实不是事实类型之一，则返回DEFAULT。
        
        基于提供的事实无效化候选和新事实，确定新事实与哪些现有事实矛盾。
        返回包含新事实矛盾的所有事实idx的列表。
        如果没有矛盾的事实，返回空列表。

        指导原则：
        1. 一些事实可能非常相似，但会有关键差异，特别是围绕事实中的数值。
            不要将这些事实标记为重复。
        """,
        ),
    ]


versions: Versions = {'edge': edge, 'edge_list': edge_list, 'resolve_edge': resolve_edge}