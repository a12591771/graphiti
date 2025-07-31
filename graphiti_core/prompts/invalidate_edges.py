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

from typing import Any, Protocol, TypedDict

from pydantic import BaseModel, Field

from .models import Message, PromptFunction, PromptVersion


class InvalidatedEdges(BaseModel):
    contradicted_facts: list[int] = Field(
        ...,
        description='应该被无效化的事实ID列表。如果没有事实应该被无效化，列表应该为空。',
    )


class Prompt(Protocol):
    v1: PromptVersion
    v2: PromptVersion


class Versions(TypedDict):
    v1: PromptFunction
    v2: PromptFunction


def v1(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='你是一个AI助手，帮助确定知识图中的哪些关系应该仅基于较新信息中的明确矛盾而被无效化。',
        ),
        Message(
            role='user',
            content=f"""
               基于提供的现有边和新边及其时间戳，确定哪些关系（如果有）应该由于较新边中的矛盾或更新而被标记为过期。
               使用边的开始和结束日期来确定哪些边要被标记为过期。
                只有在有明确证据表明关系不再为真时，才将关系标记为无效。
                不要仅仅因为关系在剧集中没有提到就无效化关系。你可以使用当前剧集和之前的剧集以及每个边的事实来理解关系的上下文。

                历史剧集：
                {context['previous_episodes']}

                当前剧集：
                {context['current_episode']}

                现有边（按时间戳排序，最新的在前）：
                {context['existing_edges']}

                新边：
                {context['new_edges']}

                每个边格式为："UUID | 源节点 - 边名称 - 目标节点（事实：边事实），开始日期（结束日期，可选）)"
            """,
        ),
    ]


def v2(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='你是一个确定哪些事实相互矛盾的AI助手。',
        ),
        Message(
            role='user',
            content=f"""
               基于提供的现有事实和新事实，确定新事实与哪些现有事实矛盾。
               返回包含新事实矛盾的所有事实ID的列表。
               如果没有矛盾的事实，返回空列表。

                <现有事实>
                {context['existing_edges']}
                </现有事实>

                <新事实>
                {context['new_edge']}
                </新事实>
            """,
        ),
    ]


versions: Versions = {'v1': v1, 'v2': v2}