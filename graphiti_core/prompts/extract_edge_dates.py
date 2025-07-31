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


class EdgeDates(BaseModel):
    valid_at: str | None = Field(
        None,
        description='边事实描述的关系成立或建立的日期和时间。YYYY-MM-DDTHH:MM:SS.SSSSSSZ或null。',
    )
    invalid_at: str | None = Field(
        None,
        description='边事实描述的关系停止为真或结束的日期和时间。YYYY-MM-DDTHH:MM:SS.SSSSSSZ或null。',
    )


class Prompt(Protocol):
    v1: PromptVersion


class Versions(TypedDict):
    v1: PromptFunction


def v1(context: dict[str, Any]) -> list[Message]:
    return [
        Message(
            role='system',
            content='你是一个为图边提取日期时间信息的AI助手，只关注与边事实中描述的关系建立或变化直接相关的日期。',
        ),
        Message(
            role='user',
            content=f"""
            <历史消息>
            {context['previous_episodes']}
            </历史消息>
            <当前消息>
            {context['current_episode']}
            </当前消息>
            <参考时间戳>
            {context['reference_timestamp']}
            </参考时间戳>
            
            <事实>
            {context['edge_fact']}
            </事实>

            重要：只有当时间信息是提供的事实的一部分时才提取时间信息。否则忽略提到的时间。确保尽力确定日期，如果只提到相对时间（例如10年前，2分钟前）基于提供的参考时间戳。
            如果关系不是跨越性质的，但你仍然能够确定日期，只设置valid_at。
            定义：
            - valid_at：边事实描述的关系成立或建立的日期和时间。
            - invalid_at：边事实描述的关系停止为真或结束的日期和时间。

            任务：
            分析对话并确定是否有作为边事实一部分的日期。只有当日期明确与关系本身的形成或改变相关时才设置日期。

            指导原则：
            1. 使用ISO 8601格式（YYYY-MM-DDTHH:MM:SS.SSSSSSZ）表示日期时间。
            2. 在确定valid_at和invalid_at日期时，使用参考时间戳作为当前时间。
            3. 如果事实以现在时书写，使用参考时间戳作为valid_at日期
            4. 如果没有找到建立或改变关系的时间信息，将字段留为null。
            5. 不要从相关事件推断日期。只使用直接陈述建立或改变关系的日期。
			6. 对于直接与关系相关的相对时间提及，基于参考时间戳计算实际日期时间。
            7. 如果只提到日期而没有具体时间，对该日期使用00:00:00（午夜）。
            8. 如果只提到年份，使用该年的1月1日00:00:00。
            9. 始终包含时区偏移（如果没有提到具体时区，对UTC使用Z）。
            10. 讨论某事不再为真的事实应该根据被否定的事实何时成立来设置valid_at。
            """,
        ),
    ]


versions: Versions = {'v1': v1}