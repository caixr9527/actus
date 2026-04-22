#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/2/24 18:41
@Author : caixiaorong01@outlook.com
@File   : event.py
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any, Dict, Self, Type, Literal, List, Union, get_args

from pydantic import BaseModel, Field, ConfigDict, field_serializer

from app.domain.models import (
    Event,
    File,
    ExecutionStatus,
    WaitEvent,
    ToolEvent,
    ToolEventStatus,
    PlanEvent,
    PlanEventStatus,
    StepEvent,
    TextStreamChannel,
)
from app.domain.services.runtime.normalizers import normalize_event_payload

class BaseEventData(BaseModel):
    """基础事件数据"""
    event_id: Optional[str] = None  # 事件id
    created_at: datetime = Field(default_factory=datetime.now)  # 事件时间

    @field_serializer("created_at", when_used="json")
    def serialize_created_at(self, value: datetime) -> int:
        """将 created_at 序列化为秒级时间戳。"""
        return int(value.timestamp())

    @classmethod
    def base_event_data(cls, event: Event) -> Dict[str, Any]:
        """类方法，用于将事件Domain模型转换成基础事件数据字典"""
        return {
            "event_id": event.id,
            "created_at": int(event.created_at.timestamp()),
        }

    @classmethod
    def from_event(cls, event: Event) -> Self:
        """从事件Domain模型中构建基础事件数据"""
        return cls(
            **cls.base_event_data(event),
            **event.model_dump(mode="json", exclude={"id", "type", "created_at"}),
        )


class BaseSSEEvent(BaseModel):
    """基础流式事件数据类型"""
    event: str  # 事件类型
    data: BaseEventData  # 数据

    @classmethod
    def from_event(cls, event: Event) -> Self:
        """将事件Domain模型转换成基础流式事件"""
        # 获取data字段的类型注解，如果不存在则默认使用BaseEventData
        data_class: Type[BaseEventData] = cls.__annotations__.get("data", BaseEventData)

        # 构造SSE事件对象，设置事件类型和数据
        return cls(
            event=event.type,
            data=data_class.from_event(event),
        )


class CommonEventData(BaseEventData):
    """通用事件数据，让结构允许填充额外的数据"""
    model_config = ConfigDict(extra="allow")


class CommonSSEEvent(BaseSSEEvent):
    """通用事件"""
    event: str
    data: CommonEventData


class MessageEventData(BaseEventData):
    """消息事件数据"""
    role: Literal["user", "assistant"] = "assistant"
    message: str = ""
    attachments: List[File] = Field(default_factory=list)
    stage: Literal["intermediate", "final"] = "intermediate"


class MessageSSEEvent(BaseSSEEvent):
    """流式消息事件数据响应结构"""
    event: Literal["message"] = "message"
    data: MessageEventData

    @classmethod
    def from_event(cls, event: Event) -> Self:
        return cls(
            data=MessageEventData(
                **BaseEventData.base_event_data(event),
                role=event.role,
                message=event.message,
                attachments=event.attachments,
                stage=event.stage,
            )
        )


class TitleEventData(BaseEventData):
    """标题事件数据"""
    title: str


class TitleSSEEvent(BaseSSEEvent):
    """标题流式事件"""
    event: Literal["title"] = "title"
    data: TitleEventData


class StepEventData(BaseEventData):
    """步骤事件数据"""
    id: str  # 步骤id
    status: ExecutionStatus  # 步骤执行状态
    description: str  # 步骤描述
    outcome: Optional["StepOutcomeData"] = None


class StepOutcomeData(BaseModel):
    """步骤结果数据"""
    done: bool = False
    summary: str = ""
    produced_artifacts: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)
    facts_learned: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    next_hint: Optional[str] = None
    reused_from_run_id: Optional[str] = None
    reused_from_step_id: Optional[str] = None


class StepSSEEvent(BaseSSEEvent):
    """步骤流式事件"""
    event: Literal["step"] = "step"
    data: StepEventData

    @classmethod
    def from_event(cls, event: StepEvent) -> Self:
        return cls(
            data=StepEventData(
                **BaseEventData.base_event_data(event),
                status=event.step.status,
                id=event.step.id,
                description=event.step.description,
                outcome=(
                    StepOutcomeData.model_validate(event.step.outcome.model_dump(mode="json"))
                    if event.step.outcome is not None
                    else None
                ),
            )
        )


class PlanEventData(BaseEventData):
    """计划事件数据"""
    steps: List[StepEventData]
    title: str = ""
    goal: str = ""
    message: str = ""
    language: str = ""
    status: PlanEventStatus = PlanEventStatus.CREATED
    plan_status: ExecutionStatus = ExecutionStatus.PENDING


class PlanSSEEvent(BaseSSEEvent):
    """计划流式事件"""
    event: Literal["plan"] = "plan"
    data: PlanEventData

    @classmethod
    def from_event(cls, event: PlanEvent) -> Self:
        return cls(
            data=PlanEventData(
                **BaseEventData.base_event_data(event),
                steps=[
                    StepEventData(
                        **BaseEventData.base_event_data(event),
                        id=step.id,
                        status=step.status,
                        description=step.description,
                        outcome=(
                            StepOutcomeData.model_validate(step.outcome.model_dump(mode="json"))
                            if step.outcome is not None
                            else None
                        ),
                    )
                    for step in event.plan.steps
                ],
                title=event.plan.title,
                goal=event.plan.goal,
                message=event.plan.message,
                language=event.plan.language,
                status=event.status,
                plan_status=event.plan.status,
            )
        )


class ToolEventData(BaseEventData):
    """工具事件数据"""
    tool_call_id: str  # 工具调用id
    name: str  # 工具箱名字
    status: ToolEventStatus  # 工具状态
    function: str  # 工具名字
    args: Dict[str, Any]  # 工具参数
    content: Optional[Any] = None  # 工具调用结果


class ToolSSEEvent(BaseSSEEvent):
    """工具流式事件"""
    event: Literal["tool"] = "tool"
    data: ToolEventData

    @classmethod
    def from_event(cls, event: ToolEvent) -> Self:
        return cls(
            data=ToolEventData(
                **BaseEventData.base_event_data(event),
                tool_call_id=event.tool_call_id,
                name=event.tool_name,
                status=event.status,
                function=event.function_name,
                args=event.function_args,
                content=event.tool_content,
            )
        )


class DoneSSEEvent(BaseSSEEvent):
    """停止流式事件"""
    event: Literal["done"] = "done"


class WaitEventData(BaseEventData):
    """等待事件数据"""

    interrupt_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)


class WaitSSEEvent(BaseSSEEvent):
    """等待人类输入流式事件"""
    event: Literal["wait"] = "wait"
    data: WaitEventData

    @classmethod
    def from_event(cls, event: WaitEvent) -> Self:
        return cls(
            data=WaitEventData(
                **BaseEventData.base_event_data(event),
                interrupt_id=event.interrupt_id,
                payload=dict(event.payload or {}),
            )
        )


class TextStreamStartEventData(BaseEventData):
    """文本流开始事件数据。"""

    stream_id: str
    channel: TextStreamChannel
    run_id: Optional[str] = None
    session_id: Optional[str] = None
    stage: Literal["planner", "summary", "final"]
    is_replay: bool = True


class TextStreamStartSSEEvent(BaseSSEEvent):
    """文本流开始 SSE 事件。"""

    event: Literal["text_stream_start"] = "text_stream_start"
    data: TextStreamStartEventData


class TextStreamDeltaEventData(BaseEventData):
    """文本流增量事件数据。"""

    stream_id: str
    channel: TextStreamChannel
    text: str = ""
    sequence: int = 0


class TextStreamDeltaSSEEvent(BaseSSEEvent):
    """文本流增量 SSE 事件。"""

    event: Literal["text_stream_delta"] = "text_stream_delta"
    data: TextStreamDeltaEventData


class TextStreamEndEventData(BaseEventData):
    """文本流结束事件数据。"""

    stream_id: str
    channel: TextStreamChannel
    full_text_length: int = 0
    reason: Literal["completed", "cancelled", "error"] = "completed"


class TextStreamEndSSEEvent(BaseSSEEvent):
    """文本流结束 SSE 事件。"""

    event: Literal["text_stream_end"] = "text_stream_end"
    data: TextStreamEndEventData


class ErrorEventData(BaseEventData):
    """错误事件数据"""
    error: str
    error_key: Optional[str] = None
    error_params: Optional[Dict[str, Any]] = None


class ErrorSSEEvent(BaseSSEEvent):
    """错误流式事件"""
    event: Literal["error"] = "error"
    data: ErrorEventData


AgentSSEEvent = Union[
    CommonSSEEvent,
    MessageSSEEvent,
    TitleSSEEvent,
    StepSSEEvent,
    PlanSSEEvent,
    ToolSSEEvent,
    DoneSSEEvent,
    ErrorSSEEvent,
    WaitSSEEvent,
    TextStreamStartSSEEvent,
    TextStreamDeltaSSEEvent,
    TextStreamEndSSEEvent,
]


@dataclass
class EventMapping:
    """事件映射数据类，用于存储事件映射信息，涵盖流式事件类型、数据类、事件类型字符串"""
    sse_event_class: Type[BaseSSEEvent]
    data_class: Type[BaseEventData]
    event_type: str


class EventMapper:
    """事件映射类，利用Python自身提供的自省机制，将业务逻辑中的Event转换成适合流式传输的AgentSSEEvent"""
    # 缓存映射(type: EventMapping)
    _cache_mapping: Optional[Dict[str, EventMapping]] = None

    @staticmethod
    def _get_event_type_mapping() -> Dict[str, EventMapping]:
        """通过反射动态构建从事件类型字符串到AgentSSEEvent的映射"""
        # 检查缓存是否已存在，如果存在则直接返回缓存的映射
        if EventMapper._cache_mapping is not None:
            return EventMapper._cache_mapping

        # 获取所有AgentSSEEvent的联合类型参数
        sse_event_classes = get_args(AgentSSEEvent)
        mapping = {}

        # 遍历每个SSE事件类，构建事件类型到EventMapping的映射
        for sse_event_class in sse_event_classes:
            # 跳过基础类BaseSSEEvent
            if sse_event_class == BaseSSEEvent:
                continue

            # 检查类是否有注解且包含'event'字段
            if hasattr(sse_event_class, "__annotations__") and "event" in sse_event_class.__annotations__:
                event_field = sse_event_class.__annotations__["event"]

                # 确保event字段有类型参数且至少有一个参数
                if hasattr(event_field, "__args__") and len(event_field.__args__) > 0:
                    event_type = event_field.__args__[0]

                    # 尝试获取'data'字段的类型注解
                    data_class = None
                    if hasattr(sse_event_class, "__annotations__") and "data" in sse_event_class.__annotations__:
                        data_class = sse_event_class.__annotations__["data"]

                    # 构建并存储EventMapping对象
                    mapping[event_type] = EventMapping(
                        sse_event_class=sse_event_class,
                        data_class=data_class,
                        event_type=event_type
                    )

        # 缓存映射结果并返回
        EventMapper._cache_mapping = mapping
        return mapping

    @staticmethod
    def event_to_sse_event(event: Event) -> AgentSSEEvent:
        """将领域事件转换为Agent流式事件模型"""
        # live SSE 与历史回放统一吃同一份干净事件，避免再出现事件层 outcome 旁路。
        event = normalize_event_payload(event)
        # 获取事件类型到SSE事件类的映射关系
        event_type_mapping = EventMapper._get_event_type_mapping()

        # 根据事件类型查找对应的事件映射配置
        event_mapping = event_type_mapping.get(event.type)

        # 如果找到匹配的事件映射，则使用对应SSE事件类构建流式事件
        if event_mapping:
            sse_event = event_mapping.sse_event_class.from_event(event)
        else:
            # 如果未找到匹配的事件映射，则使用通用SSE事件类进行转换
            sse_event = CommonSSEEvent.from_event(event)
        return sse_event

    @staticmethod
    def events_to_sse_events(events: List[Event]) -> List[AgentSSEEvent]:
        """将领域事件模型列表转换为SSE流式事件列表"""
        return list(filter(lambda x: x is not None, [
            EventMapper.event_to_sse_event(event) for event in events
        ]))
