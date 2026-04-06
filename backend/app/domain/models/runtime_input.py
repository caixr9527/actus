#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行时输入模型。"""
from typing import Any, Annotated, Union, Literal

from pydantic import BaseModel, Field

from .event import MessageEvent


class ResumeInput(BaseModel):
    """恢复 LangGraph interrupt 的内部输入事件。"""

    type: Literal["resume"] = "resume"
    value: Any = None


class ContinueCancelledTaskInput(BaseModel):
    """继续已取消任务的内部控制输入。"""

    type: Literal["continue_cancelled_task"] = "continue_cancelled_task"


RuntimeInputPayload = Annotated[
    Union[
        MessageEvent,
        ResumeInput,
        ContinueCancelledTaskInput,
    ],
    Field(discriminator="type"),
]


class RuntimeInput(BaseModel):
    """任务输入包络。

    request_id 是 service 层 chat 请求的稳定边界标识。
    TaskRunner 只按 request_id 串行消费输入，Service 也据此等待属于自己的 run 收敛。
    """

    request_id: str
    payload: RuntimeInputPayload
