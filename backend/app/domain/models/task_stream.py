#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""任务内部输出流模型。"""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

from .event import Event


class TaskStreamEventRecord(BaseModel):
    """输出流中的业务事件记录。"""

    kind: Literal["event"] = "event"
    event: Event


class TaskRequestStartedRecord(BaseModel):
    """标记某个 chat/resume 请求对应的 run 开始真正消费。"""

    kind: Literal["request_started"] = "request_started"
    request_id: str


class TaskRequestFinishedRecord(BaseModel):
    """标记某个 chat/resume 请求对应的 run 已落出终态。"""

    kind: Literal["request_finished"] = "request_finished"
    request_id: str
    terminal_event_type: Literal["wait", "done", "error"]


class TaskRequestRejectedRecord(BaseModel):
    """标记尚未开始执行的排队请求已被明确拒绝。"""

    kind: Literal["request_rejected"] = "request_rejected"
    request_id: str
    message: str
    error_key: str | None = None


TaskStreamRecord = Annotated[
    Union[
        TaskStreamEventRecord,
        TaskRequestStartedRecord,
        TaskRequestFinishedRecord,
        TaskRequestRejectedRecord,
    ],
    Field(discriminator="kind"),
]
