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


RuntimeInput = Annotated[
    Union[
        MessageEvent,
        ResumeInput,
    ],
    Field(discriminator="type"),
]
