#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/17 16:38
@Author : caixiaorong01@outlook.com
@File   : task.py
"""
from abc import ABC, abstractmethod
from typing import Protocol, Optional
from app.domain.external.message_queue import MessageQueue


class TaskRunner(ABC):
    """任务执行器抽象类"""

    @abstractmethod
    async def invoke(self, task: "Task") -> None:
        """执行任务"""
        raise NotImplementedError("invoke未实现")

    @abstractmethod
    async def destroy(self) -> None:
        """销毁任务执行器"""
        raise NotImplementedError("destroy未实现")

    @abstractmethod
    async def on_done(self, task: "Task") -> None:
        """任务完成回调"""
        raise NotImplementedError("on_done未实现")


class Task(Protocol):
    """任务抽象类"""

    async def invoke(self) -> None:
        """任务执行方法"""
        ...

    async def cancel(self) -> bool:
        """取消任务并等待运行态收敛完成。"""
        ...

    @property
    def input_stream(self) -> MessageQueue:
        """任务输入流"""
        ...

    @property
    def output_stream(self) -> MessageQueue:
        """任务输出流"""
        ...

    @property
    def id(self) -> None:
        """任务ID"""
        ...

    @property
    def done(self) -> bool:
        """任务是否完成"""
        ...

    @classmethod
    def get(cls, task_id: str) -> Optional["Task"]:
        """获取任务"""
        ...

    @classmethod
    def create(cls, task_runner: TaskRunner) -> "Task":
        """创建任务"""
        ...

    @classmethod
    async def destroy(cls) -> None:
        """销毁任务"""
        ...
