#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/18 14:29
@Author : caixiaorong01@outlook.com
@File   : redis_stream_task.py
"""
import asyncio
import logging
import uuid
from typing import Optional, Dict

from app.domain.external import Task, TaskRunner, MessageQueue
from app.infrastructure.external.message_queue import RedisStreamMessageQueue

logger = logging.getLogger(__name__)


class RedisStreamTask(Task):
    """Redis Stream任务执行器"""

    _task_registry: Dict[str, "RedisStreamTask"] = {}

    def __init__(self, task_runner: TaskRunner):
        self._task_runner = task_runner
        self._id = str(uuid.uuid4())
        self._execution_task: Optional[asyncio.Task] = None

        input_stream_name = f"task:input:{self._id}"
        output_stream_name = f"task:output:{self._id}"

        self._input_stream = RedisStreamMessageQueue(input_stream_name)
        self._output_stream = RedisStreamMessageQueue(output_stream_name)

        RedisStreamTask._task_registry[self._id] = self

    def _cleanup_registry(self) -> None:
        """清除缓存"""
        if self._id in RedisStreamTask._task_registry:
            del RedisStreamTask._task_registry[self._id]
            logger.info(f"清除任务缓存: {self._id}")

    def _on_task_done(self) -> None:
        """任务完成回调"""
        if self._task_runner:
            asyncio.create_task(self._task_runner.on_done(self))

        self._cleanup_registry()

    async def _execute_task(self) -> None:

        try:
            await self._task_runner.invoke(self)
        except asyncio.CancelledError:
            logger.info(f"取消执行任务: {self._id}")
            raise
        except Exception as e:
            logger.error(f"执行任务失败: {self._id}, 错误信息: {e}")
        finally:
            self._on_task_done()

    async def invoke(self) -> None:
        """任务执行方法"""
        if self.done:
            self._execution_task = asyncio.create_task(self._execute_task())
            logger.info(f"开始执行任务: {self._id}")

    def cancel(self) -> bool:
        """任务取消方法"""
        if not self.done and self._execution_task:
            self._execution_task.cancel()
            logger.info(f"取消任务: {self._id}")
            self._cleanup_registry()
            return True

        self._cleanup_registry()
        return False

    @property
    def input_stream(self) -> MessageQueue:
        """任务输入流"""
        return self._input_stream

    @property
    def output_stream(self) -> MessageQueue:
        """任务输出流"""
        return self._output_stream

    @property
    def id(self) -> str:
        """任务ID"""
        return self._id

    @property
    def done(self) -> bool:
        """任务是否完成"""
        if self._execution_task is None:
            return True
        return self._execution_task.done()

    @classmethod
    def get(cls, task_id: str) -> Optional["Task"]:
        """获取任务"""
        return RedisStreamTask._task_registry.get(task_id)

    @classmethod
    def create(cls, task_runner: TaskRunner) -> "Task":
        """创建任务"""
        return cls(task_runner)

    @classmethod
    async def destroy(cls) -> None:
        """销毁任务"""
        # 先对任务注册表做快照，避免 cancel() 触发 _cleanup_registry() 时
        # 在迭代过程中修改字典导致 RuntimeError。
        tasks = list(cls._task_registry.values())

        # 第一阶段：取消任务执行，尽快停止运行态。
        for task in tasks:
            try:
                task.cancel()
            except Exception as e:
                logger.error(f"取消任务失败: {task.id}, 错误信息: {e}")

        # 第二阶段：释放任务运行器持有的外部资源（如沙箱连接等）。
        # 与“取消执行”分离可以避免状态遍历与状态修改交叉。
        for task in tasks:
            if task._task_runner:
                try:
                    await task._task_runner.destroy()
                except Exception as e:
                    logger.error(f"销毁任务运行器失败: {task.id}, 错误信息: {e}")

        # 清空任务注册表，释放所有任务引用
        cls._task_registry.clear()
