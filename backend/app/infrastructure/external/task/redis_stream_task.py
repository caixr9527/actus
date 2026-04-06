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

    # 进程内任务注册表：
    # - 当前用于通过 task_id 快速定位活跃任务实例；
    # - 在 BE-LG-07 后该结构退化为实现细节，不再由 AgentService 直接依赖。
    _task_registry: Dict[str, "RedisStreamTask"] = {}

    def __init__(self, task_runner: TaskRunner):
        self._task_runner = task_runner
        self._id = str(uuid.uuid4())
        self._execution_task: Optional[asyncio.Task] = None
        self._finalize_task: Optional[asyncio.Task] = None
        self._streams_cleaned = False

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

    async def _cleanup_streams(self) -> None:
        """清理任务输入/输出流，避免Redis Stream key与消息残留"""
        if self._streams_cleaned:
            return

        cleanup_success = True
        for stream_name, stream in (("input", self._input_stream), ("output", self._output_stream)):
            try:
                ok = await stream.delete_stream()
                if ok:
                    logger.info(f"清理任务{self._id}的{stream_name}流成功")
                else:
                    cleanup_success = False
                    logger.warning(f"清理任务{self._id}的{stream_name}流返回失败")
            except Exception as e:
                cleanup_success = False
                logger.error(f"清理任务{self._id}的{stream_name}流失败: {e}")

        if cleanup_success:
            self._streams_cleaned = True

    async def _finalize_after_run(self) -> None:
        """执行任务完成后的统一回收。

        注意顺序：
        1. 先执行 runner.on_done，保证运行态收尾逻辑可以访问 task 上下文；
        2. 再清理流，避免 stop 返回前历史事件尚未可见就被删流；
        3. 最后回收 registry，彻底移除活跃任务引用。
        """
        try:
            if self._task_runner:
                await self._task_runner.on_done(self)
        finally:
            try:
                await self._cleanup_streams()
            finally:
                self._cleanup_registry()

    def _on_task_done(self) -> None:
        """任务完成回调。"""
        if self._finalize_task is not None and not self._finalize_task.done():
            return
        self._finalize_task = asyncio.create_task(self._finalize_after_run())

    async def _wait_for_finalization(self) -> None:
        finalize_task = self._finalize_task
        if finalize_task is None:
            await self._finalize_after_run()
            return
        await finalize_task

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
        # done=True 表示“当前没有活跃执行协程”：
        # - 首次调用时 _execution_task 为 None，因此允许启动；
        # - 历史执行结束后再次 invoke 也允许启动新一轮执行。
        if self.done:
            self._execution_task = asyncio.create_task(self._execute_task())
            logger.info(f"开始执行任务: {self._id}")

    async def cancel(self) -> bool:
        """取消任务，并等待执行协程完成 cancelled 收敛后再返回。"""
        if not self.done and self._execution_task:
            self._execution_task.cancel()
            logger.info(f"取消任务: {self._id}")
            try:
                await self._execution_task
            except asyncio.CancelledError:
                # 运行协程取消是预期路径；AgentTaskRunner 会在内部完成 cancelled 收敛。
                pass
            await self._wait_for_finalization()
            return True

        await self._wait_for_finalization()
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
        # 未启动执行协程时视为“已完成/可启动”，方便上层按统一条件触发 invoke。
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
                await task.cancel()
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
