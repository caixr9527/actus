#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/21 20:10
@Author : caixiaorong01@outlook.com
@File   : graph_runtime.py
"""
import logging
from typing import Callable, Optional, Protocol, Type

from app.domain.external import Browser, LLM, Sandbox, Task, TaskRunner
from app.domain.models import Session, WorkflowRunStatus
from app.domain.repositories import IUnitOfWork

logger = logging.getLogger(__name__)


class TaskRunnerFactory(Protocol):
    """任务执行器工厂协议，隔离 GraphRuntime 与具体 Runner 构造细节。"""

    def __call__(
            self,
            *,
            session: Session,
            llm: LLM,
            sandbox: Sandbox,
            browser: Browser,
    ) -> TaskRunner:
        """根据会话与运行时资源构建任务执行器。"""
        ...


class GraphRuntime(Protocol):
    """Graph 运行时访问协议，统一管理任务实例生命周期。"""

    async def get_task(self, session: Session) -> Optional[Task]:
        """按会话读取任务实例。"""
        ...

    async def create_task(self, session: Session, llm: LLM) -> Task:
        """按会话创建任务实例并完成运行关联写回。"""
        ...

    async def cancel_task(self, session: Session) -> bool:
        """按会话取消任务实例。"""
        ...

    async def destroy(self) -> None:
        """释放运行时持有资源。"""
        ...


class DefaultGraphRuntime(GraphRuntime):
    """默认 GraphRuntime 实现。

    说明：
    1. 当前仍复用 Task/Sandbox 现有抽象，不改变外部接口协议；
    2. 统一封装 task get/create/cancel/destroy 访问点，收口 AgentService 的运行时细节；
    3. `_task_registry` 仍可作为本地实现细节存在，但不再由 AgentService 直接依赖。
    """

    def __init__(
            self,
            sandbox_cls: Type[Sandbox],
            task_cls: Type[Task],
            uow_factory: Callable[[], IUnitOfWork],
            task_runner_factory: TaskRunnerFactory,
    ) -> None:
        self._sandbox_cls = sandbox_cls
        self._task_cls = task_cls
        self._uow_factory = uow_factory
        self._task_runner_factory = task_runner_factory

    async def get_task(self, session: Session) -> Optional[Task]:
        """按会话读取任务实例（当前默认实现由 task_id 定位）。"""
        task_id = session.task_id
        if not task_id:
            return None
        return self._task_cls.get(task_id=task_id)

    async def create_task(self, session: Session, llm: LLM) -> Task:
        """创建任务并在单一流程内完成会话关联写回与异常补偿。"""
        # 先尝试复用已存在沙箱；不存在时再创建新沙箱，减少外部资源抖动。
        sandbox = None
        sandbox_id = session.sandbox_id
        created_new_sandbox = False
        if sandbox_id:
            sandbox = await self._sandbox_cls.get(id=sandbox_id)
        if not sandbox:
            sandbox = await self._sandbox_cls.create()
            created_new_sandbox = True

        # 浏览器是当前任务运行必需资源，无法获取时应立即失败。
        browser = await sandbox.get_browser()
        if not browser:
            logger.error("会话%s创建任务失败: 沙箱%s创建浏览器失败", session.id, sandbox_id)
            raise RuntimeError(f"会话{session.id}创建任务失败: 沙箱{sandbox_id},创建浏览器失败")

        task = None
        try:
            # 构建任务执行器并创建任务实例。
            task_runner = self._task_runner_factory(
                session=session,
                llm=llm,
                sandbox=sandbox,
                browser=browser,
            )
            task = self._task_cls.create(task_runner=task_runner)
        except Exception:
            # 任务实例构造失败时，若本轮新建了沙箱则立即回收，避免外部资源泄漏。
            if created_new_sandbox:
                try:
                    await sandbox.destroy()
                except Exception as destroy_err:
                    logger.error("会话%s任务构造失败后销毁沙箱失败: %s", session.id, destroy_err)
            raise

        # 记录旧值用于补偿回滚，避免数据库写回失败后内存态悬挂。
        previous_sandbox_id = session.sandbox_id
        previous_task_id = session.task_id
        previous_current_run_id = session.current_run_id

        # 在单一写回流程中完成 run 创建与会话关联落库，保证关联视图一致性。
        session.sandbox_id = sandbox.id
        session.task_id = task.id
        try:
            async with self._uow_factory() as uow:
                run = await uow.workflow_run.create_for_session(
                    session=session,
                    status=WorkflowRunStatus.RUNNING,
                    thread_id=session.id,
                )
                session.current_run_id = run.id
                await uow.session.save(session=session)
            return task
        except Exception as save_err:
            logger.error("会话%s写回sandbox/task关联失败，开始补偿: %s", session.id, save_err)

            # 补偿1：撤销任务实例，避免无主任务继续留在运行时注册表。
            try:
                task.cancel()
            except Exception as cancel_err:
                logger.error("会话%s补偿取消任务失败: %s", session.id, cancel_err)

            # 补偿2：若本轮新建了沙箱，任务写回失败时释放资源，避免孤儿沙箱。
            if created_new_sandbox:
                try:
                    await sandbox.destroy()
                except Exception as destroy_err:
                    logger.error("会话%s补偿销毁沙箱失败: %s", session.id, destroy_err)

            # 补偿3：回滚会话内存态并尽力落库，降低后续读到脏关联字段的概率。
            session.sandbox_id = previous_sandbox_id
            session.task_id = previous_task_id
            session.current_run_id = previous_current_run_id
            try:
                async with self._uow_factory() as uow:
                    await uow.session.save(session=session)
            except Exception as rollback_err:
                logger.error("会话%s补偿回滚关联字段失败: %s", session.id, rollback_err)
            raise

    async def cancel_task(self, session: Session) -> bool:
        """取消会话当前任务，不存在任务时返回 False。"""
        task = await self.get_task(session=session)
        if task is None:
            return False
        return task.cancel()

    async def destroy(self) -> None:
        """销毁运行时任务实例。"""
        await self._task_cls.destroy()
