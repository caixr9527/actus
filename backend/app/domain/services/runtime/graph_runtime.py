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
from app.domain.models import RuntimeStateSnapshot, Session, Workspace
from app.domain.repositories import IUnitOfWork
from app.domain.services.workspace_runtime import WorkspaceManager

logger = logging.getLogger(__name__)


def _log_core(level: int, event: str, **fields: object) -> None:
    """核心链路日志统一走 runtime_logging 格式；导入失败时降级普通日志。"""
    try:
        from app.domain.services.runtime.contracts.runtime_logging import log_runtime

        log_runtime(logger, level, event, **fields)
    except Exception:
        logger.log(level, "规划执行 事件=%s 字段=%s", event, fields)


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


class RuntimeStateCoordinatorPort(Protocol):
    """GraphRuntime 依赖的 Runtime 状态协调端口。"""

    async def start_run(
            self,
            session_id: str,
            *,
            sandbox_id: str,
            task_id: str,
            thread_id: str | None = None,
    ) -> RuntimeStateSnapshot:
        ...

    async def reopen_running_run(
            self,
            session_id: str,
            *,
            run_id: str,
            sandbox_id: str,
            task_id: str,
    ) -> RuntimeStateSnapshot:
        ...


class GraphRuntime(Protocol):
    """Graph 运行时访问协议，统一管理任务实例生命周期。"""

    async def get_task(self, session: Session) -> Optional[Task]:
        """按会话读取任务实例。"""
        ...

    async def create_task(self, session: Session, llm: LLM) -> Task:
        """按会话创建任务实例并完成运行关联写回。"""
        ...

    async def resume_task(self, session: Session, llm: LLM) -> Task:
        """为已有运行重建任务实例并复用当前 run。"""
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
            runtime_state_coordinator: RuntimeStateCoordinatorPort,
    ) -> None:
        self._sandbox_cls = sandbox_cls
        self._task_cls = task_cls
        self._uow_factory = uow_factory
        self._task_runner_factory = task_runner_factory
        self._workspace_manager = WorkspaceManager(uow_factory=uow_factory)
        self._runtime_state_coordinator = runtime_state_coordinator

    @staticmethod
    def _resolve_log_workspace_id(session: Session, workspace: Optional[Workspace]) -> str:
        """日志统一使用已绑定 workspace_id，避免输出临时对象 id 造成排障混淆。"""
        session_workspace_id = str(session.workspace_id or "").strip()
        if session_workspace_id:
            return session_workspace_id
        if workspace is None:
            return ""
        # 尚未绑定前，workspace 可能是临时对象，不输出该 id 以保证链路口径一致。
        return ""

    async def get_task(self, session: Session) -> Optional[Task]:
        """按会话读取任务实例。"""
        workspace = await self._get_workspace(session=session)
        if workspace is None or not workspace.task_id:
            return None
        return self._task_cls.get(task_id=workspace.task_id)

    async def _get_workspace(self, session: Session) -> Optional[Workspace]:
        return await self._workspace_manager.get_workspace(session=session)

    async def _build_task_instance(
            self,
            session: Session,
            llm: LLM,
            workspace: Workspace,
    ) -> tuple[Task, Sandbox, bool]:
        """构建任务实例，统一复用创建/恢复场景。"""
        _log_core(
            logging.INFO,
            "构建任务实例开始",
            session_id=session.id,
            workspace_id=self._resolve_log_workspace_id(session, workspace),
            sandbox_id=str(workspace.sandbox_id or "").strip(),
        )
        # 先尝试复用已存在沙箱；不存在时再创建新沙箱，减少外部资源抖动。
        sandbox = None
        sandbox_id = workspace.sandbox_id
        created_new_sandbox = False
        if sandbox_id:
            sandbox = await self._sandbox_cls.get(id=sandbox_id)
        if not sandbox:
            sandbox = await self._sandbox_cls.create()
            created_new_sandbox = True
            _log_core(
                logging.INFO,
                "创建新沙箱",
                session_id=session.id,
                workspace_id=self._resolve_log_workspace_id(session, workspace),
                sandbox_id=sandbox.id,
            )
        else:
            _log_core(
                logging.INFO,
                "复用已有沙箱",
                session_id=session.id,
                workspace_id=self._resolve_log_workspace_id(session, workspace),
                sandbox_id=sandbox.id,
            )

        # 浏览器是当前任务运行必需资源，无法获取时应立即失败。
        browser = await sandbox.get_browser()
        if not browser:
            _log_core(
                logging.ERROR,
                "构建任务实例失败: 沙箱浏览器不可用",
                session_id=session.id,
                workspace_id=self._resolve_log_workspace_id(session, workspace),
                sandbox_id=str(sandbox_id or ""),
            )
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
            _log_core(
                logging.INFO,
                "构建任务实例成功",
                session_id=session.id,
                workspace_id=self._resolve_log_workspace_id(session, workspace),
                task_id=task.id,
                sandbox_id=sandbox.id,
                created_new_sandbox=created_new_sandbox,
            )
        except Exception:
            # 任务实例构造失败时，若本轮新建了沙箱则立即回收，避免外部资源泄漏。
            if created_new_sandbox:
                try:
                    await sandbox.destroy()
                except Exception as destroy_err:
                    logger.error("会话%s任务构造失败后销毁沙箱失败: %s", session.id, destroy_err)
            raise
        return task, sandbox, created_new_sandbox

    async def create_task(self, session: Session, llm: LLM) -> Task:
        """创建任务并在单一流程内完成会话关联写回与异常补偿。"""
        workspace = await self._get_workspace(session=session)
        if workspace is None:
            workspace = Workspace(session_id=session.id)
        _log_core(
            logging.INFO,
            "创建任务开始",
            session_id=session.id,
            workspace_id=self._resolve_log_workspace_id(session, workspace),
            source_run_id=str(session.current_run_id or "").strip(),
        )

        task, sandbox, created_new_sandbox = await self._build_task_instance(
            session=session,
            llm=llm,
            workspace=workspace,
        )

        # 记录旧值用于补偿回滚，避免数据库写回失败后内存态悬挂。
        previous_workspace_id = session.workspace_id
        previous_current_run_id = session.current_run_id
        previous_status = session.status
        previous_workspace = workspace.model_copy(deep=True)

        try:
            snapshot = await self._runtime_state_coordinator.start_run(
                session_id=session.id,
                sandbox_id=sandbox.id,
                task_id=task.id,
                thread_id=session.id,
            )
            session.workspace_id = snapshot.workspace_id
            session.current_run_id = snapshot.run_id
            session.status = snapshot.session_status
            _log_core(
                logging.INFO,
                "创建任务完成",
                session_id=session.id,
                workspace_id=snapshot.workspace_id,
                run_id=snapshot.run_id,
                task_id=task.id,
                sandbox_id=sandbox.id,
                created_new_sandbox=created_new_sandbox,
            )
            return task
        except Exception as save_err:
            _log_core(
                logging.ERROR,
                "创建任务写回失败，开始补偿",
                session_id=session.id,
                workspace_id=workspace.id,
                task_id=getattr(task, "id", ""),
                sandbox_id=getattr(sandbox, "id", ""),
                error=str(save_err),
            )

            # 补偿1：撤销任务实例，避免无主任务继续留在运行时注册表。
            try:
                await task.cancel()
            except Exception as cancel_err:
                logger.error("会话%s补偿取消任务失败: %s", session.id, cancel_err)

            # 补偿2：若本轮新建了沙箱，任务写回失败时释放资源，避免孤儿沙箱。
            if created_new_sandbox:
                try:
                    await sandbox.destroy()
                except Exception as destroy_err:
                    logger.error("会话%s补偿销毁沙箱失败: %s", session.id, destroy_err)

            # 补偿3：回滚会话内存态并尽力落库，降低后续读到脏关联字段的概率。
            session.workspace_id = previous_workspace_id
            session.current_run_id = previous_current_run_id
            session.status = previous_status
            workspace = previous_workspace
            try:
                async with self._uow_factory() as uow:
                    if previous_workspace_id:
                        await uow.workspace.save(workspace=workspace)
                    await uow.session.save(session=session)
            except Exception as rollback_err:
                logger.error("会话%s补偿回滚关联字段失败: %s", session.id, rollback_err)
            raise

    async def resume_task(self, session: Session, llm: LLM) -> Task:
        """为已有 run 重建任务实例，但不新建 workflow_run。"""
        workspace = await self._get_workspace(session=session)
        if workspace is None:
            raise RuntimeError(f"会话{session.id}缺少 workspace，无法恢复执行")
        resolved_run_id = str(workspace.current_run_id or "").strip()
        if not resolved_run_id:
            raise RuntimeError(f"会话{session.id}缺少 current_run_id，无法恢复执行")
        _log_core(
            logging.INFO,
            "恢复任务开始",
            session_id=session.id,
            workspace_id=workspace.id,
            run_id=resolved_run_id,
        )

        task, sandbox, created_new_sandbox = await self._build_task_instance(
            session=session,
            llm=llm,
            workspace=workspace,
        )
        previous_workspace_id = session.workspace_id
        previous_current_run_id = session.current_run_id
        previous_status = session.status
        previous_workspace = workspace.model_copy(deep=True)

        try:
            snapshot = await self._runtime_state_coordinator.reopen_running_run(
                session_id=session.id,
                run_id=resolved_run_id,
                sandbox_id=sandbox.id,
                task_id=task.id,
            )
            session.workspace_id = snapshot.workspace_id
            session.current_run_id = snapshot.run_id
            session.status = snapshot.session_status
            _log_core(
                logging.INFO,
                "恢复任务完成",
                session_id=session.id,
                workspace_id=snapshot.workspace_id,
                run_id=resolved_run_id,
                task_id=task.id,
                sandbox_id=sandbox.id,
                created_new_sandbox=created_new_sandbox,
            )
            return task
        except Exception as save_err:
            _log_core(
                logging.ERROR,
                "恢复任务写回失败，开始补偿",
                session_id=session.id,
                workspace_id=workspace.id,
                run_id=resolved_run_id,
                task_id=getattr(task, "id", ""),
                sandbox_id=getattr(sandbox, "id", ""),
                error=str(save_err),
            )

            try:
                await task.cancel()
            except Exception as cancel_err:
                logger.error("会话%s恢复任务补偿取消失败: %s", session.id, cancel_err)

            if created_new_sandbox:
                try:
                    await sandbox.destroy()
                except Exception as destroy_err:
                    logger.error("会话%s恢复任务补偿销毁沙箱失败: %s", session.id, destroy_err)

            session.workspace_id = previous_workspace_id
            session.current_run_id = previous_current_run_id
            session.status = previous_status
            workspace = previous_workspace
            try:
                async with self._uow_factory() as uow:
                    await uow.workspace.save(workspace=workspace)
                    await uow.session.save(session=session)
            except Exception as rollback_err:
                logger.error("会话%s恢复任务补偿回滚关联字段失败: %s", session.id, rollback_err)
            raise

    async def cancel_task(self, session: Session) -> bool:
        """取消会话当前任务，并等待运行态收敛结束；不存在任务时返回 False。"""
        task = await self.get_task(session=session)
        if task is None:
            return False
        return await task.cancel()

    async def destroy(self) -> None:
        """销毁运行时任务实例。"""
        await self._task_cls.destroy()
