#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/21 22:10
@Author : caixiaorong01@outlook.com
@File   : checkpoint_store_adapter.py
"""
import logging
from typing import Any, Callable, Dict, Optional, Tuple

from app.domain.models import CheckpointRef
from app.domain.repositories import IUnitOfWork
from app.domain.services.workspace_runtime import WorkspaceManager

logger = logging.getLogger(__name__)


class CheckpointStoreAdapter:
    """LangGraph checkpoint 与 Session/WorkflowRun 映射适配器。"""

    def __init__(
            self,
            session_id: str,
            uow_factory: Callable[[], IUnitOfWork],
    ) -> None:
        self._session_id = session_id
        self._uow_factory = uow_factory
        self._workspace_manager = WorkspaceManager(uow_factory=uow_factory)

    @staticmethod
    def _normalize_checkpoint_namespace(namespace: Optional[str]) -> str:
        return str(namespace) if namespace is not None else ""

    async def _load_run_context(
            self,
    ) -> Tuple[Optional[str], str, Optional[str], Optional[str]]:
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id=self._session_id)
            if session is None:
                raise ValueError(f"会话[{self._session_id}]不存在，无法构建checkpoint上下文")

            current_run_id = await self._workspace_manager.resolve_current_run_id(
                session=session,
                uow=uow,
            )
            if not current_run_id:
                # 尚未创建 durable run 时，退化为 session 级 thread 运行。
                return None, session.id, None, None

            run = await uow.workflow_run.get_by_id(current_run_id)
            if run is None:
                logger.warning(
                    "会话[%s]当前运行[%s]不存在，LangGraph 将退化为 session 级 thread 配置",
                    self._session_id,
                    current_run_id,
                )
                return None, session.id, None, None

            return run.id, run.thread_id or session.id, run.checkpoint_namespace, run.checkpoint_id

    async def resolve_invoke_config(self) -> Tuple[Dict[str, Dict[str, str]], Optional[str]]:
        """解析 LangGraph 调用配置，并返回当前 run_id（若存在）。"""
        run_id, thread_id, checkpoint_namespace, checkpoint_id = await self._load_run_context()

        # 对齐 LangGraph configurable 约定：
        # thread_id 标识线程，checkpoint_ns/checkpoint_id 标识恢复点。
        configurable = {
            "thread_id": thread_id,
            "checkpoint_ns": self._normalize_checkpoint_namespace(checkpoint_namespace),
        }
        if checkpoint_id:
            configurable["checkpoint_id"] = checkpoint_id

        return {"configurable": configurable}, run_id

    @classmethod
    async def _get_latest_checkpoint_tuple(cls, checkpointer: Any, config: Dict[str, Any]) -> Any:
        async_get_tuple = getattr(checkpointer, "aget_tuple", None)
        if callable(async_get_tuple):
            try:
                return await async_get_tuple(config)
            except NotImplementedError:
                # 某些 checkpointer 仅实现同步接口。
                pass

        get_tuple = getattr(checkpointer, "get_tuple", None)
        if callable(get_tuple):
            return get_tuple(config)

        return None

    def _extract_checkpoint_ref(self, checkpoint_config: Optional[Dict[str, Any]]) -> Tuple[str, Optional[str]]:
        configurable = (checkpoint_config or {}).get("configurable") or {}
        checkpoint_namespace = self._normalize_checkpoint_namespace(configurable.get("checkpoint_ns"))
        checkpoint_id_raw = configurable.get("checkpoint_id")
        checkpoint_id = str(checkpoint_id_raw) if checkpoint_id_raw else None
        return checkpoint_namespace, checkpoint_id

    async def sync_latest_checkpoint_ref(
            self,
            run_id: Optional[str],
            checkpointer: Any,
            invoke_config: Dict[str, Dict[str, str]],
    ) -> CheckpointRef | None:
        """从 checkpointer 读取最新 checkpoint 引用，不直接写入数据库状态。"""
        if not run_id or checkpointer is None:
            return None

        configurable = invoke_config.get("configurable") or {}
        lookup_config = {
            "configurable": {
                "thread_id": str(configurable.get("thread_id", self._session_id)),
                "checkpoint_ns": self._normalize_checkpoint_namespace(configurable.get("checkpoint_ns")),
            }
        }

        try:
            # 只带 thread/ns 查询“最新 checkpoint”，避免依赖调用前的 checkpoint_id。
            checkpoint_tuple = await self._get_latest_checkpoint_tuple(
                checkpointer=checkpointer,
                config=lookup_config,
            )
            if checkpoint_tuple is None:
                return None

            checkpoint_namespace, checkpoint_id = self._extract_checkpoint_ref(checkpoint_tuple.config)
            if not checkpoint_id:
                return None

            return CheckpointRef(
                namespace=checkpoint_namespace,
                checkpoint_id=checkpoint_id,
            )
        except Exception as e:
            logger.warning("会话[%s]同步checkpoint引用失败: %s", self._session_id, e)
            return None
