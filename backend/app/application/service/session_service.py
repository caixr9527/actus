#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/2/8 15:43
@Author : caixiaorong01@outlook.com
@File   : session_service.py
"""
import logging
from typing import List, Callable, Type, Any, Dict, Optional

from app.application.contracts import FileReadResult, ShellReadResult
from app.application.errors import NotFoundError, ServerError, ValidationError
from app.application.errors import error_keys
from app.application.service.model_config_service import ModelConfigService
from app.domain.external import Sandbox
from app.domain.models import File, Session, WorkflowRunEventRecord
from app.domain.repositories import IUnitOfWork

logger = logging.getLogger(__name__)


class SessionService:
    """会话服务"""

    def __init__(
            self,
            uow_factory: Callable[[], IUnitOfWork],
            sandbox_cls: Type[Sandbox],
            model_config_service: ModelConfigService | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._sandbox_cls = sandbox_cls
        self._model_config_service = model_config_service

    async def _get_owned_session_or_raise(self, user_id: str, session_id: str) -> Session:
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id=session_id, user_id=user_id)
        if not session:
            logger.error(f"任务会话不存在或无权访问: {session_id}, user_id={user_id}")
            raise NotFoundError(
                msg=f"任务会话不存在: {session_id}",
                error_key=error_keys.SESSION_NOT_FOUND,
                error_params={"session_id": session_id},
            )
        return session

    async def create_session(self, user_id: str) -> Session:
        logger.info("创建任务会话")
        session = Session(title="新会话", user_id=user_id)
        # 每次写操作都创建新的UoW，避免在服务对象中复用事务状态。
        async with self._uow_factory() as uow:
            await uow.session.save(session)
        logger.info(f"创建任务会话成功: {session.id}")
        return session

    async def get_all_sessions(self, user_id: str) -> List[Session]:
        # 读操作同样使用短生命周期UoW，保证连接可及时归还连接池。
        async with self._uow_factory() as uow:
            return await uow.session.get_all(user_id=user_id)

    async def clear_unread_message_count(self, user_id: str, session_id: str) -> None:
        logger.info(f"清除任务会话未读消息数: {session_id}")
        await self._get_owned_session_or_raise(user_id=user_id, session_id=session_id)
        async with self._uow_factory() as uow:
            await uow.session.update_unread_message_count(session_id=session_id, count=0)

    async def delete_session(self, user_id: str, session_id: str) -> None:
        logger.info(f"删除任务会话: {session_id}")
        await self._get_owned_session_or_raise(user_id=user_id, session_id=session_id)
        async with self._uow_factory() as uow:
            await uow.session.delete_by_id(session_id=session_id)
        logger.info(f"删除任务会话成功: {session_id}")

    async def get_session(self, user_id: str, session_id: str) -> Session | None:
        async with self._uow_factory() as uow:
            return await uow.session.get_by_id(session_id=session_id, user_id=user_id)

    async def get_session_detail(
            self,
            user_id: str,
            session_id: str,
    ) -> tuple[Session | None, list[WorkflowRunEventRecord], Dict[str, Dict[str, Any]]]:
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id=session_id, user_id=user_id)
            if session is None:
                return None, [], {}

            event_records = await uow.workflow_run.list_event_records_by_session(session_id=session.id)
            runtime_extensions_by_run_id: Dict[str, Dict[str, Any]] = {}
            run_ids = [
                normalized_run_id
                for normalized_run_id in dict.fromkeys(
                    str(record.run_id or "").strip()
                    for record in event_records
                )
                if normalized_run_id
            ]
            for run_id in run_ids:
                run = await uow.workflow_run.get_by_id(run_id)
                if run is None:
                    continue
                runtime_metadata = run.runtime_metadata if isinstance(run.runtime_metadata, dict) else {}
                runtime_extensions_by_run_id[run_id] = self._extract_runtime_extensions_from_metadata(runtime_metadata)
            return session, event_records, runtime_extensions_by_run_id

    @staticmethod
    def _summarize_input_parts(input_parts: list[dict[str, Any]]) -> Dict[str, Any]:
        by_type: Dict[str, int] = {}
        for part in input_parts:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "").strip().lower() or "unknown"
            by_type[part_type] = by_type.get(part_type, 0) + 1
        return {
            "total": len(input_parts),
            "by_type": by_type,
        }

    @classmethod
    def _extract_runtime_extensions_from_metadata(
            cls,
            runtime_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        graph_contract = runtime_metadata.get("graph_state_contract")
        if not isinstance(graph_contract, dict):
            return {}
        graph_state = graph_contract.get("graph_state")
        if not isinstance(graph_state, dict):
            return {}

        input_parts = graph_state.get("input_parts")
        normalized_input_parts = [item for item in input_parts if isinstance(item, dict)] if isinstance(
            input_parts, list
        ) else []
        graph_metadata = graph_state.get("metadata")
        normalized_graph_metadata = graph_metadata if isinstance(graph_metadata, dict) else {}
        input_policy = normalized_graph_metadata.get("input_policy")
        normalized_input_policy = input_policy if isinstance(input_policy, dict) else {}

        unsupported_parts = normalized_input_policy.get("unsupported_parts")
        normalized_unsupported_parts = (
            [item for item in unsupported_parts if isinstance(item, dict)]
            if isinstance(unsupported_parts, list)
            else []
        )
        downgrade_reasons: list[str] = []
        for unsupported_part in normalized_unsupported_parts:
            reason = str(unsupported_part.get("reason") or "").strip()
            if reason and reason not in downgrade_reasons:
                downgrade_reasons.append(reason)

        runtime_extensions: Dict[str, Any] = {
            "input_part_summary": cls._summarize_input_parts(normalized_input_parts),
            "unsupported_parts": normalized_unsupported_parts,
        }
        if downgrade_reasons:
            runtime_extensions["downgrade_reason"] = downgrade_reasons[0]
            runtime_extensions["downgrade_reasons"] = downgrade_reasons

        return runtime_extensions

    async def get_runtime_extensions(
            self,
            user_id: str,
            session_id: str,
    ) -> tuple[Optional[str], Dict[str, Any]]:
        """读取当前运行的 runtime 扩展信息，供 SSE extensions.runtime 注入。"""
        session = await self._get_owned_session_or_raise(user_id=user_id, session_id=session_id)
        run_id = session.current_run_id
        if not run_id:
            return None, {}

        async with self._uow_factory() as uow:
            run = await uow.workflow_run.get_by_id(run_id)
        if run is None:
            return run_id, {}

        runtime_metadata = run.runtime_metadata if isinstance(run.runtime_metadata, dict) else {}
        return run_id, self._extract_runtime_extensions_from_metadata(runtime_metadata)

    async def set_current_model(self, user_id: str, session_id: str, model_id: str) -> Session:
        logger.info(f"更新会话当前模型: session_id={session_id}, model_id={model_id}")
        session = await self._get_owned_session_or_raise(user_id=user_id, session_id=session_id)

        if model_id != "auto":
            if self._model_config_service is None:
                raise RuntimeError("ModelConfigService 未注入，无法校验模型 ID")
            model = await self._model_config_service.get_enabled_model_by_id(model_id=model_id)
            if model is None:
                raise ValidationError(
                    msg=f"模型[{model_id}]不存在或未启用",
                    error_key=error_keys.SESSION_MODEL_ID_INVALID,
                    error_params={"model_id": model_id},
                )

        async with self._uow_factory() as uow:
            await uow.session.update_current_model_id(
                session_id=session_id,
                current_model_id=model_id,
            )

        session.current_model_id = model_id
        return session

    async def get_session_files(self, user_id: str, session_id: str) -> List[File]:
        logger.info(f"获取任务会话文件列表: {session_id}")
        session = await self._get_owned_session_or_raise(user_id=user_id, session_id=session_id)
        return session.files

    async def read_file(self, user_id: str, session_id: str, filepath: str) -> FileReadResult:
        logger.info(f"获取会话：{session_id} 中文件路径：{filepath} 的内容")
        session = await self._get_owned_session_or_raise(user_id=user_id, session_id=session_id)

        # 检查会话是否关联了沙盒
        if not session.sandbox_id:
            raise NotFoundError(
                msg="任务会话未关联沙盒",
                error_key=error_keys.SESSION_SANDBOX_NOT_BOUND,
                error_params={"session_id": session_id},
            )

        # 根据沙盒ID获取沙盒实例
        sandbox = await self._sandbox_cls.get(id=session.sandbox_id)
        if not sandbox:
            raise NotFoundError(
                msg="任务会话未关联沙盒或已销毁",
                error_key=error_keys.SESSION_SANDBOX_UNAVAILABLE,
                error_params={"session_id": session_id, "sandbox_id": session.sandbox_id},
            )

        # 调用沙盒读取文件方法
        result = await sandbox.read_file(file_path=filepath)
        if result.success:
            # 返回文件读取结果
            return FileReadResult(**result.data)

        # 文件读取失败，抛出服务器错误
        raise ServerError(
            msg=result.message,
            error_key=error_keys.SESSION_FILE_READ_FAILED,
            error_params={"session_id": session_id, "filepath": filepath},
        )

    async def read_shell_output(self, user_id: str, session_id: str, shell_session_id: str) -> ShellReadResult:
        logger.info(f"获取会话：{session_id} 中Shell会话ID：{shell_session_id} 的输出")
        session = await self._get_owned_session_or_raise(user_id=user_id, session_id=session_id)

        # 检查会话是否关联了沙盒
        if not session.sandbox_id:
            raise NotFoundError(
                msg="任务会话未关联沙盒",
                error_key=error_keys.SESSION_SANDBOX_NOT_BOUND,
                error_params={"session_id": session_id},
            )

        # 根据沙盒ID获取沙盒实例
        sandbox = await self._sandbox_cls.get(id=session.sandbox_id)
        if not sandbox:
            raise NotFoundError(
                msg="任务会话未关联沙盒或已销毁",
                error_key=error_keys.SESSION_SANDBOX_UNAVAILABLE,
                error_params={"session_id": session_id, "sandbox_id": session.sandbox_id},
            )
        result = await sandbox.read_shell_output(session_id=shell_session_id, console=True)
        if result.success:
            # 读取成功，返回结果
            return ShellReadResult(**result.data)
        raise ServerError(
            msg=result.message,
            error_key=error_keys.SESSION_SHELL_READ_FAILED,
            error_params={"session_id": session_id, "shell_session_id": shell_session_id},
        )

    async def get_vnc_url(self, user_id: str, session_id: str) -> str:
        logger.info(f"获取会话：{session_id} 的VNC地址")
        session = await self._get_owned_session_or_raise(user_id=user_id, session_id=session_id)

        # 检查会话是否关联了沙盒
        if not session.sandbox_id:
            raise NotFoundError(
                msg="任务会话未关联沙盒",
                error_key=error_keys.SESSION_SANDBOX_NOT_BOUND,
                error_params={"session_id": session_id},
            )

        # 根据沙盒ID获取沙盒实例
        sandbox = await self._sandbox_cls.get(id=session.sandbox_id)
        if not sandbox:
            raise NotFoundError(
                msg="任务会话未关联沙盒或已销毁",
                error_key=error_keys.SESSION_SANDBOX_UNAVAILABLE,
                error_params={"session_id": session_id, "sandbox_id": session.sandbox_id},
            )

        return sandbox.vnc_url
