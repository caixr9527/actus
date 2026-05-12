#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/2/8 15:43
@Author : caixiaorong01@outlook.com
@File   : session_service.py
"""
import logging
from typing import List, Callable, Type

from app.application.contracts import FileReadResult, ShellReadResult
from app.application.errors import NotFoundError, ServerError, ValidationError
from app.application.errors import error_keys
from app.application.service.runtime_access_control_service import RuntimeAccessControlService
from app.application.service.model_config_service import ModelConfigService
from app.domain.external import Sandbox
from app.domain.models import File, Session, WorkflowRunEventRecord
from app.domain.models.workspace import Workspace
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.data_access_contract import (
    DataAccessAction,
    DataResourceKind,
)
from app.domain.services.runtime.contracts.sandbox_path_policy import (
    is_allowed_sandbox_read_path,
    normalize_sandbox_path,
)

logger = logging.getLogger(__name__)


class SessionService:
    """会话服务"""

    def __init__(
            self,
            uow_factory: Callable[[], IUnitOfWork],
            sandbox_cls: Type[Sandbox],
            model_config_service: ModelConfigService | None = None,
            access_control_service: RuntimeAccessControlService | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._sandbox_cls = sandbox_cls
        self._model_config_service = model_config_service
        self._access_control_service = access_control_service or RuntimeAccessControlService(
            uow_factory=uow_factory,
        )

    async def _get_owned_session_or_raise(
            self,
            user_id: str,
            session_id: str,
            action: DataAccessAction = DataAccessAction.READ,
    ) -> Session:
        await self._access_control_service.assert_session_access(
            user_id=user_id,
            session_id=session_id,
            action=action,
        )
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
        await self._get_owned_session_or_raise(
            user_id=user_id,
            session_id=session_id,
            action=DataAccessAction.UPDATE,
        )
        async with self._uow_factory() as uow:
            await uow.session.update_unread_message_count(session_id=session_id, count=0)

    async def delete_session(self, user_id: str, session_id: str) -> None:
        logger.info(f"删除任务会话: {session_id}")
        await self._get_owned_session_or_raise(
            user_id=user_id,
            session_id=session_id,
            action=DataAccessAction.DELETE,
        )
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
    ) -> tuple[Session | None, list[WorkflowRunEventRecord]]:
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id=session_id, user_id=user_id)
            if session is None:
                return None, []

            event_records = await uow.workflow_run.list_event_records_by_session(session_id=session.id)
            return session, event_records

    async def set_current_model(self, user_id: str, session_id: str, model_id: str) -> Session:
        logger.info(f"更新会话当前模型: session_id={session_id}, model_id={model_id}")
        session = await self._get_owned_session_or_raise(
            user_id=user_id,
            session_id=session_id,
            action=DataAccessAction.UPDATE,
        )

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
        return session.final_files

    async def read_file(self, user_id: str, session_id: str, filepath: str) -> FileReadResult:
        logger.info(f"获取会话：{session_id} 中文件路径：{filepath} 的内容")
        normalized_path = self._normalize_allowed_sandbox_read_path(
            user_id=user_id,
            session_id=session_id,
            filepath=filepath,
        )
        session, workspace = await self._get_sandbox_session_and_workspace_or_raise(
            user_id=user_id,
            session_id=session_id,
            resource_kind=DataResourceKind.SANDBOX_FILE,
            action=DataAccessAction.READ,
        )
        await self._log_unindexed_sandbox_path_read(
            user_id=user_id,
            session_id=session.id,
            workspace_id=workspace.id,
            filepath=normalized_path,
        )
        sandbox = await self._sandbox_cls.get(id=str(workspace.sandbox_id or ""))
        if not sandbox:
            raise NotFoundError(
                msg="任务会话未关联沙盒或已销毁",
                error_key=error_keys.SESSION_SANDBOX_UNAVAILABLE,
                error_params={"session_id": session_id, "workspace_id": workspace.id},
            )

        # 调用沙盒读取文件方法
        result = await sandbox.read_file(file_path=normalized_path)
        if result.success:
            # 返回文件读取结果
            return FileReadResult(**result.data)

        # 文件读取失败，抛出服务器错误
        raise ServerError(
            msg=result.message,
            error_key=error_keys.SESSION_FILE_READ_FAILED,
            error_params={"session_id": session_id, "filepath": normalized_path},
        )

    async def _get_workspace_or_raise(self, session: Session, user_id: str):
        workspace_id = str(session.workspace_id or "").strip()
        if not workspace_id:
            raise NotFoundError(
                msg="任务会话未关联工作区",
                error_key=error_keys.SESSION_SANDBOX_NOT_BOUND,
                error_params={"session_id": session.id},
            )

        async with self._uow_factory() as uow:
            workspace = await uow.workspace.get_by_id_for_user(
                workspace_id=workspace_id,
                user_id=user_id,
            )
        if workspace is None:
            raise NotFoundError(
                msg="任务会话关联工作区不存在或已销毁",
                error_key=error_keys.SESSION_SANDBOX_UNAVAILABLE,
                error_params={"session_id": session.id, "workspace_id": workspace_id},
            )
        if not workspace.sandbox_id:
            raise NotFoundError(
                msg="任务会话工作区未关联沙盒",
                error_key=error_keys.SESSION_SANDBOX_NOT_BOUND,
                error_params={"session_id": session.id, "workspace_id": workspace_id},
            )
        return workspace

    async def _get_sandbox_session_and_workspace_or_raise(
            self,
            *,
            user_id: str,
            session_id: str,
            resource_kind: DataResourceKind,
            action: DataAccessAction,
    ) -> tuple[Session, Workspace]:
        await self._access_control_service.assert_sandbox_access(
            user_id=user_id,
            session_id=session_id,
            resource_kind=resource_kind,
            action=action,
        )
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id=session_id, user_id=user_id)
        if session is None:
            raise NotFoundError(
                msg=f"任务会话不存在: {session_id}",
                error_key=error_keys.SESSION_NOT_FOUND,
                error_params={"session_id": session_id},
            )
        workspace = await self._get_workspace_or_raise(session=session, user_id=user_id)
        return session, workspace

    @staticmethod
    def _normalize_allowed_sandbox_read_path(
            *,
            user_id: str,
            session_id: str,
            filepath: str,
    ) -> str:
        if not is_allowed_sandbox_read_path(filepath):
            logger.warning(
                "sandbox_path_access_denied",
                extra={
                    "event": "sandbox_path_access_denied",
                    "reason_code": "sandbox_path_invalid",
                    "user_id": user_id,
                    "session_id": session_id,
                    "resource_kind": DataResourceKind.SANDBOX_FILE.value,
                    "action": DataAccessAction.READ.value,
                    "filepath": str(filepath or "").strip(),
                },
            )
            raise ValidationError(
                msg="sandbox 文件路径不允许读取",
                error_key="error.session.sandbox_path_invalid",
                error_params={"filepath": filepath},
            )
        return normalize_sandbox_path(filepath)

    async def _log_unindexed_sandbox_path_read(
            self,
            *,
            user_id: str,
            session_id: str,
            workspace_id: str,
            filepath: str,
    ) -> None:
        async with self._uow_factory() as uow:
            artifacts = await uow.workspace_artifact.list_by_user_workspace_id_and_paths(
                user_id=user_id,
                workspace_id=workspace_id,
                paths=[filepath],
            )
        if artifacts:
            return
        logger.info(
            "sandbox_path_access",
            extra={
                "event": "sandbox_path_access",
                "reason_code": "sandbox_unindexed_path_read",
                "user_id": user_id,
                "session_id": session_id,
                "workspace_id": workspace_id,
                "filepath": filepath,
            },
        )

    async def read_shell_output(self, user_id: str, session_id: str) -> ShellReadResult:
        logger.info(f"获取会话：{session_id} 的默认Shell输出")
        session, workspace = await self._get_sandbox_session_and_workspace_or_raise(
            user_id=user_id,
            session_id=session_id,
            resource_kind=DataResourceKind.SANDBOX_SHELL,
            action=DataAccessAction.READ,
        )
        shell_session_id = str(workspace.shell_session_id or "").strip()
        if not shell_session_id:
            raise NotFoundError(
                msg="任务会话工作区未关联Shell会话",
                error_key=error_keys.SESSION_SANDBOX_NOT_BOUND,
                error_params={"session_id": session_id, "workspace_id": workspace.id},
            )

        sandbox = await self._sandbox_cls.get(id=str(workspace.sandbox_id or ""))
        if not sandbox:
            raise NotFoundError(
                msg="任务会话未关联沙盒或已销毁",
                error_key=error_keys.SESSION_SANDBOX_UNAVAILABLE,
                error_params={"session_id": session_id, "workspace_id": workspace.id},
            )
        result = await sandbox.read_shell_output(session_id=shell_session_id, console=True)
        if result.success:
            # 读取成功，返回结果
            return ShellReadResult(**result.data)
        raise ServerError(
            msg=result.message,
            error_key=error_keys.SESSION_SHELL_READ_FAILED,
            error_params={"session_id": session_id, "workspace_id": workspace.id},
        )

    async def get_vnc_url(self, user_id: str, session_id: str) -> str:
        logger.info(f"获取会话：{session_id} 的VNC地址")
        session, workspace = await self._get_sandbox_session_and_workspace_or_raise(
            user_id=user_id,
            session_id=session_id,
            resource_kind=DataResourceKind.SANDBOX_VNC,
            action=DataAccessAction.STREAM,
        )
        sandbox = await self._sandbox_cls.get(id=str(workspace.sandbox_id or ""))
        if not sandbox:
            raise NotFoundError(
                msg="任务会话未关联沙盒或已销毁",
                error_key=error_keys.SESSION_SANDBOX_UNAVAILABLE,
                error_params={"session_id": session_id, "workspace_id": workspace.id},
            )

        return sandbox.vnc_url
