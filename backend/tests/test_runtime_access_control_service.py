import asyncio
from dataclasses import dataclass

import pytest

from app.application.errors import NotFoundError
from app.application.errors import error_keys
from app.application.service.runtime_access_control_service import RuntimeAccessControlService
from app.domain.models import Session, WorkflowRun, WorkflowRunStatus, Workspace
from app.domain.services.runtime.contracts.data_access_contract import (
    DataAccessAction,
    DataOrigin,
    DataResourceKind,
    PrivacyLevel,
    RetentionPolicyKind,
    default_privacy_level,
    default_trust_level,
    normalize_tenant_id,
)


@dataclass
class _SessionRepo:
    session: Session | None

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        if self.session is None or self.session.id != session_id:
            return None
        if user_id is not None and self.session.user_id != user_id:
            return None
        return self.session


@dataclass
class _WorkspaceRepo:
    workspace: Workspace | None
    workspaces: list[Workspace] | None = None

    async def get_by_id(self, workspace_id: str):
        if self.workspace is not None and self.workspace.id == workspace_id:
            return self.workspace
        return None

    async def get_by_id_for_user(self, workspace_id: str, user_id: str):
        if (
                self.workspace is not None
                and self.workspace.id == workspace_id
                and self.workspace.user_id == user_id
        ):
            return self.workspace
        return None

    async def get_by_session_id(self, session_id: str):
        workspaces = await self.list_by_session_id(session_id)
        return workspaces[0] if workspaces else None

    async def get_by_session_id_for_user(self, session_id: str, user_id: str):
        workspaces = await self.list_by_session_id(session_id)
        for workspace in workspaces:
            if workspace.user_id == user_id:
                return workspace
        return None

    async def list_by_session_id(self, session_id: str):
        if self.workspaces is not None:
            return [
                workspace
                for workspace in self.workspaces
                if workspace.session_id == session_id
            ]
        if self.workspace is not None and self.workspace.session_id == session_id:
            return [self.workspace]
        return []


@dataclass
class _WorkflowRunRepo:
    run: WorkflowRun | None

    async def get_by_id(self, run_id: str):
        if self.run is not None and self.run.id == run_id:
            return self.run
        return None

    async def get_by_id_for_user(self, run_id: str, user_id: str):
        if self.run is not None and self.run.id == run_id and self.run.user_id == user_id:
            return self.run
        return None


class _FileRepo:
    async def get_by_id_and_user_id(self, file_id: str, user_id: str):
        return None


class _WorkspaceArtifactRepo:
    async def list_by_workspace_id_and_paths(self, workspace_id: str, paths: list[str]):
        return []

    async def list_by_user_workspace_id_and_paths(self, user_id: str, workspace_id: str, paths: list[str]):
        return []


class _UoW:
    def __init__(
            self,
            *,
            session: Session | None,
            workspace: Workspace | None = None,
            workspaces: list[Workspace] | None = None,
            run: WorkflowRun | None = None,
    ) -> None:
        self.session = _SessionRepo(session=session)
        self.workspace = _WorkspaceRepo(workspace=workspace, workspaces=workspaces)
        self.workflow_run = _WorkflowRunRepo(run=run)
        self.file = _FileRepo()
        self.workspace_artifact = _WorkspaceArtifactRepo()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _build_service(uow: _UoW) -> RuntimeAccessControlService:
    return RuntimeAccessControlService(uow_factory=lambda: uow)


def test_data_access_contract_defaults_should_follow_p0_4_rules() -> None:
    assert normalize_tenant_id("user-a") == "user-a"
    assert default_privacy_level(DataOrigin.USER_MESSAGE) == PrivacyLevel.PRIVATE
    assert default_privacy_level(DataOrigin.LONG_TERM_MEMORY) == PrivacyLevel.SENSITIVE
    assert default_privacy_level(DataOrigin.SYSTEM_OPERATIONAL) == PrivacyLevel.INTERNAL
    assert default_trust_level(DataOrigin.EXTERNAL_WEB).value == "external_untrusted"


def test_resolve_session_scope_should_build_owned_scope() -> None:
    session = Session(
        id="session-a",
        user_id="user-a",
        workspace_id="workspace-a",
        current_run_id="run-a",
    )
    workspace = Workspace(id="workspace-a", session_id="session-a", user_id="user-a", current_run_id="run-a")
    run = WorkflowRun(
        id="run-a",
        session_id="session-a",
        user_id="user-a",
        status=WorkflowRunStatus.RUNNING,
        current_step_id="step-a",
    )
    service = _build_service(_UoW(session=session, workspace=workspace, run=run))

    scope = asyncio.run(service.resolve_session_scope(user_id="user-a", session_id="session-a"))

    assert scope.tenant_id == "user-a"
    assert scope.user_id == "user-a"
    assert scope.session_id == "session-a"
    assert scope.workspace_id == "workspace-a"
    assert scope.run_id == "run-a"
    assert scope.current_step_id == "step-a"


def test_resolve_session_scope_should_hide_foreign_session_as_not_found() -> None:
    session = Session(id="session-a", user_id="user-a")
    service = _build_service(_UoW(session=session))

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.resolve_session_scope(user_id="user-b", session_id="session-a"))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND
    assert exc.value.error_params == {"session_id": "session-a"}


def test_resolve_session_scope_should_reject_cross_user_current_run() -> None:
    session = Session(
        id="session-a",
        user_id="user-a",
        workspace_id="workspace-a",
        current_run_id="run-a",
    )
    workspace = Workspace(id="workspace-a", session_id="session-a", user_id="user-a", current_run_id="run-a")
    run = WorkflowRun(id="run-a", session_id="session-a", user_id="user-b")
    service = _build_service(_UoW(session=session, workspace=workspace, run=run))

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.resolve_session_scope(user_id="user-a", session_id="session-a"))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND


def test_resolve_session_scope_should_reject_multiple_workspaces() -> None:
    session = Session(id="session-a", user_id="user-a")
    workspaces = [
        Workspace(id="workspace-a", session_id="session-a", user_id="user-a"),
        Workspace(id="workspace-b", session_id="session-a", user_id="user-a"),
    ]
    service = _build_service(_UoW(session=session, workspaces=workspaces))

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.resolve_session_scope(user_id="user-a", session_id="session-a"))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND


def test_resolve_session_scope_should_reject_unbound_foreign_workspace() -> None:
    session = Session(id="session-a", user_id="user-a")
    workspaces = [
        Workspace(id="workspace-a", session_id="session-a", user_id="user-b"),
    ]
    service = _build_service(_UoW(session=session, workspaces=workspaces))

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.resolve_session_scope(user_id="user-a", session_id="session-a"))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND


def test_resolve_session_scope_should_reject_unbound_workspace_without_owner() -> None:
    session = Session(id="session-a", user_id="user-a")
    workspaces = [
        Workspace(id="workspace-a", session_id="session-a", user_id=None),
    ]
    service = _build_service(_UoW(session=session, workspaces=workspaces))

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.resolve_session_scope(user_id="user-a", session_id="session-a"))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND


def test_resolve_session_scope_should_accept_unbound_workspace_for_current_user() -> None:
    session = Session(id="session-a", user_id="user-a")
    workspaces = [
        Workspace(id="workspace-a", session_id="session-a", user_id="user-a"),
    ]
    service = _build_service(_UoW(session=session, workspaces=workspaces))

    scope = asyncio.run(service.resolve_session_scope(user_id="user-a", session_id="session-a"))

    assert scope.workspace_id == "workspace-a"


def test_resolve_session_scope_should_not_fallback_when_bound_workspace_missing() -> None:
    session = Session(
        id="session-a",
        user_id="user-a",
        workspace_id="workspace-missing",
    )
    workspaces = [
        Workspace(id="workspace-other", session_id="session-a", user_id="user-a"),
    ]
    service = _build_service(_UoW(session=session, workspace=None, workspaces=workspaces))

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.resolve_session_scope(user_id="user-a", session_id="session-a"))

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND
    assert exc.value.error_params == {"session_id": "session-a"}


def test_assert_session_replay_access_should_not_accept_cursor_or_event_id() -> None:
    session = Session(id="session-a", user_id="user-a")
    service = _build_service(_UoW(session=session))

    scope = asyncio.run(
        service.assert_session_replay_access(user_id="user-a", session_id="session-a")
    )

    assert scope.session_id == "session-a"
    assert scope.run_id is None


def test_classify_data_should_use_default_retention_policy() -> None:
    service = _build_service(_UoW(session=None))

    result = service.classify_data(
        origin=DataOrigin.LONG_TERM_MEMORY,
        requested_privacy_level=None,
        retention_policy=None,
    )

    assert result.privacy_level == PrivacyLevel.SENSITIVE
    assert result.retention_policy == RetentionPolicyKind.USER_MEMORY


def test_assert_sandbox_access_should_require_session_scope() -> None:
    service = _build_service(_UoW(session=None))

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(
            service.assert_sandbox_access(
                user_id="user-a",
                session_id="session-a",
                resource_kind=DataResourceKind.SANDBOX_FILE,
                action=DataAccessAction.READ,
            )
        )

    assert exc.value.error_key == error_keys.SESSION_NOT_FOUND
