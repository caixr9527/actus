import asyncio
import logging
from datetime import datetime, timedelta

from app.application.service.runtime_observation_service import RuntimeObservationService
from app.domain.models import (
    MessageEvent,
    Session,
    SessionStatus,
    WorkflowRun,
    WorkflowRunEventRecord,
    WorkflowRunStatus,
    Workspace,
)
from app.domain.services.runtime.contracts.sandbox_capability_profile_contract import (
    SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY,
    RuntimeToolCapabilitySnapshot,
    RuntimeToolCapabilitySnapshotItem,
    SandboxCapabilityItem,
    SandboxCapabilityKind,
    SandboxCapabilityPromptSummary,
    SandboxCapabilityStatus,
    SandboxProfileRefreshReason,
    build_sandbox_capability_profile_from_draft,
)
from app.interfaces.schemas.session import RuntimeObservationResponse


_EXPIRES_AT_UNSET = object()


class _SessionRepo:
    def __init__(self, session: Session) -> None:
        self.session = session

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        if session_id != self.session.id:
            return None
        if user_id is not None and user_id != self.session.user_id:
            return None
        return self.session

    async def get_by_id_for_update(self, session_id: str):
        return self.session if session_id == self.session.id else None

    async def update_runtime_state(self, session_id: str, *, status: SessionStatus, current_run_id=None, **_kwargs):
        self.session.status = status
        if current_run_id is not None:
            self.session.current_run_id = current_run_id


class _WorkspaceRepo:
    def __init__(self, workspace: Workspace | None) -> None:
        self.workspace = workspace

    async def get_by_id(self, workspace_id: str):
        if self.workspace is None:
            return None
        return self.workspace if self.workspace.id == workspace_id else None

    async def get_by_id_for_user(self, workspace_id: str, user_id: str):
        if self.workspace is None:
            return None
        if self.workspace.id != workspace_id or self.workspace.user_id != user_id:
            return None
        return self.workspace

    async def get_by_session_id(self, session_id: str):
        if self.workspace is None:
            return None
        return self.workspace if self.workspace.session_id == session_id else None

    async def get_by_session_id_for_user(self, session_id: str, user_id: str):
        if self.workspace is None:
            return None
        if self.workspace.session_id != session_id or self.workspace.user_id != user_id:
            return None
        return self.workspace

    async def list_by_session_id(self, session_id: str):
        if self.workspace is not None and self.workspace.session_id == session_id:
            return [self.workspace]
        return []


class _WorkflowRunRepo:
    def __init__(self, run: WorkflowRun | None, records: list[WorkflowRunEventRecord]) -> None:
        self.run = run
        self.records = records

    async def get_by_id_for_user(self, run_id: str, user_id: str):
        if self.run is None or self.run.id != run_id or self.run.user_id != user_id:
            return None
        return self.run

    async def get_by_id_for_update(self, run_id: str):
        if self.run is not None and self.run.id == run_id:
            return self.run
        return None

    async def list_event_records_by_session(self, session_id: str):
        return [record for record in self.records if record.session_id == session_id]


class _UoW:
    def __init__(
            self,
            *,
            session: Session,
            workspace: Workspace | None,
            run: WorkflowRun | None,
            records: list[WorkflowRunEventRecord] | None = None,
    ) -> None:
        self.session = _SessionRepo(session)
        self.workspace = _WorkspaceRepo(workspace)
        self.workflow_run = _WorkflowRunRepo(run, records or [])

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def commit(self):
        return None

    async def rollback(self):
        return None


def _build_profile_payload(
        *,
        generated_at: datetime | None = None,
        expires_at: datetime | None | object = _EXPIRES_AT_UNSET,
        prompt_stale: bool = False,
) -> dict:
    now = generated_at or datetime.now()
    draft = {
        "schema_version": "sandbox_capability_profile.v1",
        "profile_id": "profile-1",
        "user_id": "user-1",
        "session_id": "session-1",
        "workspace_id": "workspace-1",
        "run_id": "run-1",
        "sandbox_id": "sandbox-1",
        "generated_at": now,
        "expires_at": now + timedelta(minutes=30) if expires_at is _EXPIRES_AT_UNSET else expires_at,
        "refresh_reason": SandboxProfileRefreshReason.SANDBOX_CREATED,
        "health_status": SandboxCapabilityStatus.AVAILABLE,
        "cwd": "/workspace",
        "capabilities": [
            SandboxCapabilityItem(
                kind=SandboxCapabilityKind.PYTHON,
                name="python3",
                status=SandboxCapabilityStatus.AVAILABLE,
                version="3.12.1",
            )
        ],
        "runtime_tool_capabilities": RuntimeToolCapabilitySnapshot(
            items=[
                RuntimeToolCapabilitySnapshotItem(
                    capability_id="local_shell",
                    tool_family="shell",
                    source="local",
                )
            ]
        ),
        "prompt_summary": SandboxCapabilityPromptSummary(
            health_status=SandboxCapabilityStatus.AVAILABLE,
            cwd="/workspace",
            available_runtime={"python": "3.12.1"},
            available_tools=["shell"],
            unavailable_capabilities=["node"],
            requires_confirmation=["shell"],
            generated_at=now,
            sandbox_profile_stale=prompt_stale,
        ),
    }
    return build_sandbox_capability_profile_from_draft(draft).model_dump(mode="json")


def _build_observation(
        *,
        workspace: Workspace | None,
        records: list[WorkflowRunEventRecord] | None = None,
) -> tuple[RuntimeObservationService, _UoW]:
    session = Session(
        id="session-1",
        user_id="user-1",
        workspace_id="workspace-1" if workspace is not None else None,
        current_run_id="run-1",
        status=SessionStatus.RUNNING,
    )
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        user_id="user-1",
        status=WorkflowRunStatus.RUNNING,
    )
    uow = _UoW(session=session, workspace=workspace, run=run, records=records)
    return RuntimeObservationService(uow_factory=lambda: uow), uow


def _message_record(event_id: str) -> WorkflowRunEventRecord:
    event = MessageEvent(id=event_id, role="assistant", message="ok")
    return WorkflowRunEventRecord(
        run_id="run-1",
        session_id="session-1",
        user_id="user-1",
        event_id=event_id,
        event_type=event.type,
        event_payload=event,
    )


def test_runtime_observation_should_return_sandbox_profile_projection() -> None:
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        current_run_id="run-1",
        environment_summary={
            SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: _build_profile_payload(),
        },
    )
    service, _uow = _build_observation(workspace=workspace, records=[_message_record("evt-1")])

    result = asyncio.run(service.build_session_observation(user_id="user-1", session_id="session-1"))

    assert result.sandbox_profile is not None
    assert result.sandbox_profile.schema_version == "sandbox_capability_profile.v1"
    assert result.sandbox_profile.health_status == "available"
    assert result.sandbox_profile.stale is False
    assert result.sandbox_profile.unavailable_capabilities == ["node"]
    assert result.sandbox_profile.requires_confirmation == ["shell"]
    assert result.cursor.latest_event_id == "evt-1"


def test_runtime_observation_should_return_null_profile_without_changing_capabilities_when_missing() -> None:
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        current_run_id="run-1",
        environment_summary={},
    )
    service, _uow = _build_observation(workspace=workspace)

    result = asyncio.run(service.build_session_observation(user_id="user-1", session_id="session-1"))

    assert result.sandbox_profile is None
    assert result.capabilities.can_cancel is True
    assert result.capabilities.can_send_message is False


def test_runtime_observation_response_should_include_null_sandbox_profile_field() -> None:
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        current_run_id="run-1",
        environment_summary={},
    )
    service, _uow = _build_observation(workspace=workspace)

    result = asyncio.run(service.build_session_observation(user_id="user-1", session_id="session-1"))
    response = RuntimeObservationResponse.from_result(result)

    assert response.model_dump()["sandbox_profile"] is None


def test_runtime_observation_should_return_null_profile_when_session_has_no_workspace() -> None:
    session = Session(id="session-1", user_id="user-1", status=SessionStatus.PENDING)
    uow = _UoW(session=session, workspace=None, run=None, records=[])
    service = RuntimeObservationService(uow_factory=lambda: uow)

    result = asyncio.run(service.build_session_observation(user_id="user-1", session_id="session-1"))

    assert result.sandbox_profile is None
    assert result.capabilities.can_send_message is True


def test_runtime_observation_should_return_null_profile_for_invalid_payload(caplog) -> None:
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        current_run_id="run-1",
        environment_summary={
            SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: {
                **_build_profile_payload(),
                "legacy_extra": "invalid",
            },
        },
    )
    service, _uow = _build_observation(workspace=workspace)

    with caplog.at_level(logging.WARNING, logger="app.domain.services.workspace_runtime.service"):
        result = asyncio.run(service.build_session_observation(user_id="user-1", session_id="session-1"))

    assert result.sandbox_profile is None
    assert result.capabilities.can_cancel is True
    assert any(record.message == "sandbox_profile_invalid_payload" for record in caplog.records)


def test_runtime_observation_should_mark_profile_stale_for_prompt_flag_expired_or_missing_expiry() -> None:
    now = datetime.now()
    scenarios = [
        _build_profile_payload(prompt_stale=True),
        _build_profile_payload(generated_at=now - timedelta(hours=2), expires_at=now - timedelta(minutes=1)),
        _build_profile_payload(expires_at=None),
    ]

    for profile_payload in scenarios:
        workspace = Workspace(
            id="workspace-1",
            session_id="session-1",
            user_id="user-1",
            current_run_id="run-1",
            environment_summary={
                SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: profile_payload,
            },
        )
        service, _uow = _build_observation(workspace=workspace)

        result = asyncio.run(service.build_session_observation(user_id="user-1", session_id="session-1"))

        assert result.sandbox_profile is not None
        assert result.sandbox_profile.stale is True


def test_runtime_observation_profile_projection_should_not_change_events_or_cursor() -> None:
    records = [_message_record("evt-1")]
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        current_run_id="run-1",
        environment_summary={
            SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: _build_profile_payload(),
        },
    )
    service, uow = _build_observation(workspace=workspace, records=records)

    result = asyncio.run(service.build_session_observation(user_id="user-1", session_id="session-1"))

    assert result.sandbox_profile is not None
    assert result.cursor.latest_event_id == "evt-1"
    assert len(uow.workflow_run.records) == 1
