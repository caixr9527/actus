#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RuntimeStateCoordinator 骨架合同测试。"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime

from app.application.service.runtime_state_coordinator import RuntimeStateCoordinator
from app.domain.models import (
    CheckpointRef,
    DoneEvent,
    ErrorEvent,
    ExecutionStatus,
    MessageEvent,
    Plan,
    PlanEvent,
    StepEvent,
    StepEventStatus,
    RuntimeEventProjection,
    Session,
    SessionStatus,
    Step,
    WaitEvent,
    WorkflowRun,
    WorkflowRunEventRecord,
    WorkflowRunStatus,
    Workspace,
)
from app.domain.repositories.workflow_run_repository import UNSET_CURRENT_STEP_ID


@dataclass
class FakeSessionRepository:
    session: Session
    runtime_state_updates: list[dict] = field(default_factory=list)

    async def get_by_id_for_update(self, session_id: str) -> Session | None:
        return self.session if self.session.id == session_id else None

    async def save(self, session: Session) -> None:
        self.session = session

    async def update_runtime_state(
            self,
            session_id: str,
            *,
            status: SessionStatus,
            current_run_id: str | None = None,
            title: str | None = None,
            latest_message: str | None = None,
            latest_message_at: datetime | None = None,
            increment_unread: bool = False,
    ) -> None:
        self.session.status = status
        if current_run_id is not None:
            self.session.current_run_id = current_run_id
        if title is not None:
            self.session.title = title
        if latest_message is not None:
            self.session.latest_message = latest_message
        if latest_message_at is not None:
            self.session.latest_message_at = latest_message_at
        if increment_unread:
            self.session.unread_message_count += 1
        self.runtime_state_updates.append(
            {
                "session_id": session_id,
                "status": status,
                "current_run_id": current_run_id,
                "title": title,
                "latest_message": latest_message,
                "latest_message_at": latest_message_at,
                "increment_unread": increment_unread,
            }
        )


@dataclass
class FakeWorkflowRunRepository:
    run: WorkflowRun
    event_records: list[WorkflowRunEventRecord] = field(default_factory=list)
    status_updates: list[dict] = field(default_factory=list)
    replaced_plan_count: int = 0
    upserted_step_count: int = 0
    checkpoint_ref_updates: list[dict] = field(default_factory=list)
    runtime_metadata_updates: list[dict] = field(default_factory=list)
    unfinished_steps_cancelled_count: int = 0
    created_runs: list[WorkflowRun] = field(default_factory=list)

    async def get_by_id_for_update(self, run_id: str) -> WorkflowRun | None:
        return self.run if self.run.id == run_id else None

    async def get_by_id(self, run_id: str) -> WorkflowRun | None:
        return self.run if self.run.id == run_id else None

    async def create_for_session(
            self,
            session: Session,
            status: WorkflowRunStatus = WorkflowRunStatus.RUNNING,
            thread_id: str | None = None,
    ) -> WorkflowRun:
        self.run = WorkflowRun(
            id="run-started-1",
            session_id=session.id,
            user_id=session.user_id,
            status=status,
            thread_id=thread_id,
        )
        self.created_runs.append(self.run)
        return self.run

    async def update_status(
            self,
            run_id: str,
            *,
            status: WorkflowRunStatus,
            finished_at: datetime | None = None,
            last_event_at: datetime | None = None,
            current_step_id=UNSET_CURRENT_STEP_ID,
    ) -> None:
        self.run.status = status
        self.run.finished_at = finished_at
        self.run.last_event_at = last_event_at
        if current_step_id is not UNSET_CURRENT_STEP_ID:
            self.run.current_step_id = current_step_id
        self.status_updates.append(
            {
                "run_id": run_id,
                "status": status,
                "finished_at": finished_at,
                "last_event_at": last_event_at,
                "current_step_id": current_step_id,
            }
        )

    async def add_event_record_if_absent(
            self,
            session_id: str,
            run_id: str,
            event,
    ) -> bool:
        if any(record.event_id == event.id and record.run_id == run_id for record in self.event_records):
            return False
        self.event_records.append(
            WorkflowRunEventRecord(
                run_id=run_id,
                session_id=session_id,
                event_id=event.id,
                event_type=event.type,
                event_payload=event,
                created_at=event.created_at,
            )
        )
        return True

    async def update_checkpoint_ref(
            self,
            run_id: str,
            checkpoint_namespace: str | None,
            checkpoint_id: str | None,
    ) -> None:
        self.run.checkpoint_namespace = checkpoint_namespace
        self.run.checkpoint_id = checkpoint_id
        self.checkpoint_ref_updates.append(
            {
                "run_id": run_id,
                "checkpoint_namespace": checkpoint_namespace,
                "checkpoint_id": checkpoint_id,
            }
        )

    async def update_runtime_metadata(
            self,
            run_id: str,
            runtime_metadata: dict,
            current_step_id: str | None,
    ) -> None:
        self.run.runtime_metadata.update(runtime_metadata)
        self.run.current_step_id = current_step_id
        self.runtime_metadata_updates.append(
            {
                "run_id": run_id,
                "runtime_metadata": runtime_metadata,
                "current_step_id": current_step_id,
            }
        )

    async def replace_steps_from_plan(self, run_id: str, plan: Plan) -> None:
        self.replaced_plan_count += 1

    async def upsert_step_from_event(self, run_id: str, event) -> None:
        self.upserted_step_count += 1

    async def mark_unfinished_steps_cancelled(self, run_id: str) -> None:
        self.unfinished_steps_cancelled_count += 1

    async def list_events(self, run_id: str | None):
        return [
            record.event_payload
            for record in self.event_records
            if record.run_id == run_id
        ]

    async def list_event_records_by_session(self, session_id: str) -> list[WorkflowRunEventRecord]:
        return [record for record in self.event_records if record.session_id == session_id]


@dataclass
class FakeWorkspaceRepository:
    workspace: Workspace | None = None

    async def save(self, workspace: Workspace) -> None:
        self.workspace = workspace.model_copy(deep=True)

    async def get_by_id(self, workspace_id: str) -> Workspace | None:
        if self.workspace and self.workspace.id == workspace_id:
            return self.workspace
        return None

    async def get_by_session_id(self, session_id: str) -> Workspace | None:
        if self.workspace and self.workspace.session_id == session_id:
            return self.workspace
        return None


class FakeUnitOfWork:
    def __init__(
            self,
            *,
            session: Session,
            run: WorkflowRun,
            workspace: Workspace | None = None,
            event_records: list[WorkflowRunEventRecord] | None = None,
    ) -> None:
        self.session = FakeSessionRepository(session=session)
        self.workflow_run = FakeWorkflowRunRepository(
            run=run,
            event_records=event_records or [],
        )
        self.workspace = FakeWorkspaceRepository(workspace=workspace)
        self.committed = False
        self.rolled_back = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            await self.rollback()
            return False
        await self.commit()
        return False

    async def commit(self) -> None:
        self.committed = True

    async def rollback(self) -> None:
        self.rolled_back = True


def _build_uow(
        *,
        session_status: SessionStatus = SessionStatus.RUNNING,
        run_status: WorkflowRunStatus = WorkflowRunStatus.RUNNING,
        workspace_run_id: str | None = "run-1",
        event_records: list[WorkflowRunEventRecord] | None = None,
) -> FakeUnitOfWork:
    session = Session(
        id="session-1",
        workspace_id="workspace-1",
        current_run_id="session-run-1",
        status=session_status,
    )
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        status=run_status,
        checkpoint_namespace="",
        checkpoint_id="checkpoint-1",
    )
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        current_run_id=workspace_run_id,
    )
    return FakeUnitOfWork(
        session=session,
        run=run,
        workspace=workspace,
        event_records=event_records,
    )


def _coordinator_with_uow(uow: FakeUnitOfWork) -> RuntimeStateCoordinator:
    return RuntimeStateCoordinator(uow_factory=lambda: uow)


def _confirm_interrupt() -> dict:
    return {
        "kind": "confirm",
        "prompt": "是否继续？",
    }


def test_build_snapshot_should_use_workspace_run_id_as_current_run() -> None:
    uow = _build_uow()
    coordinator = _coordinator_with_uow(uow)

    snapshot = asyncio.run(coordinator.build_snapshot("session-1"))

    assert snapshot.session_id == "session-1"
    assert snapshot.workspace_id == "workspace-1"
    assert snapshot.run_id == "run-1"
    assert snapshot.workspace_run_id == "run-1"
    assert snapshot.session_run_id == "session-run-1"
    assert snapshot.has_checkpoint is True


def test_build_snapshot_should_compute_continuable_cancelled_plan_from_latest_plan_event() -> None:
    cancelled_plan_event = PlanEvent(
        id="plan-event-1",
        plan=Plan(
            steps=[Step(id="step-1", status=ExecutionStatus.CANCELLED)],
            status=ExecutionStatus.CANCELLED,
        ),
    )
    event_record = WorkflowRunEventRecord(
        run_id="run-1",
        session_id="session-1",
        event_id=cancelled_plan_event.id,
        event_type=cancelled_plan_event.type,
        event_payload=cancelled_plan_event,
    )
    uow = _build_uow(
        session_status=SessionStatus.CANCELLED,
        run_status=WorkflowRunStatus.CANCELLED,
        event_records=[event_record],
    )
    coordinator = _coordinator_with_uow(uow)

    snapshot = asyncio.run(coordinator.build_snapshot("session-1"))

    assert snapshot.has_continuable_cancelled_plan is True


def test_start_run_should_create_run_bind_workspace_and_mark_running() -> None:
    uow = _build_uow(
        session_status=SessionStatus.PENDING,
        run_status=WorkflowRunStatus.PENDING,
        workspace_run_id=None,
    )
    uow.session.session.current_run_id = None
    coordinator = _coordinator_with_uow(uow)

    snapshot = asyncio.run(
        coordinator.start_run(
            "session-1",
            sandbox_id="sandbox-1",
            task_id="task-1",
            thread_id="thread-1",
        )
    )

    assert snapshot.run_id == "run-started-1"
    assert snapshot.session_status == SessionStatus.RUNNING
    assert snapshot.run_status == WorkflowRunStatus.RUNNING
    assert uow.session.session.current_run_id == "run-started-1"
    assert uow.workspace.workspace.current_run_id == "run-started-1"
    assert uow.workspace.workspace.sandbox_id == "sandbox-1"
    assert uow.workspace.workspace.task_id == "task-1"


def test_reopen_running_run_should_bind_environment_and_mark_running() -> None:
    uow = _build_uow(
        session_status=SessionStatus.WAITING,
        run_status=WorkflowRunStatus.WAITING,
    )
    coordinator = _coordinator_with_uow(uow)

    snapshot = asyncio.run(
        coordinator.reopen_running_run(
            "session-1",
            run_id="run-1",
            sandbox_id="sandbox-1",
            task_id="task-1",
        )
    )

    assert snapshot.run_id == "run-1"
    assert snapshot.session_status == SessionStatus.RUNNING
    assert snapshot.run_status == WorkflowRunStatus.RUNNING
    assert uow.session.session.current_run_id == "run-1"
    assert uow.workflow_run.run.status == WorkflowRunStatus.RUNNING
    assert uow.workspace.workspace.current_run_id == "run-1"
    assert uow.workspace.workspace.sandbox_id == "sandbox-1"
    assert uow.workspace.workspace.task_id == "task-1"


def test_persist_runtime_event_should_mark_waiting_in_one_transaction() -> None:
    uow = _build_uow()
    coordinator = _coordinator_with_uow(uow)
    event = WaitEvent(id="wait-event-1", payload={"kind": "confirm"})

    result = asyncio.run(coordinator.persist_runtime_event("session-1", event))

    assert result.event_inserted is True
    assert result.transition_applied is True
    assert result.to_session_status == SessionStatus.WAITING
    assert result.to_run_status == WorkflowRunStatus.WAITING
    assert uow.session.session.status == SessionStatus.WAITING
    assert uow.workflow_run.run.status == WorkflowRunStatus.WAITING
    assert uow.workflow_run.run.last_event_at == event.created_at
    assert uow.committed is True


def test_accept_user_message_should_persist_with_stream_id_and_mark_running() -> None:
    uow = _build_uow()
    coordinator = _coordinator_with_uow(uow)
    event = MessageEvent(id="original-event-id", role="user", message="hello")

    result = asyncio.run(
        coordinator.accept_user_message(
            "session-1",
            event,
            latest_message_at=event.created_at,
            stream_event_id="input-stream-1",
        )
    )

    assert result.event_inserted is True
    assert result.to_session_status == SessionStatus.RUNNING
    assert result.to_run_status == WorkflowRunStatus.RUNNING
    assert event.id == "original-event-id"
    assert uow.workflow_run.event_records[0].event_id == "input-stream-1"
    assert uow.workflow_run.event_records[0].event_payload.id == "input-stream-1"
    assert uow.session.session.latest_message == "hello"


def test_mark_resume_requested_should_use_pending_interrupt_to_mark_running() -> None:
    uow = _build_uow(
        session_status=SessionStatus.WAITING,
        run_status=WorkflowRunStatus.WAITING,
    )
    coordinator = _coordinator_with_uow(uow)

    result = asyncio.run(
        coordinator.mark_resume_requested(
            "session-1",
            request_id="request-1",
            pending_interrupt={"kind": "confirm"},
        )
    )

    assert result.transition_applied is True
    assert result.to_session_status == SessionStatus.RUNNING
    assert result.to_run_status == WorkflowRunStatus.RUNNING
    assert uow.session.session.status == SessionStatus.RUNNING
    assert uow.workflow_run.run.status == WorkflowRunStatus.RUNNING


def test_persist_runtime_event_should_mark_completed_and_update_projection() -> None:
    uow = _build_uow()
    coordinator = _coordinator_with_uow(uow)
    event = DoneEvent(id="done-event-1")
    projection = RuntimeEventProjection(
        latest_message="已完成",
        latest_message_at=event.created_at,
        increment_unread=True,
    )

    result = asyncio.run(
        coordinator.persist_runtime_event(
            "session-1",
            event,
            projection=projection,
        )
    )

    assert result.to_session_status == SessionStatus.COMPLETED
    assert result.to_run_status == WorkflowRunStatus.COMPLETED
    assert uow.session.session.latest_message == "已完成"
    assert uow.session.session.unread_message_count == 1
    assert uow.workflow_run.run.finished_at == event.created_at


def test_persist_runtime_event_should_mark_failed() -> None:
    uow = _build_uow()
    coordinator = _coordinator_with_uow(uow)
    event = ErrorEvent(id="error-event-1", error="boom")

    result = asyncio.run(coordinator.persist_runtime_event("session-1", event))

    assert result.to_session_status == SessionStatus.FAILED
    assert result.to_run_status == WorkflowRunStatus.FAILED
    assert uow.session.session.status == SessionStatus.FAILED
    assert uow.workflow_run.run.status == WorkflowRunStatus.FAILED


def test_persist_runtime_event_should_mark_cancelled_from_cancelled_plan_event() -> None:
    uow = _build_uow()
    coordinator = _coordinator_with_uow(uow)
    event = PlanEvent(
        id="cancel-plan-event-1",
        plan=Plan(
            steps=[Step(id="step-1", status=ExecutionStatus.CANCELLED)],
            status=ExecutionStatus.CANCELLED,
        ),
    )

    result = asyncio.run(coordinator.persist_runtime_event("session-1", event))

    assert result.to_session_status == SessionStatus.CANCELLED
    assert result.to_run_status == WorkflowRunStatus.CANCELLED
    assert uow.session.session.status == SessionStatus.CANCELLED
    assert uow.workflow_run.run.status == WorkflowRunStatus.CANCELLED
    assert uow.workflow_run.replaced_plan_count == 1


def test_persist_runtime_event_should_sync_step_projection_without_status_transition() -> None:
    uow = _build_uow()
    coordinator = _coordinator_with_uow(uow)
    event = StepEvent(
        id="step-event-1",
        step=Step(id="step-1", status=ExecutionStatus.RUNNING),
        status=StepEventStatus.STARTED,
    )

    result = asyncio.run(coordinator.persist_runtime_event("session-1", event))

    assert result.event_inserted is True
    assert result.transition_applied is False
    assert result.ignored_reason == "event_has_no_runtime_status_effect"
    assert uow.workflow_run.upserted_step_count == 1
    assert uow.session.session.status == SessionStatus.RUNNING
    assert uow.workflow_run.run.status == WorkflowRunStatus.RUNNING


def test_persist_runtime_event_should_not_override_cancelled_with_late_done() -> None:
    uow = _build_uow(
        session_status=SessionStatus.CANCELLED,
        run_status=WorkflowRunStatus.CANCELLED,
    )
    coordinator = _coordinator_with_uow(uow)
    event = DoneEvent(id="late-done-event-1")

    result = asyncio.run(coordinator.persist_runtime_event("session-1", event))

    assert result.event_inserted is True
    assert result.transition_applied is False
    assert result.ignored_reason == "terminal_run_ignores_late_event"
    assert uow.workflow_run.status_updates == []
    assert uow.session.session.status == SessionStatus.CANCELLED
    assert uow.workflow_run.run.status == WorkflowRunStatus.CANCELLED


def test_cancel_current_run_should_mark_run_session_and_unfinished_steps_cancelled() -> None:
    plan_event = PlanEvent(
        id="plan-event-1",
        plan=Plan(
            steps=[
                Step(id="step-1", status=ExecutionStatus.COMPLETED),
                Step(id="step-2", status=ExecutionStatus.RUNNING),
            ],
        ),
    )
    event_record = WorkflowRunEventRecord(
        run_id="run-1",
        session_id="session-1",
        event_id=plan_event.id,
        event_type=plan_event.type,
        event_payload=plan_event,
    )
    uow = _build_uow(event_records=[event_record])
    uow.workflow_run.run.current_step_id = "step-2"
    coordinator = _coordinator_with_uow(uow)

    result = asyncio.run(
        coordinator.cancel_current_run(
            "session-1",
            reason="test_cancel",
        )
    )

    assert result.transition_applied is True
    assert result.to_session_status == SessionStatus.CANCELLED
    assert result.to_run_status == WorkflowRunStatus.CANCELLED
    assert uow.session.session.status == SessionStatus.CANCELLED
    assert uow.workflow_run.run.status == WorkflowRunStatus.CANCELLED
    assert uow.workflow_run.run.current_step_id is None
    assert uow.workflow_run.upserted_step_count == 1
    assert uow.workflow_run.replaced_plan_count == 1
    assert uow.workflow_run.unfinished_steps_cancelled_count == 1
    persisted_events = [record.event_payload for record in uow.workflow_run.event_records]
    assert isinstance(persisted_events[-2], StepEvent)
    assert persisted_events[-2].step.id == "step-2"
    assert persisted_events[-2].step.status == ExecutionStatus.CANCELLED
    assert isinstance(persisted_events[-1], PlanEvent)
    assert [step.status for step in persisted_events[-1].plan.steps] == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.CANCELLED,
    ]


def test_cancel_current_run_should_fix_session_when_run_already_cancelled() -> None:
    uow = _build_uow(
        session_status=SessionStatus.RUNNING,
        run_status=WorkflowRunStatus.CANCELLED,
    )
    coordinator = _coordinator_with_uow(uow)

    result = asyncio.run(
        coordinator.cancel_current_run(
            "session-1",
            reason="test_cancel",
        )
    )

    assert result.to_session_status == SessionStatus.CANCELLED
    assert result.to_run_status == WorkflowRunStatus.CANCELLED
    assert uow.session.session.status == SessionStatus.CANCELLED
    assert uow.workflow_run.status_updates == []


def test_cancel_current_run_should_fix_run_when_session_already_cancelled() -> None:
    uow = _build_uow(
        session_status=SessionStatus.CANCELLED,
        run_status=WorkflowRunStatus.RUNNING,
    )
    coordinator = _coordinator_with_uow(uow)

    result = asyncio.run(
        coordinator.cancel_current_run(
            "session-1",
            reason="test_cancel",
        )
    )

    assert result.to_session_status == SessionStatus.CANCELLED
    assert result.to_run_status == WorkflowRunStatus.CANCELLED
    assert uow.session.session.status == SessionStatus.CANCELLED
    assert uow.workflow_run.run.status == WorkflowRunStatus.CANCELLED
    assert uow.workflow_run.unfinished_steps_cancelled_count == 1


def test_cancel_current_run_should_not_cancel_steps_when_run_completed() -> None:
    uow = _build_uow(
        session_status=SessionStatus.COMPLETED,
        run_status=WorkflowRunStatus.COMPLETED,
    )
    coordinator = _coordinator_with_uow(uow)

    result = asyncio.run(
        coordinator.cancel_current_run(
            "session-1",
            reason="late_cancel",
        )
    )

    assert result.to_run_status == WorkflowRunStatus.COMPLETED
    assert result.to_session_status == SessionStatus.COMPLETED
    assert uow.workflow_run.unfinished_steps_cancelled_count == 0


def test_cancel_current_run_should_not_cancel_steps_when_run_failed() -> None:
    uow = _build_uow(
        session_status=SessionStatus.FAILED,
        run_status=WorkflowRunStatus.FAILED,
    )
    coordinator = _coordinator_with_uow(uow)

    result = asyncio.run(
        coordinator.cancel_current_run(
            "session-1",
            reason="late_cancel",
        )
    )

    assert result.to_run_status == WorkflowRunStatus.FAILED
    assert result.to_session_status == SessionStatus.FAILED
    assert uow.workflow_run.unfinished_steps_cancelled_count == 0


def test_reconcile_current_run_should_warn_when_session_and_workspace_run_id_conflict() -> None:
    uow = _build_uow(workspace_run_id="run-1")
    uow.session.session.current_run_id = "stale-run-1"
    coordinator = _coordinator_with_uow(uow)

    result = asyncio.run(
        coordinator.reconcile_current_run(
            "session-1",
            reason="before_chat",
        )
    )

    assert result.snapshot_after.run_id == "run-1"
    assert result.warnings == ["session_current_run_id_mismatch_workspace_current_run_id"]


def test_reconcile_current_run_should_sync_session_to_waiting_when_run_waiting() -> None:
    uow = _build_uow(
        session_status=SessionStatus.RUNNING,
        run_status=WorkflowRunStatus.WAITING,
    )
    uow.session.session.current_run_id = "run-1"
    coordinator = _coordinator_with_uow(uow)

    result = asyncio.run(
        coordinator.reconcile_current_run(
            "session-1",
            reason="before_chat",
        )
    )

    assert result.snapshot_before.session_status == SessionStatus.RUNNING
    assert result.snapshot_after.session_status == SessionStatus.WAITING
    assert result.snapshot_after.run_status == WorkflowRunStatus.WAITING
    assert uow.session.session.status == SessionStatus.WAITING
    assert uow.workflow_run.run.status == WorkflowRunStatus.WAITING
    assert result.actions == ["sync_session_status_from_run_waiting"]
    assert result.warnings == ["session_status_mismatch_run_status"]


def test_sync_graph_state_should_mark_waiting_and_update_metadata_checkpoint_ref() -> None:
    uow = _build_uow()
    uow.workflow_run.run.current_step_id = "step-old"
    coordinator = _coordinator_with_uow(uow)
    state = {
        "run_id": "run-1",
        "current_step_id": "step-1",
        "pending_interrupt": _confirm_interrupt(),
        "graph_metadata": {"projection": {"run_status": "running"}},
        "emitted_events": [],
    }

    snapshot = asyncio.run(
        coordinator.sync_graph_state(
            "run-1",
            state,
            checkpoint_ref=CheckpointRef(namespace="", checkpoint_id="cp-new"),
        )
    )

    assert snapshot.run_status == WorkflowRunStatus.WAITING
    assert snapshot.session_status == SessionStatus.WAITING
    assert snapshot.pending_interrupt == {
        "kind": "confirm",
        "title": "",
        "prompt": "是否继续？",
        "details": "",
        "attachments": [],
        "suggest_user_takeover": "none",
        "confirm_label": "继续",
        "cancel_label": "取消",
        "confirm_resume_value": True,
        "cancel_resume_value": False,
        "emphasis": "default",
    }
    assert snapshot.graph_projection_status == WorkflowRunStatus.RUNNING
    assert uow.workflow_run.run.status == WorkflowRunStatus.WAITING
    assert uow.session.session.status == SessionStatus.WAITING
    assert uow.workflow_run.run.checkpoint_id == "cp-new"
    assert uow.workflow_run.run.current_step_id == "step-1"
    assert uow.workflow_run.runtime_metadata_updates[0]["runtime_metadata"]["graph_state_contract"]["graph_state"][
               "pending_interrupt"
           ] == {
               "kind": "confirm",
               "title": "",
               "prompt": "是否继续？",
               "details": "",
               "attachments": [],
               "suggest_user_takeover": "none",
               "confirm_label": "继续",
               "cancel_label": "取消",
               "confirm_resume_value": True,
               "cancel_resume_value": False,
               "emphasis": "default",
           }


def test_sync_graph_state_should_not_regress_completed_with_stale_pending_interrupt() -> None:
    uow = _build_uow(
        session_status=SessionStatus.COMPLETED,
        run_status=WorkflowRunStatus.COMPLETED,
    )
    uow.workflow_run.run.current_step_id = "step-old"
    coordinator = _coordinator_with_uow(uow)
    state = {
        "run_id": "run-1",
        "current_step_id": "step-stale",
        "pending_interrupt": _confirm_interrupt(),
        "graph_metadata": {"projection": {"run_status": "waiting"}},
        "emitted_events": [],
    }

    snapshot = asyncio.run(
        coordinator.sync_graph_state(
            "run-1",
            state,
            checkpoint_ref=CheckpointRef(namespace="", checkpoint_id="cp-new"),
        )
    )

    assert snapshot.run_status == WorkflowRunStatus.COMPLETED
    assert snapshot.session_status == SessionStatus.COMPLETED
    assert uow.workflow_run.run.status == WorkflowRunStatus.COMPLETED
    assert uow.session.session.status == SessionStatus.COMPLETED
    assert uow.workflow_run.status_updates == []
    assert uow.workflow_run.run.checkpoint_id == "cp-new"
    assert uow.workflow_run.run.current_step_id is None


def test_sync_graph_state_should_fix_session_when_run_already_waiting() -> None:
    uow = _build_uow(
        session_status=SessionStatus.RUNNING,
        run_status=WorkflowRunStatus.WAITING,
    )
    coordinator = _coordinator_with_uow(uow)
    state = {
        "run_id": "run-1",
        "current_step_id": "step-1",
        "pending_interrupt": _confirm_interrupt(),
        "graph_metadata": {"projection": {"run_status": "waiting"}},
        "emitted_events": [],
    }

    snapshot = asyncio.run(coordinator.sync_graph_state("run-1", state))

    assert snapshot.run_status == WorkflowRunStatus.WAITING
    assert snapshot.session_status == SessionStatus.WAITING
    assert uow.workflow_run.status_updates == []
    assert uow.session.session.status == SessionStatus.WAITING


def test_sync_graph_state_should_prefer_terminal_graph_projection_over_stale_pending_interrupt() -> None:
    uow = _build_uow(
        session_status=SessionStatus.RUNNING,
        run_status=WorkflowRunStatus.RUNNING,
    )
    coordinator = _coordinator_with_uow(uow)
    state = {
        "run_id": "run-1",
        "current_step_id": "step-stale",
        "pending_interrupt": _confirm_interrupt(),
        "graph_metadata": {"projection": {"run_status": "completed"}},
        "emitted_events": [],
    }

    snapshot = asyncio.run(coordinator.sync_graph_state("run-1", state))

    assert snapshot.run_status == WorkflowRunStatus.COMPLETED
    assert snapshot.session_status == SessionStatus.COMPLETED
    assert uow.workflow_run.run.status == WorkflowRunStatus.COMPLETED
    assert uow.session.session.status == SessionStatus.COMPLETED
    assert uow.workflow_run.run.current_step_id is None


def test_sync_graph_state_should_not_regress_terminal_session_with_stale_pending_interrupt() -> None:
    uow = _build_uow(
        session_status=SessionStatus.COMPLETED,
        run_status=WorkflowRunStatus.RUNNING,
    )
    coordinator = _coordinator_with_uow(uow)
    state = {
        "run_id": "run-1",
        "current_step_id": "step-stale",
        "pending_interrupt": _confirm_interrupt(),
        "graph_metadata": {"projection": {"run_status": "running"}},
        "emitted_events": [],
    }

    snapshot = asyncio.run(coordinator.sync_graph_state("run-1", state))

    assert snapshot.run_status == WorkflowRunStatus.COMPLETED
    assert snapshot.session_status == SessionStatus.COMPLETED
    assert uow.workflow_run.run.status == WorkflowRunStatus.COMPLETED
    assert uow.session.session.status == SessionStatus.COMPLETED
    assert uow.workflow_run.run.current_step_id is None


def test_sync_graph_state_should_not_override_failed_with_graph_completed() -> None:
    uow = _build_uow(
        session_status=SessionStatus.FAILED,
        run_status=WorkflowRunStatus.FAILED,
    )
    coordinator = _coordinator_with_uow(uow)
    state = {
        "run_id": "run-1",
        "current_step_id": "step-stale",
        "graph_metadata": {"projection": {"run_status": "completed"}},
        "emitted_events": [],
    }

    snapshot = asyncio.run(coordinator.sync_graph_state("run-1", state))

    assert snapshot.run_status == WorkflowRunStatus.FAILED
    assert snapshot.session_status == SessionStatus.FAILED
    assert uow.workflow_run.run.status == WorkflowRunStatus.FAILED
    assert uow.workflow_run.status_updates == []
    assert uow.workflow_run.run.current_step_id is None


def test_sync_graph_state_should_not_override_cancelled_with_graph_completed() -> None:
    uow = _build_uow(
        session_status=SessionStatus.CANCELLED,
        run_status=WorkflowRunStatus.CANCELLED,
    )
    coordinator = _coordinator_with_uow(uow)
    state = {
        "run_id": "run-1",
        "current_step_id": "step-stale",
        "pending_interrupt": _confirm_interrupt(),
        "graph_metadata": {"projection": {"run_status": "completed"}},
        "emitted_events": [],
    }

    snapshot = asyncio.run(coordinator.sync_graph_state("run-1", state))

    assert snapshot.run_status == WorkflowRunStatus.CANCELLED
    assert snapshot.session_status == SessionStatus.CANCELLED
    assert uow.workflow_run.run.status == WorkflowRunStatus.CANCELLED
    assert uow.workflow_run.status_updates == []
    assert uow.workflow_run.run.current_step_id is None
