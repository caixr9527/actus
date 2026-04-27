#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 状态机合同测试。"""

import pytest

from app.domain.models import (
    RuntimeCommand,
    RuntimeStateSnapshot,
    SessionStatus,
    WorkflowRunStatus,
)
from app.domain.services.runtime.state_machine import (
    InvalidRuntimeTransition,
    can_accept_user_message,
    can_cancel,
    can_continue_cancelled,
    can_resume,
    is_terminal,
    resolve_transition,
)


def _snapshot(
        session_status: SessionStatus,
        run_status: WorkflowRunStatus | None = None,
        *,
        has_checkpoint: bool = False,
        pending_interrupt: dict | None = None,
        has_continuable_cancelled_plan: bool = False,
) -> RuntimeStateSnapshot:
    return RuntimeStateSnapshot(
        session_id="session-1",
        run_id="run-1" if run_status else None,
        session_status=session_status,
        run_status=run_status,
        has_checkpoint=has_checkpoint,
        pending_interrupt=pending_interrupt or {},
        has_continuable_cancelled_plan=has_continuable_cancelled_plan,
    )


def test_resolve_transition_should_start_pending_runtime() -> None:
    snapshot = _snapshot(SessionStatus.PENDING)

    transition = resolve_transition(snapshot, RuntimeCommand.START)

    assert transition.from_session_status == SessionStatus.PENDING
    assert transition.from_run_status is None
    assert transition.to_session_status == SessionStatus.RUNNING
    assert transition.to_run_status == WorkflowRunStatus.RUNNING


def test_resolve_transition_should_accept_message_on_running_runtime() -> None:
    snapshot = _snapshot(SessionStatus.RUNNING, WorkflowRunStatus.RUNNING)

    transition = resolve_transition(snapshot, RuntimeCommand.USER_MESSAGE)

    assert transition.to_session_status == SessionStatus.RUNNING
    assert transition.to_run_status == WorkflowRunStatus.RUNNING
    assert transition.reason == "append_user_message"
    assert can_accept_user_message(snapshot) is True


@pytest.mark.parametrize(
    ("session_status", "run_status"),
    [
        (SessionStatus.COMPLETED, WorkflowRunStatus.COMPLETED),
        (SessionStatus.FAILED, WorkflowRunStatus.FAILED),
        (SessionStatus.CANCELLED, WorkflowRunStatus.CANCELLED),
    ],
)
def test_resolve_transition_should_start_new_run_after_terminal_user_message(
        session_status: SessionStatus,
        run_status: WorkflowRunStatus,
) -> None:
    snapshot = _snapshot(session_status, run_status)

    transition = resolve_transition(snapshot, RuntimeCommand.USER_MESSAGE)

    assert transition.to_session_status == SessionStatus.RUNNING
    assert transition.to_run_status == WorkflowRunStatus.RUNNING
    assert transition.reason == "start_new_run"


def test_resolve_transition_should_reject_user_message_while_waiting() -> None:
    snapshot = _snapshot(SessionStatus.WAITING, WorkflowRunStatus.WAITING)

    with pytest.raises(InvalidRuntimeTransition) as exc_info:
        resolve_transition(snapshot, RuntimeCommand.USER_MESSAGE)

    assert exc_info.value.reason == "session_resume_required"
    assert can_accept_user_message(snapshot) is False


def test_resolve_transition_should_reject_user_message_when_session_waiting_but_run_running() -> None:
    snapshot = _snapshot(SessionStatus.WAITING, WorkflowRunStatus.RUNNING)

    with pytest.raises(InvalidRuntimeTransition) as exc_info:
        resolve_transition(snapshot, RuntimeCommand.USER_MESSAGE)

    assert exc_info.value.reason == "session_resume_required"
    assert can_accept_user_message(snapshot) is False


def test_resolve_transition_should_resume_waiting_runtime_with_interrupt_checkpoint() -> None:
    snapshot = _snapshot(
        SessionStatus.WAITING,
        WorkflowRunStatus.WAITING,
        has_checkpoint=True,
        pending_interrupt={"value": {"kind": "confirm"}},
    )

    transition = resolve_transition(snapshot, RuntimeCommand.RESUME)

    assert transition.to_session_status == SessionStatus.RUNNING
    assert transition.to_run_status == WorkflowRunStatus.RUNNING
    assert transition.reason == "resume_from_interrupt"
    assert can_resume(snapshot) is True


@pytest.mark.parametrize(
    "snapshot",
    [
        _snapshot(SessionStatus.RUNNING, WorkflowRunStatus.RUNNING),
        _snapshot(SessionStatus.WAITING, WorkflowRunStatus.WAITING, has_checkpoint=True),
        _snapshot(
            SessionStatus.WAITING,
            WorkflowRunStatus.WAITING,
            has_checkpoint=False,
            pending_interrupt={"value": {"kind": "confirm"}},
        ),
        _snapshot(SessionStatus.COMPLETED, WorkflowRunStatus.COMPLETED),
        _snapshot(SessionStatus.FAILED, WorkflowRunStatus.FAILED),
        _snapshot(SessionStatus.CANCELLED, WorkflowRunStatus.CANCELLED),
    ],
)
def test_resolve_transition_should_reject_invalid_resume(snapshot: RuntimeStateSnapshot) -> None:
    with pytest.raises(InvalidRuntimeTransition):
        resolve_transition(snapshot, RuntimeCommand.RESUME)


def test_resolve_transition_should_mark_running_runtime_waiting_on_wait_event() -> None:
    snapshot = _snapshot(SessionStatus.RUNNING, WorkflowRunStatus.RUNNING)

    transition = resolve_transition(snapshot, RuntimeCommand.WAIT)

    assert transition.to_session_status == SessionStatus.WAITING
    assert transition.to_run_status == WorkflowRunStatus.WAITING


@pytest.mark.parametrize(
    ("command", "expected_session_status", "expected_run_status"),
    [
        (RuntimeCommand.COMPLETE, SessionStatus.COMPLETED, WorkflowRunStatus.COMPLETED),
        (RuntimeCommand.FAIL, SessionStatus.FAILED, WorkflowRunStatus.FAILED),
    ],
)
def test_resolve_transition_should_mark_runtime_terminal_on_done_or_error_event(
        command: RuntimeCommand,
        expected_session_status: SessionStatus,
        expected_run_status: WorkflowRunStatus,
) -> None:
    snapshot = _snapshot(SessionStatus.RUNNING, WorkflowRunStatus.RUNNING)

    transition = resolve_transition(snapshot, command)

    assert transition.to_session_status == expected_session_status
    assert transition.to_run_status == expected_run_status


@pytest.mark.parametrize(
    "snapshot",
    [
        _snapshot(SessionStatus.RUNNING, WorkflowRunStatus.RUNNING),
        _snapshot(SessionStatus.WAITING, WorkflowRunStatus.WAITING),
    ],
)
def test_resolve_transition_should_cancel_active_runtime(snapshot: RuntimeStateSnapshot) -> None:
    transition = resolve_transition(snapshot, RuntimeCommand.CANCEL)

    assert transition.to_session_status == SessionStatus.CANCELLED
    assert transition.to_run_status == WorkflowRunStatus.CANCELLED
    assert can_cancel(snapshot) is True


@pytest.mark.parametrize(
    ("session_status", "run_status", "late_command"),
    [
        (SessionStatus.COMPLETED, WorkflowRunStatus.COMPLETED, RuntimeCommand.WAIT),
        (SessionStatus.COMPLETED, WorkflowRunStatus.COMPLETED, RuntimeCommand.FAIL),
        (SessionStatus.FAILED, WorkflowRunStatus.FAILED, RuntimeCommand.COMPLETE),
        (SessionStatus.FAILED, WorkflowRunStatus.FAILED, RuntimeCommand.WAIT),
        (SessionStatus.CANCELLED, WorkflowRunStatus.CANCELLED, RuntimeCommand.COMPLETE),
        (SessionStatus.CANCELLED, WorkflowRunStatus.CANCELLED, RuntimeCommand.FAIL),
        (SessionStatus.CANCELLED, WorkflowRunStatus.CANCELLED, RuntimeCommand.WAIT),
    ],
)
def test_resolve_transition_should_ignore_late_runtime_event_after_terminal_status(
        session_status: SessionStatus,
        run_status: WorkflowRunStatus,
        late_command: RuntimeCommand,
) -> None:
    snapshot = _snapshot(session_status, run_status)

    transition = resolve_transition(snapshot, late_command)

    assert transition.to_session_status == session_status
    assert transition.to_run_status == run_status
    assert transition.reason == "terminal_run_ignores_late_event"


@pytest.mark.parametrize(
    ("session_status", "run_status"),
    [
        (SessionStatus.COMPLETED, WorkflowRunStatus.COMPLETED),
        (SessionStatus.FAILED, WorkflowRunStatus.FAILED),
        (SessionStatus.CANCELLED, WorkflowRunStatus.CANCELLED),
    ],
)
def test_resolve_transition_should_make_cancel_terminal_status_idempotent(
        session_status: SessionStatus,
        run_status: WorkflowRunStatus,
) -> None:
    snapshot = _snapshot(session_status, run_status)

    transition = resolve_transition(snapshot, RuntimeCommand.CANCEL)

    assert transition.to_session_status == session_status
    assert transition.to_run_status == run_status
    assert transition.reason == "cancel_terminal_run_idempotent"


def test_resolve_transition_should_continue_cancelled_when_plan_is_continuable() -> None:
    snapshot = _snapshot(
        SessionStatus.CANCELLED,
        WorkflowRunStatus.CANCELLED,
        has_continuable_cancelled_plan=True,
    )

    transition = resolve_transition(snapshot, RuntimeCommand.CONTINUE_CANCELLED)

    assert transition.to_session_status == SessionStatus.RUNNING
    assert transition.to_run_status == WorkflowRunStatus.RUNNING
    assert transition.reason == "continue_cancelled_plan"
    assert can_continue_cancelled(snapshot) is True


def test_resolve_transition_should_reject_continue_cancelled_without_continuable_plan() -> None:
    snapshot = _snapshot(
        SessionStatus.CANCELLED,
        WorkflowRunStatus.CANCELLED,
        has_continuable_cancelled_plan=False,
    )

    with pytest.raises(InvalidRuntimeTransition) as exc_info:
        resolve_transition(snapshot, RuntimeCommand.CONTINUE_CANCELLED)

    assert exc_info.value.reason == "cancelled_plan_not_continuable"
    assert can_continue_cancelled(snapshot) is False


@pytest.mark.parametrize(
    "status",
    [
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
        SessionStatus.CANCELLED,
        WorkflowRunStatus.COMPLETED,
        WorkflowRunStatus.FAILED,
        WorkflowRunStatus.CANCELLED,
    ],
)
def test_is_terminal_should_detect_terminal_statuses(
        status: SessionStatus | WorkflowRunStatus,
) -> None:
    assert is_terminal(status) is True


@pytest.mark.parametrize(
    "status",
    [
        SessionStatus.PENDING,
        SessionStatus.RUNNING,
        SessionStatus.WAITING,
        WorkflowRunStatus.PENDING,
        WorkflowRunStatus.RUNNING,
        WorkflowRunStatus.WAITING,
    ],
)
def test_is_terminal_should_reject_non_terminal_statuses(
        status: SessionStatus | WorkflowRunStatus,
) -> None:
    assert is_terminal(status) is False
