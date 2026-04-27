#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 生命周期状态机纯函数。"""

from app.domain.models.runtime_state import RuntimeCommand, RuntimeStateSnapshot, RuntimeTransition
from app.domain.models.session import SessionStatus
from app.domain.models.workflow_run import WorkflowRunStatus


class InvalidRuntimeTransition(ValueError):
    """Runtime 状态机拒绝的非法状态转移。"""

    def __init__(self, command: RuntimeCommand, reason: str) -> None:
        self.command = command
        self.reason = reason
        super().__init__(f"invalid runtime transition: command={command.value}, reason={reason}")


def is_terminal(status: WorkflowRunStatus | SessionStatus) -> bool:
    """判断状态是否已经进入不可被迟到事件覆盖的终态。"""

    return status in {
        WorkflowRunStatus.COMPLETED,
        WorkflowRunStatus.FAILED,
        WorkflowRunStatus.CANCELLED,
        SessionStatus.COMPLETED,
        SessionStatus.FAILED,
        SessionStatus.CANCELLED,
    }


def can_accept_user_message(snapshot: "RuntimeStateSnapshot") -> bool:
    """普通用户消息是否可以进入当前 Runtime。"""

    if _is_waiting(snapshot):
        return False
    return True


def can_resume(snapshot: "RuntimeStateSnapshot") -> bool:
    """当前 Runtime 是否具备从等待点恢复执行的条件。"""

    return (
        _status_value(snapshot.session_status) == SessionStatus.WAITING.value
        and _status_value(snapshot.run_status) == WorkflowRunStatus.WAITING.value
        and snapshot.has_checkpoint
        and bool(snapshot.pending_interrupt)
    )


def can_cancel(snapshot: "RuntimeStateSnapshot") -> bool:
    """当前 Runtime 是否允许取消。"""

    return _effective_status_value(snapshot) in {
        WorkflowRunStatus.RUNNING.value,
        WorkflowRunStatus.WAITING.value,
        SessionStatus.RUNNING.value,
        SessionStatus.WAITING.value,
    }


def can_continue_cancelled(snapshot: "RuntimeStateSnapshot") -> bool:
    """当前 Runtime 是否允许基于已取消计划显式继续。"""

    return (
        _is_cancelled(snapshot)
        and snapshot.has_continuable_cancelled_plan
    )


def resolve_transition(
        snapshot: "RuntimeStateSnapshot",
        command: RuntimeCommand,
) -> "RuntimeTransition":
    """根据当前快照和命令解析状态转移，不产生任何副作用。"""

    if command == RuntimeCommand.START:
        return _resolve_start(snapshot, command)
    if command == RuntimeCommand.USER_MESSAGE:
        return _resolve_user_message(snapshot, command)
    if command == RuntimeCommand.RESUME:
        return _resolve_resume(snapshot, command)
    if command == RuntimeCommand.CONTINUE_CANCELLED:
        return _resolve_continue_cancelled(snapshot, command)
    if command == RuntimeCommand.WAIT:
        return _resolve_runtime_event(snapshot, command, SessionStatus.WAITING, WorkflowRunStatus.WAITING)
    if command == RuntimeCommand.COMPLETE:
        return _resolve_runtime_event(snapshot, command, SessionStatus.COMPLETED, WorkflowRunStatus.COMPLETED)
    if command == RuntimeCommand.FAIL:
        return _resolve_runtime_event(snapshot, command, SessionStatus.FAILED, WorkflowRunStatus.FAILED)
    if command == RuntimeCommand.CANCEL:
        return _resolve_cancel(snapshot, command)
    if command == RuntimeCommand.RECONCILE:
        return _noop(snapshot, command, "reconcile_no_status_change")

    raise InvalidRuntimeTransition(command, "unsupported_command")


def _resolve_start(snapshot: "RuntimeStateSnapshot", command: RuntimeCommand) -> "RuntimeTransition":
    if _effective_status_value(snapshot) != SessionStatus.PENDING.value:
        raise InvalidRuntimeTransition(command, "start_requires_pending_status")
    return _transition(snapshot, command, SessionStatus.RUNNING, WorkflowRunStatus.RUNNING, "start_run")


def _resolve_user_message(snapshot: "RuntimeStateSnapshot", command: RuntimeCommand) -> "RuntimeTransition":
    if _is_waiting(snapshot):
        raise InvalidRuntimeTransition(command, "session_resume_required")
    if _is_running(snapshot):
        return _transition(snapshot, command, SessionStatus.RUNNING, WorkflowRunStatus.RUNNING, "append_user_message")
    if _is_terminal_effective_status(snapshot):
        return _transition(snapshot, command, SessionStatus.RUNNING, WorkflowRunStatus.RUNNING, "start_new_run")
    if _effective_status_value(snapshot) == SessionStatus.PENDING.value:
        return _transition(snapshot, command, SessionStatus.RUNNING, WorkflowRunStatus.RUNNING, "start_run_from_message")
    raise InvalidRuntimeTransition(command, "unsupported_user_message_status")


def _resolve_resume(snapshot: "RuntimeStateSnapshot", command: RuntimeCommand) -> "RuntimeTransition":
    if can_resume(snapshot):
        return _transition(snapshot, command, SessionStatus.RUNNING, WorkflowRunStatus.RUNNING, "resume_from_interrupt")
    if _is_waiting(snapshot):
        raise InvalidRuntimeTransition(command, "resume_checkpoint_unavailable")
    if is_terminal(snapshot.session_status) or _run_is_terminal(snapshot):
        raise InvalidRuntimeTransition(command, "terminal_run_cannot_resume")
    raise InvalidRuntimeTransition(command, "session_not_waiting")


def _resolve_continue_cancelled(
        snapshot: "RuntimeStateSnapshot",
        command: RuntimeCommand,
) -> "RuntimeTransition":
    if can_continue_cancelled(snapshot):
        return _transition(
            snapshot,
            command,
            SessionStatus.RUNNING,
            WorkflowRunStatus.RUNNING,
            "continue_cancelled_plan",
        )
    if _is_cancelled(snapshot):
        raise InvalidRuntimeTransition(command, "cancelled_plan_not_continuable")
    raise InvalidRuntimeTransition(command, "continue_cancelled_requires_cancelled_status")


def _resolve_runtime_event(
        snapshot: "RuntimeStateSnapshot",
        command: RuntimeCommand,
        to_session_status: SessionStatus,
        to_run_status: WorkflowRunStatus,
) -> "RuntimeTransition":
    if _run_is_terminal(snapshot):
        return _noop(snapshot, command, "terminal_run_ignores_late_event")
    if is_terminal(snapshot.session_status):
        return _noop(snapshot, command, "terminal_session_ignores_late_event")
    if _effective_status_value(snapshot) not in {
        SessionStatus.RUNNING.value,
        SessionStatus.WAITING.value,
        WorkflowRunStatus.RUNNING.value,
        WorkflowRunStatus.WAITING.value,
    }:
        raise InvalidRuntimeTransition(command, "runtime_event_requires_active_run")
    return _transition(snapshot, command, to_session_status, to_run_status, f"runtime_event_{command.value}")


def _resolve_cancel(snapshot: "RuntimeStateSnapshot", command: RuntimeCommand) -> "RuntimeTransition":
    if is_terminal(snapshot.session_status) or _run_is_terminal(snapshot):
        return _noop(snapshot, command, "cancel_terminal_run_idempotent")
    if can_cancel(snapshot):
        return _transition(snapshot, command, SessionStatus.CANCELLED, WorkflowRunStatus.CANCELLED, "cancel_active_run")
    raise InvalidRuntimeTransition(command, "cancel_requires_running_or_waiting")


def _transition(
        snapshot: "RuntimeStateSnapshot",
        command: RuntimeCommand,
        to_session_status: SessionStatus,
        to_run_status: WorkflowRunStatus | None,
        reason: str,
) -> "RuntimeTransition":
    return RuntimeTransition(
        command=command,
        from_session_status=snapshot.session_status,
        from_run_status=snapshot.run_status,
        to_session_status=to_session_status,
        to_run_status=to_run_status,
        reason=reason,
    )


def _noop(
        snapshot: "RuntimeStateSnapshot",
        command: RuntimeCommand,
        reason: str,
) -> "RuntimeTransition":
    return _transition(
        snapshot,
        command,
        snapshot.session_status,
        snapshot.run_status,
        reason,
    )


def _run_is_terminal(snapshot: "RuntimeStateSnapshot") -> bool:
    return snapshot.run_status is not None and is_terminal(snapshot.run_status)


def _is_running(snapshot: "RuntimeStateSnapshot") -> bool:
    return _effective_status_value(snapshot) == SessionStatus.RUNNING.value


def _is_waiting(snapshot: "RuntimeStateSnapshot") -> bool:
    return (
        snapshot.session_status == SessionStatus.WAITING
        or snapshot.run_status == WorkflowRunStatus.WAITING
    )


def _is_cancelled(snapshot: "RuntimeStateSnapshot") -> bool:
    return _effective_status_value(snapshot) == SessionStatus.CANCELLED.value


def _is_terminal_effective_status(snapshot: "RuntimeStateSnapshot") -> bool:
    return _effective_status_value(snapshot) in {
        SessionStatus.COMPLETED.value,
        SessionStatus.FAILED.value,
        SessionStatus.CANCELLED.value,
    }


def _effective_status_value(snapshot: "RuntimeStateSnapshot") -> str:
    return _status_value(snapshot.run_status) or snapshot.session_status.value


def _status_value(status: WorkflowRunStatus | SessionStatus | None) -> str:
    if status is None:
        return ""
    return status.value
