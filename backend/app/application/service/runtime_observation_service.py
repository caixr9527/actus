#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime observation 应用服务。

该服务只负责把 P0-1 runtime 状态快照转换成前端可观察契约，不推进状态、
不执行 graph，也不生成最终正文。
"""

import logging
from enum import Enum
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

from app.application.errors import NotFoundError, error_keys
from app.application.service.runtime_state_coordinator import RuntimeStateCoordinator
from app.domain.models import (
    DoneEvent,
    ErrorEvent,
    Event,
    ExecutionStatus,
    PlanEvent,
    SessionStatus,
    StepEvent,
    TextStreamDeltaEvent,
    TextStreamEndEvent,
    TextStreamStartEvent,
    WaitEvent,
)
from app.domain.repositories import IUnitOfWork

logger = logging.getLogger(__name__)


class RuntimeAction(str, Enum):
    """前端可请求的 runtime 交互动作。"""

    SEND_MESSAGE = "send_message"
    RESUME = "resume"
    CANCEL = "cancel"
    CONTINUE_CANCELLED = "continue_cancelled"


class RuntimeCapabilityResult(BaseModel):
    """当前 runtime 状态下允许的用户动作。"""

    can_send_message: bool = True
    can_resume: bool = False
    can_cancel: bool = False
    can_continue_cancelled: bool = False
    disabled_reasons: dict[RuntimeAction, str] = Field(default_factory=dict)


class RuntimeCursorResult(BaseModel):
    """持久事件回放游标。"""

    latest_event_id: str | None = None
    replay_from_event_id: str | None = None
    has_more: bool = False


class RuntimeInteractionResult(BaseModel):
    """当前可见的人机交互上下文。"""

    kind: Literal["none", "wait"] = "none"
    interrupt_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class RuntimeObservationResult(BaseModel):
    """会话详情 runtime 观察快照。"""

    session_id: str
    run_id: str | None = None
    status: SessionStatus
    current_step_id: str | None = None
    cursor: RuntimeCursorResult
    capabilities: RuntimeCapabilityResult
    interaction: RuntimeInteractionResult


class RuntimeEventMetaResult(BaseModel):
    """单个事件的 runtime 观察元数据。"""

    session_id: str
    run_id: str | None = None
    status_after_event: SessionStatus | None = None
    current_step_id: str | None = None
    source_event_id: str | None = None
    cursor_event_id: str | None = None
    durability: Literal["persistent", "live_only"] = "persistent"
    visibility: Literal["timeline", "draft", "control", "hidden"] = "timeline"


class RuntimeObservableEventResult(BaseModel):
    """领域事件加 runtime 元数据的统一 envelope。"""

    event: Event
    runtime: RuntimeEventMetaResult


class RuntimeObservationService:
    """构建 runtime observation 快照与事件 envelope 的应用层入口。"""

    def __init__(
            self,
            *,
            uow_factory: Callable[[], IUnitOfWork],
            runtime_state_coordinator: RuntimeStateCoordinator | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._runtime_state_coordinator = runtime_state_coordinator or RuntimeStateCoordinator(
            uow_factory=uow_factory,
        )

    async def build_session_observation(
            self,
            *,
            user_id: str,
            session_id: str,
    ) -> RuntimeObservationResult:
        """构建 session detail 的 runtime 快照，返回前必须执行 before_observation 对账。"""
        await self._ensure_owned_session(user_id=user_id, session_id=session_id)
        reconcile_result = await self._runtime_state_coordinator.reconcile_current_run(
            session_id=session_id,
            reason="before_observation",
        )
        snapshot = reconcile_result.snapshot_after
        status = (
            SessionStatus(snapshot.run_status.value)
            if snapshot.run_status is not None
            else snapshot.session_status
        )
        latest_event_id, latest_wait_event = await self._load_latest_runtime_events(
            session_id=session_id,
            run_id=snapshot.run_id,
        )
        interaction = self._build_interaction(
            status=status,
            latest_wait_event=latest_wait_event,
        )
        capabilities = await self.build_capabilities(
            session_id=session_id,
            status=status,
            latest_wait_event=latest_wait_event,
            has_current_run=snapshot.run_status is not None,
            has_continuable_cancelled_plan=snapshot.has_continuable_cancelled_plan,
        )

        logger.info(
            "构建runtime observation",
            extra={
                "session_id": session_id,
                "run_id": snapshot.run_id,
                "status": status.value,
                "latest_event_id": latest_event_id,
            },
        )
        return RuntimeObservationResult(
            session_id=session_id,
            run_id=snapshot.run_id if snapshot.run_status is not None else None,
            status=status,
            current_step_id=snapshot.current_step_id,
            cursor=RuntimeCursorResult(latest_event_id=latest_event_id),
            capabilities=capabilities,
            interaction=interaction,
        )

    async def build_observable_event(
            self,
            *,
            session_id: str,
            event: Event,
            run_id: str | None,
            source_event_id: str | None,
            cursor_event_id: str | None,
            source: Literal["snapshot", "sse"],
    ) -> RuntimeObservableEventResult:
        """为历史详情或 SSE 事件补充 runtime 元数据。"""
        durability: Literal["persistent", "live_only"] = "persistent"
        visibility: Literal["timeline", "draft", "control", "hidden"] = "timeline"
        if isinstance(event, (TextStreamStartEvent, TextStreamDeltaEvent, TextStreamEndEvent)):
            durability = "live_only"
            visibility = "draft"
            source_event_id = None
            cursor_event_id = None

        return RuntimeObservableEventResult(
            event=event,
            runtime=RuntimeEventMetaResult(
                session_id=session_id,
                run_id=run_id,
                status_after_event=self._resolve_status_after_event(event),
                current_step_id=self._resolve_current_step_id(event),
                source_event_id=source_event_id,
                cursor_event_id=cursor_event_id if durability == "persistent" else None,
                durability=durability,
                visibility=visibility,
            ),
        )

    async def build_capabilities(
            self,
            *,
            session_id: str,
            status: SessionStatus,
            latest_wait_event: WaitEvent | None,
            has_current_run: bool = True,
            has_continuable_cancelled_plan: bool = False,
    ) -> RuntimeCapabilityResult:
        """按 runtime 状态矩阵计算用户动作能力。"""
        reasons: dict[RuntimeAction, str] = {}
        if not has_current_run:
            return RuntimeCapabilityResult(
                can_send_message=True,
                can_resume=False,
                can_cancel=False,
                can_continue_cancelled=False,
                disabled_reasons={
                    RuntimeAction.RESUME: "runtime_wait_required",
                    RuntimeAction.CANCEL: "runtime_not_cancellable",
                    RuntimeAction.CONTINUE_CANCELLED: "runtime_cancelled_plan_unavailable",
                },
            )

        can_send_message = status in {
            SessionStatus.COMPLETED,
            SessionStatus.FAILED,
            SessionStatus.CANCELLED,
        }
        can_resume = status == SessionStatus.WAITING and latest_wait_event is not None
        can_cancel = status in {SessionStatus.RUNNING, SessionStatus.WAITING}
        can_continue_cancelled = (
            status == SessionStatus.CANCELLED
            and has_continuable_cancelled_plan
        )

        if status == SessionStatus.PENDING:
            can_send_message = False
        if not can_send_message:
            reasons[RuntimeAction.SEND_MESSAGE] = f"runtime_{status.value}"
        if not can_resume:
            reasons[RuntimeAction.RESUME] = "runtime_wait_required"
        if not can_cancel:
            reasons[RuntimeAction.CANCEL] = "runtime_not_cancellable"
        if not can_continue_cancelled:
            reasons[RuntimeAction.CONTINUE_CANCELLED] = "runtime_cancelled_plan_unavailable"

        return RuntimeCapabilityResult(
            can_send_message=can_send_message,
            can_resume=can_resume,
            can_cancel=can_cancel,
            can_continue_cancelled=can_continue_cancelled,
            disabled_reasons=reasons,
        )

    async def _ensure_owned_session(self, *, user_id: str, session_id: str) -> None:
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(session_id=session_id, user_id=user_id)
        if session is None:
            raise NotFoundError(
                msg=f"任务会话不存在: {session_id}",
                error_key=error_keys.SESSION_NOT_FOUND,
                error_params={"session_id": session_id},
            )

    async def _load_latest_runtime_events(
            self,
            *,
            session_id: str,
            run_id: str | None,
    ) -> tuple[str | None, WaitEvent | None]:
        async with self._uow_factory() as uow:
            records = await uow.workflow_run.list_event_records_by_session(session_id)

        latest_event_id = records[-1].event_id if records else None
        latest_wait_event = None
        for record in reversed(records):
            if run_id is not None and record.run_id != run_id:
                continue
            if isinstance(record.event_payload, WaitEvent):
                latest_wait_event = record.event_payload
                break
        return latest_event_id, latest_wait_event

    @staticmethod
    def _build_interaction(
            *,
            status: SessionStatus,
            latest_wait_event: WaitEvent | None,
    ) -> RuntimeInteractionResult:
        if status != SessionStatus.WAITING or latest_wait_event is None:
            return RuntimeInteractionResult()
        return RuntimeInteractionResult(
            kind="wait",
            interrupt_id=latest_wait_event.interrupt_id,
            payload=dict(latest_wait_event.payload or {}),
        )

    @staticmethod
    def _resolve_status_after_event(event: Event) -> SessionStatus | None:
        if isinstance(event, WaitEvent):
            return SessionStatus.WAITING
        if isinstance(event, DoneEvent):
            return SessionStatus.COMPLETED
        if isinstance(event, ErrorEvent):
            return SessionStatus.FAILED
        if isinstance(event, PlanEvent) and event.plan.status == ExecutionStatus.CANCELLED:
            return SessionStatus.CANCELLED
        return None

    @staticmethod
    def _resolve_current_step_id(event: Event) -> str | None:
        if isinstance(event, StepEvent):
            return event.step.id
        return None
