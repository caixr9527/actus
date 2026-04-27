#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 状态一致性协调服务。"""

import logging
from typing import Any, Callable

from app.domain.models import (
    BaseEvent,
    CheckpointRef,
    DoneEvent,
    ErrorEvent,
    ExecutionStatus,
    MessageEvent,
    PlanEvent,
    RuntimeCommand,
    RuntimeEventPersistResult,
    RuntimeEventProjection,
    RuntimeReconcileResult,
    RuntimeStateSnapshot,
    RuntimeStateSource,
    SessionStatus,
    StepEvent,
    WaitEvent,
    WorkflowRunStatus,
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.state_machine import (
    is_terminal,
    resolve_transition,
)
from app.domain.services.runtime.langgraph_state import GraphStateContractMapper, get_graph_projection
from app.domain.services.runtime.cancellation import build_cancelled_runtime_events

logger = logging.getLogger(__name__)


class RuntimeStateCoordinator:
    """Runtime 状态转移的应用层唯一业务入口。"""

    def __init__(self, uow_factory: Callable[[], IUnitOfWork]) -> None:
        self.uow_factory = uow_factory

    async def build_snapshot(
            self,
            session_id: str,
            *,
            source: RuntimeStateSource = RuntimeStateSource.RECONCILE,
            uow: IUnitOfWork | None = None,
    ) -> RuntimeStateSnapshot:
        """构建当前 Runtime 快照，供状态机和后续对账使用。"""
        if uow is not None:
            return await self._build_snapshot_in_uow(
                uow=uow,
                session_id=session_id,
                source=source,
            )

        async with self.uow_factory() as active_uow:
            return await self._build_snapshot_in_uow(
                uow=active_uow,
                session_id=session_id,
                source=source,
            )

    async def persist_runtime_event(
            self,
            session_id: str,
            event: BaseEvent,
            *,
            projection: RuntimeEventProjection | None = None,
            uow: IUnitOfWork | None = None,
            allow_status_transition: bool = True,
    ) -> RuntimeEventPersistResult:
        """同事务持久化 Runtime 事件，并按状态机收敛 run/session 状态。"""
        projection = projection or RuntimeEventProjection()
        if uow is not None:
            return await self._persist_runtime_event_in_uow(
                uow=uow,
                session_id=session_id,
                event=event,
                projection=projection,
                allow_status_transition=allow_status_transition,
            )

        async with self.uow_factory() as uow:
            return await self._persist_runtime_event_in_uow(
                uow=uow,
                session_id=session_id,
                event=event,
                projection=projection,
                allow_status_transition=allow_status_transition,
            )

    async def accept_user_message(
            self,
            session_id: str,
            event: MessageEvent,
            *,
            latest_message_at,
            stream_event_id: str,
    ) -> RuntimeEventPersistResult:
        """持久化用户输入事件，并把 run/session 收敛到 running。"""
        persisted_event = event.model_copy(deep=True)
        persisted_event.id = stream_event_id
        projection = RuntimeEventProjection(
            latest_message=event.message,
            latest_message_at=latest_message_at,
        )
        async with self.uow_factory() as uow:
            return await self._persist_user_input_in_uow(
                uow=uow,
                session_id=session_id,
                event=persisted_event,
                command=RuntimeCommand.USER_MESSAGE,
                projection=projection,
            )

    async def mark_resume_requested(
            self,
            session_id: str,
            *,
            request_id: str,
            pending_interrupt: dict | None = None,
    ) -> RuntimeEventPersistResult:
        """恢复输入进入队列后，把等待态 run/session 收敛到 running。"""
        async with self.uow_factory() as uow:
            return await self._apply_input_transition_in_uow(
                uow=uow,
                session_id=session_id,
                command=RuntimeCommand.RESUME,
                trigger=request_id,
                pending_interrupt=pending_interrupt,
            )

    async def mark_continue_cancelled_requested(
            self,
            session_id: str,
            *,
            request_id: str,
    ) -> RuntimeEventPersistResult:
        """继续已取消任务的新 run 已创建后，确认 run/session 为 running。

        注意：该方法不负责 cancelled plan 可继续性的原始校验；
        调用前必须已完成 continue_cancelled 可用性校验并创建新 run。
        """
        async with self.uow_factory() as uow:
            return await self._apply_input_transition_in_uow(
                uow=uow,
                session_id=session_id,
                command=RuntimeCommand.USER_MESSAGE,
                trigger=request_id,
            )

    async def reconcile_current_run(
            self,
            session_id: str,
            *,
            reason: str,
    ) -> RuntimeReconcileResult:
        """PR-2 只提供只读对账骨架，状态修复由后续 PR 接入。"""
        snapshot = await self.build_snapshot(
            session_id=session_id,
            source=RuntimeStateSource.RECONCILE,
        )
        warnings = []
        if snapshot.session_run_id and snapshot.workspace_run_id:
            if snapshot.session_run_id != snapshot.workspace_run_id:
                warnings.append("session_current_run_id_mismatch_workspace_current_run_id")
        logger.info(
            "运行状态对账完成",
            extra={
                "session_id": session_id,
                "run_id": snapshot.run_id,
                "trigger": reason,
                "warnings": warnings,
            },
        )
        return RuntimeReconcileResult(
            snapshot_before=snapshot,
            snapshot_after=snapshot,
            actions=[],
            warnings=warnings,
        )

    async def cancel_current_run(
            self,
            session_id: str,
            *,
            reason: str,
    ) -> RuntimeEventPersistResult:
        """取消当前运行，并统一收敛 run/session/未完成 step 为 cancelled。"""
        async with self.uow_factory() as uow:
            snapshot = await self._build_snapshot_in_uow(
                uow=uow,
                session_id=session_id,
                source=RuntimeStateSource.CANCEL,
            )
            if snapshot.run_id is None:
                await uow.session.update_runtime_state(
                    session_id,
                    status=SessionStatus.CANCELLED,
                )
                return RuntimeEventPersistResult(
                    event_inserted=False,
                    transition_applied=snapshot.session_status != SessionStatus.CANCELLED,
                    from_session_status=snapshot.session_status,
                    to_session_status=SessionStatus.CANCELLED,
                    from_run_status=snapshot.run_status,
                    to_run_status=snapshot.run_status,
                    ignored_reason="cancel_without_current_run",
                )

            run = await uow.workflow_run.get_by_id_for_update(snapshot.run_id)
            if run is None:
                await uow.session.update_runtime_state(
                    session_id,
                    status=SessionStatus.CANCELLED,
                )
                return RuntimeEventPersistResult(
                    event_inserted=False,
                    transition_applied=snapshot.session_status != SessionStatus.CANCELLED,
                    from_session_status=snapshot.session_status,
                    to_session_status=SessionStatus.CANCELLED,
                    from_run_status=snapshot.run_status,
                    to_run_status=snapshot.run_status,
                    ignored_reason="cancel_current_run_missing",
                )

            event_inserted = False
            cancelled_plan_event = None
            cancelled_step_event = None
            should_write_cancel_events = (
                    snapshot.run_status is not None
                    and snapshot.run_status != WorkflowRunStatus.CANCELLED
                    and not is_terminal(snapshot.run_status)
                    and (
                            not is_terminal(snapshot.session_status)
                            or snapshot.session_status == SessionStatus.CANCELLED
                    )
            )
            if should_write_cancel_events:
                run_events = await uow.workflow_run.list_events(snapshot.run_id)
                runtime_metadata = run.runtime_metadata if isinstance(run.runtime_metadata, dict) else {}
                cancelled_plan_event, cancelled_step_event = build_cancelled_runtime_events(
                    runtime_metadata,
                    run_events=run_events,
                    current_step_id=run.current_step_id,
                )
                for event in [cancelled_step_event, cancelled_plan_event]:
                    if event is None:
                        continue
                    inserted = await uow.workflow_run.add_event_record_if_absent(
                        session_id=session_id,
                        run_id=snapshot.run_id,
                        event=event,
                    )
                    event_inserted = event_inserted or inserted
                    await self._sync_step_projection(
                        uow=uow,
                        run_id=snapshot.run_id,
                        event=event,
                    )

            result = await self._apply_command_transition_in_uow(
                uow=uow,
                session_id=session_id,
                snapshot=snapshot,
                command=RuntimeCommand.CANCEL,
                projection=RuntimeEventProjection(),
                event_inserted=event_inserted,
                trigger_event=cancelled_plan_event or cancelled_step_event,
            )
            result.event_inserted = event_inserted
            result.ignored_reason = result.ignored_reason or reason
            should_cancel_unfinished_steps = (
                    snapshot.run_status is not None
                    and snapshot.run_status != WorkflowRunStatus.CANCELLED
                    and not is_terminal(snapshot.run_status)
            )
            if should_cancel_unfinished_steps:
                await uow.workflow_run.mark_unfinished_steps_cancelled(snapshot.run_id)
            if snapshot.session_status == SessionStatus.CANCELLED and snapshot.run_status != WorkflowRunStatus.CANCELLED:
                await uow.workflow_run.update_status(
                    snapshot.run_id,
                    status=WorkflowRunStatus.CANCELLED,
                    current_step_id=None,
                )
                result.to_run_status = WorkflowRunStatus.CANCELLED
                result.transition_applied = True
            elif snapshot.run_status == WorkflowRunStatus.CANCELLED and snapshot.session_status != SessionStatus.CANCELLED:
                await uow.session.update_runtime_state(
                    session_id,
                    status=SessionStatus.CANCELLED,
                )
                result.to_session_status = SessionStatus.CANCELLED
                result.transition_applied = True
            return result

    async def sync_graph_state(
            self,
            run_id: str,
            state: dict[str, Any],
            *,
            checkpoint_ref: CheckpointRef | None = None,
            source: RuntimeStateSource = RuntimeStateSource.GRAPH_STATE,
    ) -> RuntimeStateSnapshot:
        """同步 graph state 投影，并按统一规则对账 run/session 状态。"""
        runtime_metadata = GraphStateContractMapper.build_runtime_metadata(state)
        current_step_id = state.get("current_step_id")
        pending_interrupt = GraphStateContractMapper.get_pending_interrupt(state)
        graph_projection_status = self._resolve_graph_projection_status(state)

        async with self.uow_factory() as uow:
            run = await uow.workflow_run.get_by_id_for_update(run_id)
            if run is None:
                raise ValueError(f"运行[{run_id}]不存在，无法同步graph state")

            snapshot = await self._build_snapshot_in_uow(
                uow=uow,
                session_id=run.session_id,
                source=source,
            )
            if snapshot.run_id != run_id:
                logger.warning(
                    "graph state同步目标不是当前运行",
                    extra={
                        "session_id": run.session_id,
                        "run_id": run_id,
                        "current_run_id": snapshot.run_id,
                        "trigger": source.value,
                    },
                )
                return snapshot

            if checkpoint_ref is not None:
                await uow.workflow_run.update_checkpoint_ref(
                    run_id=run_id,
                    checkpoint_namespace=checkpoint_ref.namespace,
                    checkpoint_id=checkpoint_ref.checkpoint_id,
                )

            await uow.workflow_run.update_runtime_metadata(
                run_id=run_id,
                runtime_metadata=runtime_metadata,
                current_step_id=(
                    None
                    if snapshot.run_status is not None and is_terminal(snapshot.run_status)
                    else current_step_id
                ),
            )

            to_run_status = self._resolve_graph_reconciled_run_status(
                snapshot=snapshot,
                pending_interrupt=pending_interrupt,
                graph_projection_status=graph_projection_status,
            )
            to_session_status = SessionStatus(to_run_status.value) if to_run_status is not None else None
            run_status_changed = to_run_status is not None and to_run_status != snapshot.run_status
            session_status_changed = to_session_status is not None and to_session_status != snapshot.session_status
            if run_status_changed or session_status_changed:
                if run_status_changed:
                    await uow.workflow_run.update_status(
                        run_id,
                        status=to_run_status,
                        current_step_id=(
                            None
                            if is_terminal(to_run_status)
                            else current_step_id
                        ),
                    )
                await uow.session.update_runtime_state(
                    run.session_id,
                    status=to_session_status,
                )
                logger.info(
                    "graph state对账完成",
                    extra={
                        "session_id": run.session_id,
                        "run_id": run_id,
                        "from_status": snapshot.run_status.value if snapshot.run_status else None,
                        "to_status": to_run_status.value if to_run_status else None,
                        "trigger": source.value,
                        "checkpoint_id": checkpoint_ref.checkpoint_id if checkpoint_ref else None,
                        "pending_interrupt": bool(pending_interrupt),
                    },
                )

            after_snapshot = await self._build_snapshot_in_uow(
                uow=uow,
                session_id=run.session_id,
                source=source,
            )
            after_snapshot.pending_interrupt = pending_interrupt
            after_snapshot.graph_projection_status = graph_projection_status
            return after_snapshot

    async def _persist_runtime_event_in_uow(
            self,
            *,
            uow: IUnitOfWork,
            session_id: str,
            event: BaseEvent,
            projection: RuntimeEventProjection,
            allow_status_transition: bool,
    ) -> RuntimeEventPersistResult:
        snapshot = await self._build_snapshot_in_uow(
            uow=uow,
            session_id=session_id,
            source=RuntimeStateSource.STREAM_EVENT,
        )
        if snapshot.run_id is None:
            raise RuntimeError(f"会话[{session_id}]缺少current run，无法写入Runtime事件")

        event_inserted = await uow.workflow_run.add_event_record_if_absent(
            session_id=session_id,
            run_id=snapshot.run_id,
            event=event,
        )
        await self._sync_step_projection(
            uow=uow,
            run_id=snapshot.run_id,
            event=event,
        )

        command = self._command_from_event(event)
        if command is None or not allow_status_transition:
            await self._update_session_projection(
                uow=uow,
                session_id=session_id,
                status=snapshot.session_status,
                projection=projection,
                event_inserted=event_inserted,
            )
            return RuntimeEventPersistResult(
                event_inserted=event_inserted,
                transition_applied=False,
                from_session_status=snapshot.session_status,
                to_session_status=snapshot.session_status,
                from_run_status=snapshot.run_status,
                to_run_status=snapshot.run_status,
                ignored_reason=(
                    "status_transition_deferred"
                    if command is not None and not allow_status_transition
                    else "event_has_no_runtime_status_effect"
                ),
            )

        transition = resolve_transition(snapshot, command)
        transition_applied = (
                transition.to_session_status != transition.from_session_status
                or transition.to_run_status != transition.from_run_status
        )
        if transition_applied:
            await self._apply_transition(
                uow=uow,
                snapshot=snapshot,
                session_id=session_id,
                run_id=snapshot.run_id,
                event=event,
                transition=transition,
                projection=projection,
                event_inserted=event_inserted,
            )
            self._log_transition(snapshot=snapshot, transition=transition)
        else:
            await self._update_session_projection(
                uow=uow,
                session_id=session_id,
                status=snapshot.session_status,
                projection=projection,
                event_inserted=event_inserted,
            )

        return RuntimeEventPersistResult(
            event_inserted=event_inserted,
            transition_applied=transition_applied,
            from_session_status=transition.from_session_status,
            to_session_status=transition.to_session_status,
            from_run_status=transition.from_run_status,
            to_run_status=transition.to_run_status,
            ignored_reason="" if transition_applied else transition.reason,
        )

    async def _persist_user_input_in_uow(
            self,
            *,
            uow: IUnitOfWork,
            session_id: str,
            event: MessageEvent,
            command: RuntimeCommand,
            projection: RuntimeEventProjection,
    ) -> RuntimeEventPersistResult:
        snapshot = await self._build_snapshot_in_uow(
            uow=uow,
            session_id=session_id,
            source=RuntimeStateSource.REQUEST,
        )
        if snapshot.run_id is None:
            raise RuntimeError(f"会话[{session_id}]缺少current run，无法写入用户输入事件")

        event_inserted = await uow.workflow_run.add_event_record_if_absent(
            session_id=session_id,
            run_id=snapshot.run_id,
            event=event,
        )
        result = await self._apply_command_transition_in_uow(
            uow=uow,
            session_id=session_id,
            snapshot=snapshot,
            command=command,
            projection=projection,
            event_inserted=event_inserted,
            trigger_event=event,
        )
        result.event_inserted = event_inserted
        return result

    async def _apply_input_transition_in_uow(
            self,
            *,
            uow: IUnitOfWork,
            session_id: str,
            command: RuntimeCommand,
            trigger: str,
            pending_interrupt: dict | None = None,
    ) -> RuntimeEventPersistResult:
        snapshot = await self._build_snapshot_in_uow(
            uow=uow,
            session_id=session_id,
            source=RuntimeStateSource.REQUEST,
        )
        if pending_interrupt is not None:
            snapshot.pending_interrupt = pending_interrupt
            snapshot.has_checkpoint = True
        if snapshot.run_id is None:
            raise RuntimeError(f"会话[{session_id}]缺少current run，无法更新输入状态")
        result = await self._apply_command_transition_in_uow(
            uow=uow,
            session_id=session_id,
            snapshot=snapshot,
            command=command,
            projection=RuntimeEventProjection(),
            event_inserted=False,
            trigger_event=None,
        )
        result.ignored_reason = result.ignored_reason or trigger
        return result

    async def _apply_command_transition_in_uow(
            self,
            *,
            uow: IUnitOfWork,
            session_id: str,
            snapshot: RuntimeStateSnapshot,
            command: RuntimeCommand,
            projection: RuntimeEventProjection,
            event_inserted: bool,
            trigger_event: BaseEvent | None,
    ) -> RuntimeEventPersistResult:
        transition = resolve_transition(snapshot, command)
        transition_applied = (
                transition.to_session_status != transition.from_session_status
                or transition.to_run_status != transition.from_run_status
        )
        if transition_applied:
            if snapshot.run_id is None:
                raise RuntimeError(f"会话[{session_id}]缺少current run，无法更新Runtime状态")
            await self._apply_transition(
                uow=uow,
                snapshot=snapshot,
                session_id=session_id,
                run_id=snapshot.run_id,
                event=trigger_event,
                transition=transition,
                projection=projection,
                event_inserted=event_inserted,
            )
            self._log_transition(snapshot=snapshot, transition=transition)
        else:
            await self._update_session_projection(
                uow=uow,
                session_id=session_id,
                status=snapshot.session_status,
                projection=projection,
                event_inserted=event_inserted,
            )

        return RuntimeEventPersistResult(
            event_inserted=event_inserted,
            transition_applied=transition_applied,
            from_session_status=transition.from_session_status,
            to_session_status=transition.to_session_status,
            from_run_status=transition.from_run_status,
            to_run_status=transition.to_run_status,
            ignored_reason="" if transition_applied else transition.reason,
        )

    async def _build_snapshot_in_uow(
            self,
            *,
            uow: IUnitOfWork,
            session_id: str,
            source: RuntimeStateSource,
    ) -> RuntimeStateSnapshot:
        session = await uow.session.get_by_id_for_update(session_id)
        if session is None:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

        workspace = None
        if session.workspace_id:
            workspace = await uow.workspace.get_by_id(session.workspace_id)
        if workspace is None:
            workspace = await uow.workspace.get_by_session_id(session_id)

        run_id = str(workspace.current_run_id).strip() if workspace and workspace.current_run_id else None
        session_run_id = str(session.current_run_id).strip() if session.current_run_id else None
        if not run_id:
            run_id = session_run_id

        run = await uow.workflow_run.get_by_id_for_update(run_id) if run_id else None
        checkpoint_ref = None
        if run and (run.checkpoint_namespace or run.checkpoint_id):
            checkpoint_ref = CheckpointRef(
                namespace=run.checkpoint_namespace or "",
                checkpoint_id=run.checkpoint_id,
            )

        return RuntimeStateSnapshot(
            session_id=session.id,
            workspace_id=workspace.id if workspace else session.workspace_id,
            run_id=run.id if run else run_id,
            session_status=session.status,
            run_status=run.status if run else None,
            workspace_run_id=workspace.current_run_id if workspace else None,
            session_run_id=session.current_run_id,
            checkpoint_ref=checkpoint_ref,
            has_checkpoint=bool(checkpoint_ref and checkpoint_ref.checkpoint_id),
            current_step_id=run.current_step_id if run else None,
            last_event_at=run.last_event_at if run else None,
            has_continuable_cancelled_plan=await self._has_continuable_cancelled_plan(
                uow=uow,
                session_id=session_id,
            ),
            source=source,
        )

    async def _has_continuable_cancelled_plan(
            self,
            *,
            uow: IUnitOfWork,
            session_id: str,
    ) -> bool:
        records = await uow.workflow_run.list_event_records_by_session(session_id)
        for record in reversed(records):
            event = record.event_payload
            if not isinstance(event, PlanEvent):
                continue
            if event.plan.status != ExecutionStatus.CANCELLED:
                return False
            return any(step.status == ExecutionStatus.CANCELLED for step in event.plan.steps)
        return False

    def _command_from_event(self, event: BaseEvent) -> RuntimeCommand | None:
        if isinstance(event, WaitEvent):
            return RuntimeCommand.WAIT
        if isinstance(event, DoneEvent):
            return RuntimeCommand.COMPLETE
        if isinstance(event, ErrorEvent):
            return RuntimeCommand.FAIL
        if isinstance(event, PlanEvent) and event.plan.status == ExecutionStatus.CANCELLED:
            return RuntimeCommand.CANCEL
        return None

    @staticmethod
    def _resolve_graph_projection_status(state: dict[str, Any]) -> WorkflowRunStatus | None:
        raw_status = str(get_graph_projection(state.get("graph_metadata")).get("run_status") or "").strip()
        if not raw_status:
            return None
        try:
            return WorkflowRunStatus(raw_status)
        except ValueError:
            logger.warning(
                "graph projection状态无效",
                extra={"graph_projection_status": raw_status},
            )
            return None

    @staticmethod
    def _resolve_graph_reconciled_run_status(
            *,
            snapshot: RuntimeStateSnapshot,
            pending_interrupt: dict[str, Any],
            graph_projection_status: WorkflowRunStatus | None,
    ) -> WorkflowRunStatus | None:
        if snapshot.run_status is None:
            return None
        if is_terminal(snapshot.run_status):
            return snapshot.run_status
        if is_terminal(snapshot.session_status):
            return WorkflowRunStatus(snapshot.session_status.value)
        if graph_projection_status is not None and is_terminal(graph_projection_status):
            return graph_projection_status
        if pending_interrupt:
            return WorkflowRunStatus.WAITING
        if graph_projection_status in {
            WorkflowRunStatus.WAITING,
            WorkflowRunStatus.RUNNING,
        }:
            return graph_projection_status
        if snapshot.run_status == WorkflowRunStatus.WAITING:
            return WorkflowRunStatus.WAITING
        return None

    async def _sync_step_projection(
            self,
            *,
            uow: IUnitOfWork,
            run_id: str,
            event: BaseEvent,
    ) -> None:
        if isinstance(event, PlanEvent):
            await uow.workflow_run.replace_steps_from_plan(run_id=run_id, plan=event.plan)
        elif isinstance(event, StepEvent):
            await uow.workflow_run.upsert_step_from_event(run_id=run_id, event=event)

    async def _apply_transition(
            self,
            *,
            uow: IUnitOfWork,
            snapshot: RuntimeStateSnapshot,
            session_id: str,
            run_id: str,
            event: BaseEvent | None,
            transition,
            projection: RuntimeEventProjection,
            event_inserted: bool,
    ) -> None:
        event_created_at = event.created_at if event is not None else None
        finished_at = event_created_at if transition.to_run_status and is_terminal(transition.to_run_status) else None
        current_step_id = None if transition.to_run_status and is_terminal(
            transition.to_run_status) else snapshot.current_step_id
        await uow.workflow_run.update_status(
            run_id,
            status=transition.to_run_status,
            finished_at=finished_at,
            last_event_at=event_created_at,
            current_step_id=current_step_id,
        )
        await self._update_session_projection(
            uow=uow,
            session_id=session_id,
            status=transition.to_session_status,
            projection=projection,
            event_inserted=event_inserted,
        )

    async def _update_session_projection(
            self,
            *,
            uow: IUnitOfWork,
            session_id: str,
            status: SessionStatus,
            projection: RuntimeEventProjection,
            event_inserted: bool,
    ) -> None:
        await uow.session.update_runtime_state(
            session_id,
            status=status,
            title=projection.title,
            latest_message=projection.latest_message,
            latest_message_at=projection.latest_message_at,
            increment_unread=projection.increment_unread and event_inserted,
        )

    def _log_transition(self, *, snapshot: RuntimeStateSnapshot, transition) -> None:
        logger.info(
            "运行状态转移",
            extra={
                "session_id": snapshot.session_id,
                "workspace_id": snapshot.workspace_id,
                "run_id": snapshot.run_id,
                "trigger": transition.command.value,
                "from_session_status": transition.from_session_status.value,
                "to_session_status": transition.to_session_status.value,
                "from_run_status": transition.from_run_status.value if transition.from_run_status else None,
                "to_run_status": transition.to_run_status.value if transition.to_run_status else None,
                "checkpoint_namespace": snapshot.checkpoint_ref.namespace if snapshot.checkpoint_ref else None,
                "checkpoint_id": snapshot.checkpoint_ref.checkpoint_id if snapshot.checkpoint_ref else None,
                "has_pending_interrupt": bool(snapshot.pending_interrupt),
                "graph_projection_status": (
                    snapshot.graph_projection_status.value if snapshot.graph_projection_status else None
                ),
            },
        )
