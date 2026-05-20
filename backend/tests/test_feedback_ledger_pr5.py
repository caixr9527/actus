from __future__ import annotations

import asyncio
from datetime import datetime, timedelta

from app.application.service.feedback_event_projector import FeedbackEventProjector
from app.application.service.feedback_ledger_service import FeedbackLedgerService
from app.domain.models import WorkflowRunEventRecord
from app.domain.models.feedback import (
    FeedbackCategory,
    FeedbackReasonCode,
    FeedbackResolutionCommand,
    FeedbackResolutionReasonCode,
    FeedbackResolutionResult,
    FeedbackScopeKind,
    FeedbackSeverity,
    FeedbackStatus,
    FeedbackTargetType,
    FeedbackWriteResult,
)

from tests.test_feedback_ledger_pr2 import (
    _FeedbackRepo,
    _UoW,
    _WorkflowRunRepo,
    _access_scope,
    _event_record,
    _record,
    _runtime_command,
    _target_ref,
)


class _ProjectionFeedbackRepo(_FeedbackRepo):
    async def list_by_source_event_for_projection(
            self,
            *,
            user_id: str,
            session_id: str,
            source_run_id: str,
            source_event_id: str,
            limit: int = 100,
    ):
        return [
            record for record in self.records
            if record.user_id == user_id
            and record.session_id == session_id
            and record.source_run_id == source_run_id
            and record.source_event_id == source_event_id
        ][:limit]

    async def list_by_resolution_aggregation_key(
            self,
            *,
            user_id: str,
            session_id: str,
            source_run_id: str,
            resolution_aggregation_key: str,
            limit: int = 100,
    ):
        return [
            record for record in self.records
            if record.user_id == user_id
            and record.session_id == session_id
            and record.source_run_id == source_run_id
            and record.status != FeedbackStatus.OPEN
            and (record.resolution.resolved_by_ref or {}).get("resolution_aggregation_key") == resolution_aggregation_key
        ][:limit]


class _ProjectionWorkflowRunRepo(_WorkflowRunRepo):
    def __init__(self, records=None) -> None:
        super().__init__(records=records)
        self.feedback_events: dict[tuple[str, str], WorkflowRunEventRecord] = {}

    async def upsert_feedback_event_record(self, *, session_id: str, run_id: str, event):
        if event.type != "feedback":
            raise ValueError("only feedback")
        existing = self.feedback_events.get((run_id, event.id))
        if existing is not None:
            event = event.model_copy(update={"created_at": existing.created_at})
        record = WorkflowRunEventRecord(
            run_id=run_id,
            session_id=session_id,
            user_id="user-1",
            event_id=event.id,
            event_type=event.type,
            event_payload=event,
            created_at=event.created_at,
        )
        self.feedback_events[(run_id, event.id)] = record
        return record


class _RaisingProjector:
    async def project_record_written(self, record_ref: FeedbackWriteResult) -> None:
        raise RuntimeError("projection failed")

    async def project_resolution_updated(self, record_ref: FeedbackWriteResult) -> None:
        raise RuntimeError("projection failed")


def _service(uow: _UoW) -> FeedbackLedgerService:
    return FeedbackLedgerService(
        uow_factory=lambda: uow,
        feedback_event_projector=FeedbackEventProjector(uow_factory=lambda: uow),
    )


def test_record_success_should_upsert_one_feedback_event_per_source_event() -> None:
    async def _run() -> None:
        workflow_repo = _ProjectionWorkflowRunRepo(
            {"evt-tool-1": _event_record(event_id="evt-tool-1", event_type="tool")}
        )
        feedback_repo = _ProjectionFeedbackRepo()
        uow = _UoW(feedback_repo=feedback_repo, workflow_run_repo=workflow_repo)
        service = _service(uow)

        first = await service.record_runtime_feedback(_runtime_command())
        second = await service.record_runtime_feedback(
            _runtime_command(
                category=FeedbackCategory.PATH_ERROR,
                reason_code=FeedbackReasonCode.PATH_ERROR,
                target_ref=_target_ref(target_type=FeedbackTargetType.TOOL_CALL, target_id="call-2"),
            )
        )

        event_id = "feedback:run-1:evt-tool-1"
        projected = workflow_repo.feedback_events[("run-1", event_id)]
        payload = projected.event_payload.payload

        assert first.success is True
        assert second.success is True
        assert list(workflow_repo.feedback_events) == [("run-1", event_id)]
        assert payload.feedback_refs == [first.feedback_id, second.feedback_id]
        assert payload.counts == {"feedback_count": 2}
        assert payload.kind_counts == {"runtime_feedback": 2}
        assert payload.source_event_ids == ["evt-tool-1"]
        assert payload.runtime_metadata["aggregation_kind"] == "source_event"

    asyncio.run(_run())


def test_feedback_event_replace_should_preserve_row_and_payload_created_at() -> None:
    async def _run() -> None:
        workflow_repo = _ProjectionWorkflowRunRepo(
            {"evt-tool-1": _event_record(event_id="evt-tool-1", event_type="tool")}
        )
        feedback_repo = _ProjectionFeedbackRepo()
        uow = _UoW(feedback_repo=feedback_repo, workflow_run_repo=workflow_repo)
        service = _service(uow)

        await service.record_runtime_feedback(_runtime_command())
        event_id = "feedback:run-1:evt-tool-1"
        first_projected = workflow_repo.feedback_events[("run-1", event_id)]
        first_created_at = first_projected.created_at
        first_payload_created_at = first_projected.event_payload.created_at

        await service.record_runtime_feedback(
            _runtime_command(
                category=FeedbackCategory.PATH_ERROR,
                reason_code=FeedbackReasonCode.PATH_ERROR,
                target_ref=_target_ref(target_type=FeedbackTargetType.TOOL_CALL, target_id="call-2"),
            )
        )

        replaced = workflow_repo.feedback_events[("run-1", event_id)]
        assert replaced.created_at == first_created_at
        assert replaced.event_payload.created_at == first_payload_created_at
        assert replaced.event_payload.payload.feedback_refs == [
            feedback_repo.records[0].id,
            feedback_repo.records[1].id,
        ]

    asyncio.run(_run())


def test_resolve_feedback_should_project_by_resolution_batch_and_normalize_ref() -> None:
    async def _run() -> None:
        feedback_repo = _ProjectionFeedbackRepo(
            [
                _record(feedback_id="fb-1", severity=FeedbackSeverity.ERROR),
                _record(feedback_id="fb-2", severity=FeedbackSeverity.WARNING, target_id="evt-target-2"),
            ]
        )
        workflow_repo = _ProjectionWorkflowRunRepo()
        uow = _UoW(feedback_repo=feedback_repo, workflow_run_repo=workflow_repo)
        service = _service(uow)

        for feedback_id in ("fb-1", "fb-2"):
            await service.resolve_feedback(
                FeedbackResolutionCommand(
                    access_scope=_access_scope(),
                    feedback_id=feedback_id,
                    requested_feedback_scope_kind=FeedbackScopeKind.RUN,
                    requested_scope_id="run-1",
                    resolution=FeedbackResolutionResult(
                        status=FeedbackStatus.RESOLVED,
                        resolution_reason_code=FeedbackResolutionReasonCode.RESOLVED_BY_REPLAN,
                        resolved_by_ref={"resolver": "replan"},
                        resolved_at=datetime(2026, 5, 20, 10, 0, 0),
                        resolution_summary="后续步骤已处理。",
                    ),
                    updated_at=datetime(2026, 5, 20, 10, 0, 0),
                    resolution_batch_id="batch-1",
                )
            )

        event_id = "feedback:run-1:batch-1"
        projected = workflow_repo.feedback_events[("run-1", event_id)]
        payload = projected.event_payload.payload

        assert payload.feedback_refs == ["fb-1", "fb-2"]
        assert payload.status_counts == {"resolved": 2}
        assert payload.runtime_metadata["aggregation_kind"] == "resolution"
        assert feedback_repo.updated[0]["resolution"].resolved_by_ref["resolution_batch_id"] == "batch-1"
        assert feedback_repo.updated[0]["resolution"].resolved_by_ref["resolution_aggregation_key"] == "batch-1"
        assert feedback_repo.updated[0]["resolution"].resolved_by_ref["resolution_aggregation_kind"] == "batch"

    asyncio.run(_run())


def test_resolution_source_event_should_not_override_record_source_run_projection() -> None:
    async def _run() -> None:
        feedback_repo = _ProjectionFeedbackRepo(
            [
                _record(
                    feedback_id="fb-old",
                    scope_kind=FeedbackScopeKind.SESSION,
                    scope_id="session-1",
                    run_id="old-run",
                    source_run_id="old-run",
                    target_run_id="old-run",
                    source_event_id="evt-original",
                )
            ]
        )
        workflow_repo = _ProjectionWorkflowRunRepo(
            {"evt-resolve-new": _event_record(event_id="evt-resolve-new", run_id="new-run", event_type="message")}
        )
        uow = _UoW(feedback_repo=feedback_repo, workflow_run_repo=workflow_repo)
        service = _service(uow)

        await service.resolve_feedback(
            FeedbackResolutionCommand(
                access_scope=_access_scope(run_id="new-run"),
                feedback_id="fb-old",
                requested_feedback_scope_kind=FeedbackScopeKind.SESSION,
                requested_scope_id="session-1",
                resolution=FeedbackResolutionResult(
                    status=FeedbackStatus.RESOLVED,
                    resolution_reason_code=FeedbackResolutionReasonCode.RESOLVED_BY_USER_CONFIRMATION,
                    resolved_by_ref={"resolver": "user"},
                    resolved_at=datetime(2026, 5, 20, 10, 0, 0),
                    resolution_summary="用户确认已处理。",
                ),
                updated_at=datetime(2026, 5, 20, 10, 0, 0),
                resolution_source_event_id="evt-resolve-new",
            )
        )

        event_id = "feedback:old-run:evt-resolve-new"
        assert ("old-run", event_id) in workflow_repo.feedback_events
        assert ("new-run", "feedback:new-run:evt-resolve-new") not in workflow_repo.feedback_events
        assert workflow_repo.feedback_events[("old-run", event_id)].event_payload.payload.feedback_refs == ["fb-old"]

    asyncio.run(_run())


def test_resolution_batch_should_split_projection_by_record_source_run() -> None:
    async def _run() -> None:
        feedback_repo = _ProjectionFeedbackRepo(
            [
                _record(
                    feedback_id="fb-run-1",
                    scope_kind=FeedbackScopeKind.SESSION,
                    scope_id="session-1",
                    run_id="run-1",
                    source_run_id="run-1",
                    target_run_id="run-1",
                    source_event_id="evt-original-1",
                ),
                _record(
                    feedback_id="fb-run-2",
                    scope_kind=FeedbackScopeKind.SESSION,
                    scope_id="session-1",
                    run_id="run-2",
                    source_run_id="run-2",
                    target_run_id="run-2",
                    source_event_id="evt-original-2",
                    target_id="evt-target-2",
                    created_at=datetime(2026, 5, 20, 9, 0, 0) + timedelta(seconds=1),
                ),
            ]
        )
        workflow_repo = _ProjectionWorkflowRunRepo()
        uow = _UoW(feedback_repo=feedback_repo, workflow_run_repo=workflow_repo)
        service = _service(uow)

        for run_id, feedback_id in (("run-1", "fb-run-1"), ("run-2", "fb-run-2")):
            await service.resolve_feedback(
                FeedbackResolutionCommand(
                    access_scope=_access_scope(run_id=run_id),
                    feedback_id=feedback_id,
                    requested_feedback_scope_kind=FeedbackScopeKind.SESSION,
                    requested_scope_id="session-1",
                    resolution=FeedbackResolutionResult(
                        status=FeedbackStatus.RESOLVED,
                        resolution_reason_code=FeedbackResolutionReasonCode.RESOLVED_BY_REPLAN,
                        resolved_by_ref={"resolver": "batch"},
                        resolved_at=datetime(2026, 5, 20, 10, 0, 0),
                        resolution_summary="批量处理。",
                    ),
                    updated_at=datetime(2026, 5, 20, 10, 0, 0),
                    resolution_batch_id="batch-cross-run",
                )
            )

        assert workflow_repo.feedback_events[("run-1", "feedback:run-1:batch-cross-run")].event_payload.payload.feedback_refs == ["fb-run-1"]
        assert workflow_repo.feedback_events[("run-2", "feedback:run-2:batch-cross-run")].event_payload.payload.feedback_refs == ["fb-run-2"]

    asyncio.run(_run())


def test_projection_failure_should_not_rollback_feedback_record() -> None:
    async def _run() -> None:
        workflow_repo = _ProjectionWorkflowRunRepo(
            {"evt-tool-1": _event_record(event_id="evt-tool-1", event_type="tool")}
        )
        feedback_repo = _ProjectionFeedbackRepo()
        service = FeedbackLedgerService(
            uow_factory=lambda: _UoW(feedback_repo=feedback_repo, workflow_run_repo=workflow_repo),
            feedback_event_projector=_RaisingProjector(),
        )

        result = await service.record_runtime_feedback(_runtime_command())

        assert result.success is True
        assert len(feedback_repo.records) == 1

    asyncio.run(_run())
