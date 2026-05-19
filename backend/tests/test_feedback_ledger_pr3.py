import asyncio
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.application.service.feedback_ledger_service import FeedbackRequiredRecordError
from app.application.service.user_feedback_ingress_service import UserFeedbackIngressService
from app.domain.models import FeedbackInputEvent, User
from app.domain.models.feedback import (
    FeedbackReasonCode,
    FeedbackScopeKind,
    FeedbackSourceKind,
    FeedbackTargetRefResult,
    UserFeedbackIntent,
    UserFeedbackIntentKind,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.interfaces.dependencies.auth import get_current_user
from app.interfaces.dependencies.services import (
    get_agent_service,
    get_runtime_observation_service,
    get_runtime_access_control_service,
    get_session_service,
    get_session_stream_facade,
    get_user_feedback_ingress_service,
)
from app.interfaces.endpoints.session_routes import router as session_router
from app.interfaces.errors.exception_handlers import register_exception_handlers
from app.interfaces.schemas.session import ChatRequest


def _access_scope() -> AccessScopeResult:
    return AccessScopeResult(
        tenant_id="user-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
    )


class _WorkflowRunRepo:
    def __init__(self) -> None:
        self.records = []

    async def get_event_record_by_type_and_hash(self, *, input_hash: str, **kwargs):
        for record in reversed(self.records):
            payload = getattr(record.event_payload, "payload", None)
            if (
                    record.user_id == kwargs["user_id"]
                    and record.session_id == kwargs["session_id"]
                    and record.run_id == kwargs["run_id"]
                    and record.event_type == kwargs["event_type"]
                    and getattr(payload, "input_hash", None) == input_hash
            ):
                return record
        return None

    async def add_event_record_if_absent(self, session_id: str, run_id: str, event) -> bool:
        if any(record.run_id == run_id and record.event_id == event.id for record in self.records):
            return False
        self.records.append(
            SimpleNamespace(
                user_id="user-1",
                session_id=session_id,
                run_id=run_id,
                event_id=event.id,
                event_type=event.type,
                event_payload=event,
            )
        )
        return True

    async def get_event_record_by_event_id(self, *, event_id: str, **kwargs):
        for record in self.records:
            if (
                    record.user_id == kwargs["user_id"]
                    and record.session_id == kwargs["session_id"]
                    and record.run_id == kwargs["run_id"]
                    and record.event_id == event_id
            ):
                return record
        return None


class _UoW:
    def __init__(self, workflow_run_repo: _WorkflowRunRepo) -> None:
        self.workflow_run = workflow_run_repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FeedbackLedgerService:
    def __init__(self) -> None:
        self.commands = []

    async def record_user_feedback(self, command):
        self.commands.append(command)
        return SimpleNamespace(success=True, model_dump=lambda mode="json": {"success": True})


def test_submit_explicit_feedback_should_create_feedback_input_before_feedback_record_and_reuse_on_retry() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo()
        feedback_service = _FeedbackLedgerService()
        service = UserFeedbackIngressService(
            uow_factory=lambda: _UoW(workflow_repo),
            feedback_service=feedback_service,
        )
        target_ref = FeedbackTargetRefResult(
            target_type="message_event",
            target_id="evt-final-1",
            target_run_id="run-1",
        )
        request = SimpleNamespace(
            source_action="final_satisfaction",
            intent_kind=UserFeedbackIntentKind.DISSATISFACTION,
            target_ref=target_ref,
            reason_code=FeedbackReasonCode.USER_REPORTED_DISSATISFACTION,
            summary_hint="用户标记最终回答不满意",
            client_request_id="client-request-1",
        )

        first_result = await service.submit_explicit_feedback(access_scope=_access_scope(), request=request)
        second_result = await service.submit_explicit_feedback(access_scope=_access_scope(), request=request)

        assert first_result.success is True
        assert second_result.success is True
        assert len(workflow_repo.records) == 1
        assert isinstance(workflow_repo.records[0].event_payload, FeedbackInputEvent)
        assert len(feedback_service.commands) == 2
        assert feedback_service.commands[0].source_ref.source_event_id == workflow_repo.records[0].event_id
        assert feedback_service.commands[1].source_ref.source_event_id == workflow_repo.records[0].event_id
        assert feedback_service.commands[0].source_ref.source_kind == FeedbackSourceKind.FEEDBACK_INPUT
        assert feedback_service.commands[0].requested_feedback_scope_kind == FeedbackScopeKind.SESSION
        assert workflow_repo.records[0].event_id.startswith("feedback_input_")

    asyncio.run(_run())


def test_submit_explicit_feedback_should_reuse_feedback_input_when_client_request_id_changes() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo()
        feedback_service = _FeedbackLedgerService()
        service = UserFeedbackIngressService(
            uow_factory=lambda: _UoW(workflow_repo),
            feedback_service=feedback_service,
        )
        target_ref = FeedbackTargetRefResult(
            target_type="message_event",
            target_id="evt-final-1",
            target_run_id="run-1",
        )
        base_request = SimpleNamespace(
            source_action="final_satisfaction",
            intent_kind=UserFeedbackIntentKind.DISSATISFACTION,
            target_ref=target_ref,
            reason_code=FeedbackReasonCode.USER_REPORTED_DISSATISFACTION,
            summary_hint="同一反馈",
            client_request_id="client-request-1",
        )
        retry_request = SimpleNamespace(
            source_action="final_satisfaction",
            intent_kind=UserFeedbackIntentKind.DISSATISFACTION,
            target_ref=target_ref,
            reason_code=FeedbackReasonCode.USER_REPORTED_DISSATISFACTION,
            summary_hint="同一反馈",
            client_request_id="client-request-2",
        )

        await service.submit_explicit_feedback(access_scope=_access_scope(), request=base_request)
        await service.submit_explicit_feedback(access_scope=_access_scope(), request=retry_request)

        assert len(workflow_repo.records) == 1
        assert len({command.source_ref.source_event_id for command in feedback_service.commands}) == 1

    asyncio.run(_run())


def test_submit_explicit_feedback_should_not_persist_raw_client_request_id() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo()
        service = UserFeedbackIngressService(
            uow_factory=lambda: _UoW(workflow_repo),
            feedback_service=_FeedbackLedgerService(),
        )
        target_ref = FeedbackTargetRefResult(
            target_type="message_event",
            target_id="evt-final-1",
            target_run_id="run-1",
        )
        request = SimpleNamespace(
            source_action="final_satisfaction",
            intent_kind=UserFeedbackIntentKind.DISSATISFACTION,
            target_ref=target_ref,
            reason_code=FeedbackReasonCode.USER_REPORTED_DISSATISFACTION,
            summary_hint="不满意",
            client_request_id="tester@example.com token=abcdefghi123456",
        )

        await service.submit_explicit_feedback(access_scope=_access_scope(), request=request)

        payload_json = workflow_repo.records[0].event_payload.model_dump(mode="json")
        payload_text = str(payload_json)
        assert "client_request_id" not in payload_text
        assert "tester@example.com" not in payload_text
        assert "abcdefghi123456" not in payload_text

    asyncio.run(_run())


def test_submit_explicit_feedback_should_sanitize_feedback_input_summary() -> None:
    async def _run() -> None:
        workflow_repo = _WorkflowRunRepo()
        service = UserFeedbackIngressService(
            uow_factory=lambda: _UoW(workflow_repo),
            feedback_service=_FeedbackLedgerService(),
        )
        target_ref = FeedbackTargetRefResult(
            target_type="message_event",
            target_id="evt-final-1",
            target_run_id="run-1",
        )
        request = SimpleNamespace(
            source_action="final_satisfaction",
            intent_kind=UserFeedbackIntentKind.DISSATISFACTION,
            target_ref=target_ref,
            reason_code=FeedbackReasonCode.USER_REPORTED_DISSATISFACTION,
            summary_hint="不满意，联系我 tester@example.com，token=abcdefghi123456",
            client_request_id="client-request-2",
        )

        await service.submit_explicit_feedback(access_scope=_access_scope(), request=request)

        payload = workflow_repo.records[0].event_payload.payload
        assert payload.sanitized_summary is not None
        assert "tester@example.com" not in payload.sanitized_summary
        assert "abcdefghi123456" not in payload.sanitized_summary
        assert "[REDACTED]" in payload.sanitized_summary
        command_summary = service._feedback_service.commands[0].source_ref.source_summary
        assert "tester@example.com" not in command_summary
        assert "abcdefghi123456" not in command_summary
        assert "[REDACTED]" in command_summary

    asyncio.run(_run())


def test_message_feedback_intent_should_reject_controlled_intents() -> None:
    for intent_kind, reason_code in (
            ("cancel", "user_cancelled"),
            ("continue_cancelled", "user_continued_cancelled"),
            ("confirmation", "user_confirmed"),
            ("selection", "user_selected_option"),
            ("takeover", "user_provided_clarification"),
    ):
        try:
            ChatRequest.model_validate(
                {
                    "message": "伪造受控反馈",
                    "feedback_intent": {
                        "intent_kind": intent_kind,
                        "target_ref": {
                            "target_type": "run",
                            "target_id": "run-1",
                            "target_run_id": "run-1",
                        },
                        "reason_code": reason_code,
                    },
                }
            )
        except ValueError:
            continue
        raise AssertionError(f"expected {intent_kind} to be rejected")


def test_chat_route_should_reject_feedback_intent_extra_fields_before_ingress() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(session_router, prefix="/api")

    class _SessionService:
        def __init__(self) -> None:
            self.calls = []

        async def get_session(self, **kwargs):
            self.calls.append(kwargs)
            return None

    class _StreamFacade:
        def __init__(self) -> None:
            self.calls = []

        async def stream_chat(self, **kwargs):
            self.calls.append(kwargs)
            raise AssertionError("invalid feedback_intent should fail before stream facade")

    session_service = _SessionService()
    stream_facade = _StreamFacade()

    app.dependency_overrides[get_current_user] = lambda: User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
    )
    app.dependency_overrides[get_session_service] = lambda: session_service
    app.dependency_overrides[get_agent_service] = lambda: SimpleNamespace()
    app.dependency_overrides[get_session_stream_facade] = lambda: stream_facade
    app.dependency_overrides[get_runtime_observation_service] = lambda: SimpleNamespace()

    with TestClient(app) as client:
        response = client.post(
            "/api/sessions/session-1/chat",
            json={
                "message": "请修正上一次回答",
                "feedback_intent": {
                    "intent_kind": "correction",
                    "target_ref": {
                        "target_type": "message_event",
                        "target_id": "evt-final-1",
                        "target_run_id": "run-1",
                    },
                    "reason_code": "user_corrected_requirement",
                    "raw_request_body": "should not be accepted",
                },
            },
        )

    assert response.status_code == 422
    assert session_service.calls == []
    assert stream_facade.calls == []


def test_message_feedback_command_should_sanitize_source_summary() -> None:
    service = UserFeedbackIngressService(
        uow_factory=lambda: _UoW(_WorkflowRunRepo()),
        feedback_service=_FeedbackLedgerService(),
    )
    command = service._build_user_feedback_command(
        access_scope=_access_scope(),
        source_kind=FeedbackSourceKind.MESSAGE_EVENT,
        source_event_id="evt-message-1",
        source_run_id="run-1",
        source_summary="请联系 tester@example.com，token=abcdefghi123456，手机号 138-0000-0000",
        source_record_refs=[{"event_id": "evt-message-1"}],
        intent=UserFeedbackIntent(
            intent_kind=UserFeedbackIntentKind.CORRECTION,
            target_ref=FeedbackTargetRefResult(
                target_type="message_event",
                target_id="evt-final-1",
                target_run_id="run-1",
            ),
            reason_code=FeedbackReasonCode.USER_CORRECTED_REQUIREMENT,
            summary_hint=None,
        ),
    )

    source_summary = command.source_ref.source_summary
    assert "tester@example.com" not in source_summary
    assert "abcdefghi123456" not in source_summary
    assert "138-0000-0000" not in source_summary
    assert command.feedback_summary.summary_text == source_summary
    assert command.prompt_safe_summary.summary_text == source_summary


def test_submit_feedback_route_should_reject_missing_target_before_ingress_write() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(session_router, prefix="/api")

    class _Ingress:
        def __init__(self) -> None:
            self.calls = []

        async def submit_explicit_feedback(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(model_dump=lambda mode="json": {"success": True})

    ingress = _Ingress()

    app.dependency_overrides[get_current_user] = lambda: User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
    )
    app.dependency_overrides[get_user_feedback_ingress_service] = lambda: ingress
    app.dependency_overrides[get_runtime_access_control_service] = lambda: SimpleNamespace(
        resolve_session_scope=lambda **kwargs: _access_scope()
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/sessions/session-1/feedback",
            json={
                "source_action": "final_satisfaction",
                "intent_kind": "dissatisfaction",
                "reason_code": "user_reported_dissatisfaction",
            },
        )

    assert response.status_code == 422
    assert ingress.calls == []


def test_submit_feedback_route_should_reject_extra_fields_before_ingress_write() -> None:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(session_router, prefix="/api")

    class _Ingress:
        def __init__(self) -> None:
            self.calls = []

        async def submit_explicit_feedback(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(model_dump=lambda mode="json": {"success": True})

    ingress = _Ingress()

    app.dependency_overrides[get_current_user] = lambda: User(
        id="user-1",
        email="tester@example.com",
        password="hashed-password",
    )
    app.dependency_overrides[get_user_feedback_ingress_service] = lambda: ingress
    app.dependency_overrides[get_runtime_access_control_service] = lambda: SimpleNamespace(
        resolve_session_scope=lambda **kwargs: _access_scope()
    )

    with TestClient(app) as client:
        response = client.post(
            "/api/sessions/session-1/feedback",
            json={
                "source_action": "final_satisfaction",
                "intent_kind": "dissatisfaction",
                "target_ref": {
                    "target_type": "message_event",
                    "target_id": "evt-final-1",
                    "target_run_id": "run-1",
                },
                "reason_code": "user_reported_dissatisfaction",
                "raw_request_body": "should not be accepted",
            },
        )

    assert response.status_code == 422
    assert ingress.calls == []


def test_feedback_input_source_action_should_reject_uncontrolled_action() -> None:
    try:
        UserFeedbackIngressService.validate_feedback_input_source_action("free_form_feedback")
    except FeedbackRequiredRecordError:
        return
    raise AssertionError("expected FeedbackRequiredRecordError")
