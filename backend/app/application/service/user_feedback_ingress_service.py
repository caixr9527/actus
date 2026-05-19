#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PR3 用户反馈主链路接入服务。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable

from app.application.service.feedback_ledger_service import (
    FeedbackLedgerService,
    FeedbackRequiredRecordError,
)
from app.application.service.feedback_ledger_common import FeedbackSanitizer
from app.domain.models import (
    FeedbackInputEvent,
    MessageEvent,
    WorkflowRunEventRecord,
    normalize_wait_payload,
    resolve_wait_resume_message,
)
from app.domain.models.feedback import (
    FeedbackCategory,
    FeedbackClassificationResult,
    FeedbackDataOrigin,
    FeedbackInputEventPayloadResult,
    FeedbackPromptSafeSummaryResult,
    FeedbackReasonCode,
    FeedbackScopeKind,
    FeedbackSourceConfidence,
    FeedbackSourceKind,
    FeedbackSourceRefResult,
    FeedbackSummaryKind,
    FeedbackSummaryResult,
    FeedbackTargetRefResult,
    FeedbackWriteResult,
    UserFeedbackCaptureResult,
    UserFeedbackCommand,
    UserFeedbackIntent,
    UserFeedbackIntentKind,
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)
from app.interfaces.schemas.session import ChatRequest, SubmitFeedbackRequest


_HTTP_SOURCE_ACTIONS = {
    "final_satisfaction",
    "artifact_satisfaction",
    "explicit_correction",
    "explicit_preference",
}

_CONTROL_SOURCE_ACTIONS = {
    "cancel",
    "continue_cancelled",
}


@dataclass(frozen=True)
class FeedbackInputSourceEventResult:
    event: FeedbackInputEvent
    event_record: WorkflowRunEventRecord
    reused: bool


class UserFeedbackCapturePolicy:
    """将受控用户入口归一成结构化 feedback intent。"""

    _MESSAGE_CATEGORY_BY_INTENT = {
        UserFeedbackIntentKind.CORRECTION: FeedbackCategory.CORRECTION,
        UserFeedbackIntentKind.PREFERENCE: FeedbackCategory.PREFERENCE,
        UserFeedbackIntentKind.CLARIFICATION: FeedbackCategory.CLARIFICATION,
        UserFeedbackIntentKind.SATISFACTION: FeedbackCategory.SATISFACTION,
        UserFeedbackIntentKind.DISSATISFACTION: FeedbackCategory.DISSATISFACTION,
        UserFeedbackIntentKind.CONFIRMATION: FeedbackCategory.CONFIRMATION,
        UserFeedbackIntentKind.SELECTION: FeedbackCategory.SELECTION,
        UserFeedbackIntentKind.CANCEL: FeedbackCategory.CANCEL,
        UserFeedbackIntentKind.CONTINUE_CANCELLED: FeedbackCategory.CONTINUE_CANCELLED,
        UserFeedbackIntentKind.TAKEOVER: FeedbackCategory.TAKEOVER,
    }

    def capture_from_chat_request(self, request: ChatRequest) -> UserFeedbackCaptureResult:
        if request.feedback_intent is None:
            return UserFeedbackCaptureResult(captured=False)
        return UserFeedbackCaptureResult(
            captured=True,
            intent=UserFeedbackIntent(
                intent_kind=request.feedback_intent.intent_kind,
                target_ref=request.feedback_intent.target_ref,
                reason_code=request.feedback_intent.reason_code,
                summary_hint=request.feedback_intent.summary_hint,
            ),
        )

    @classmethod
    def category_for_intent(cls, intent_kind: UserFeedbackIntentKind) -> FeedbackCategory:
        return cls._MESSAGE_CATEGORY_BY_INTENT[intent_kind]

    @classmethod
    def resolve_scope_kind(cls, intent_kind: UserFeedbackIntentKind) -> FeedbackScopeKind:
        if intent_kind == UserFeedbackIntentKind.CONTINUE_CANCELLED:
            return FeedbackScopeKind.SESSION
        if intent_kind in {
            UserFeedbackIntentKind.CORRECTION,
            UserFeedbackIntentKind.PREFERENCE,
            UserFeedbackIntentKind.CLARIFICATION,
            UserFeedbackIntentKind.SATISFACTION,
            UserFeedbackIntentKind.DISSATISFACTION,
        }:
            return FeedbackScopeKind.SESSION
        return FeedbackScopeKind.RUN


class UserFeedbackIngressService:
    """PR3 用户反馈主链路编排。"""

    def __init__(
            self,
            *,
            uow_factory: Callable[[], IUnitOfWork],
            feedback_service: FeedbackLedgerService,
            capture_policy: UserFeedbackCapturePolicy | None = None,
    ) -> None:
        self._uow_factory = uow_factory
        self._feedback_service = feedback_service
        self._capture_policy = capture_policy or UserFeedbackCapturePolicy()

    def capture_chat_feedback(self, request: ChatRequest) -> UserFeedbackCaptureResult:
        return self._capture_policy.capture_from_chat_request(request)

    def capture_feedback_payload(self, payload: dict[str, object] | None) -> UserFeedbackCaptureResult:
        if payload is None:
            return UserFeedbackCaptureResult(captured=False)
        request = ChatRequest.model_validate(
            {
                "message": "feedback_intent",
                "feedback_intent": payload,
            }
        )
        return self.capture_chat_feedback(request)

    async def record_feedback_from_message_event(
            self,
            *,
            access_scope: AccessScopeResult,
            message_event: MessageEvent,
            capture_result: UserFeedbackCaptureResult,
    ) -> FeedbackWriteResult:
        if not capture_result.captured or capture_result.intent is None:
            raise ValueError("capture_result 必须包含结构化 feedback intent")
        source_summary = capture_result.intent.summary_hint or message_event.message
        return await self._feedback_service.record_user_feedback(
            self._build_user_feedback_command(
                access_scope=access_scope,
                source_kind=FeedbackSourceKind.MESSAGE_EVENT,
                source_event_id=message_event.id,
                source_run_id=str(access_scope.run_id) if access_scope.run_id else None,
                source_summary=source_summary,
                source_record_refs=[{"event_id": message_event.id}],
                intent=capture_result.intent,
            )
        )

    async def submit_explicit_feedback(
            self,
            *,
            access_scope: AccessScopeResult,
            request: SubmitFeedbackRequest,
    ) -> FeedbackWriteResult:
        self.validate_submit_request(request)
        source_event = await self.get_or_create_feedback_input_source_event(
            access_scope=access_scope,
            source_action=request.source_action,
            intent_kind=request.intent_kind,
            target_ref=request.target_ref,
            reason_code=request.reason_code,
            summary_hint=request.summary_hint,
        )
        intent = UserFeedbackIntent(
            intent_kind=request.intent_kind,
            target_ref=request.target_ref,
            reason_code=request.reason_code,
            summary_hint=request.summary_hint,
        )
        return await self._feedback_service.record_user_feedback(
            self._build_user_feedback_command(
                access_scope=access_scope,
                source_kind=FeedbackSourceKind.FEEDBACK_INPUT,
                source_event_id=source_event.event.id,
                source_run_id=str(access_scope.run_id) if access_scope.run_id else None,
                source_summary=request.summary_hint or request.reason_code.value,
                source_record_refs=[{"event_id": source_event.event.id}],
                intent=intent,
            )
        )

    async def record_cancel_feedback(
            self,
            *,
            access_scope: AccessScopeResult,
    ) -> FeedbackWriteResult:
        if not access_scope.run_id:
            raise FeedbackRequiredRecordError("cancel 用户反馈缺少 current run")
        target_ref = FeedbackTargetRefResult(
            target_type="run",
            target_id=str(access_scope.run_id),
            target_run_id=str(access_scope.run_id),
        )
        source_event = await self.get_or_create_feedback_input_source_event(
            access_scope=access_scope,
            source_action="cancel",
            intent_kind=UserFeedbackIntentKind.CANCEL,
            target_ref=target_ref,
            reason_code=FeedbackReasonCode.USER_CANCELLED,
            summary_hint=None,
        )
        return await self._feedback_service.record_user_feedback(
            self._build_user_feedback_command(
                access_scope=access_scope,
                source_kind=FeedbackSourceKind.FEEDBACK_INPUT,
                source_event_id=source_event.event.id,
                source_run_id=str(access_scope.run_id),
                source_summary="用户取消了当前运行",
                source_record_refs=[{"event_id": source_event.event.id}],
                intent=UserFeedbackIntent(
                    intent_kind=UserFeedbackIntentKind.CANCEL,
                    target_ref=target_ref,
                    reason_code=FeedbackReasonCode.USER_CANCELLED,
                    summary_hint=None,
                ),
            )
        )

    async def record_continue_cancelled_feedback(
            self,
            *,
            access_scope: AccessScopeResult,
            old_wait_event_id: str,
            old_cancelled_run_id: str,
    ) -> FeedbackWriteResult:
        if not access_scope.run_id:
            raise FeedbackRequiredRecordError("continue_cancelled 用户反馈缺少 current run")
        new_run_id = str(access_scope.run_id)
        old_wait_event_record = await self._load_owned_wait_event(
            access_scope=access_scope,
            wait_event_id=old_wait_event_id,
            run_id=old_cancelled_run_id,
        )
        target_ref = FeedbackTargetRefResult(
            target_type="wait_event",
            target_id=old_wait_event_record.event_id,
            target_run_id=old_wait_event_record.run_id,
        )
        source_event = await self.get_or_create_feedback_input_source_event(
            access_scope=access_scope,
            source_action="continue_cancelled",
            intent_kind=UserFeedbackIntentKind.CONTINUE_CANCELLED,
            target_ref=target_ref,
            reason_code=FeedbackReasonCode.USER_CONTINUED_CANCELLED,
            summary_hint=None,
        )
        return await self._feedback_service.record_user_feedback(
            self._build_user_feedback_command(
                access_scope=access_scope,
                source_kind=FeedbackSourceKind.FEEDBACK_INPUT,
                source_event_id=source_event.event.id,
                source_run_id=new_run_id,
                source_summary="用户继续执行已取消的任务",
                source_record_refs=[
                    {"event_id": source_event.event.id, "run_id": new_run_id},
                    {"event_id": old_wait_event_record.event_id, "run_id": old_wait_event_record.run_id},
                ],
                intent=UserFeedbackIntent(
                    intent_kind=UserFeedbackIntentKind.CONTINUE_CANCELLED,
                    target_ref=target_ref,
                    reason_code=FeedbackReasonCode.USER_CONTINUED_CANCELLED,
                    summary_hint=None,
                ),
            )
        )

    async def record_feedback_from_wait_resume(
            self,
            *,
            access_scope: AccessScopeResult,
            wait_event_id: str,
            wait_payload: dict[str, object] | None,
            resume_value: object,
    ) -> FeedbackWriteResult:
        if not access_scope.run_id:
            raise FeedbackRequiredRecordError("wait/resume 用户反馈缺少 current run")
        normalized_payload = normalize_wait_payload(wait_payload)
        kind = str(normalized_payload.get("kind") or "").strip()
        intent_kind, reason_code = self._resolve_wait_resume_feedback_semantics(
            payload_kind=kind,
            resume_value=resume_value,
            payload=normalized_payload,
        )
        summary = self._normalize_summary_text(
            resolve_wait_resume_message(normalized_payload, resume_value) or str(normalized_payload.get("prompt") or "")
        )
        target_ref = FeedbackTargetRefResult(
            target_type="wait_event",
            target_id=wait_event_id,
            target_run_id=str(access_scope.run_id),
        )
        return await self._feedback_service.record_user_feedback(
            self._build_user_feedback_command(
                access_scope=access_scope,
                source_kind=FeedbackSourceKind.WAIT_EVENT,
                source_event_id=wait_event_id,
                source_run_id=str(access_scope.run_id),
                source_summary=summary,
                source_record_refs=[{"event_id": wait_event_id}],
                intent=UserFeedbackIntent(
                    intent_kind=intent_kind,
                    target_ref=target_ref,
                    reason_code=reason_code,
                    summary_hint=summary,
                ),
            )
        )

    async def get_or_create_feedback_input_source_event(
            self,
            *,
            access_scope: AccessScopeResult,
            source_action: str,
            intent_kind: UserFeedbackIntentKind,
            target_ref: FeedbackTargetRefResult,
            reason_code: FeedbackReasonCode,
            summary_hint: str | None = None,
    ) -> FeedbackInputSourceEventResult:
        current_run_id = str(access_scope.run_id or "").strip()
        if not current_run_id:
            raise FeedbackRequiredRecordError("feedback_input source event 缺少 current run")
        self.validate_feedback_input_source_action(source_action)
        input_hash = self._build_feedback_input_hash(
            source_action=source_action,
            intent_kind=intent_kind,
            target_ref=target_ref,
            reason_code=reason_code,
            summary_hint=summary_hint,
        )
        payload = FeedbackInputEventPayloadResult(
            source_action=source_action,
            intent_kind=intent_kind,
            target_ref=target_ref,
            reason_code=reason_code,
            sanitized_summary=self._sanitize_feedback_input_summary(summary_hint),
            input_hash=input_hash,
            runtime_metadata={
                "user_id": str(access_scope.user_id),
                "session_id": str(access_scope.session_id),
                "workspace_id": str(access_scope.workspace_id),
                "source_run_id": current_run_id,
                "target_run_id": target_ref.target_run_id,
            },
        )
        async with self._uow_factory() as uow:
            existing_record = await uow.workflow_run.get_event_record_by_type_and_hash(
                user_id=str(access_scope.user_id),
                session_id=str(access_scope.session_id),
                run_id=current_run_id,
                event_type="feedback_input",
                input_hash=input_hash,
            )
            if existing_record is not None and isinstance(existing_record.event_payload, FeedbackInputEvent):
                return FeedbackInputSourceEventResult(
                    event=existing_record.event_payload,
                    event_record=existing_record,
                    reused=True,
                )

            event = FeedbackInputEvent(
                id=self._build_feedback_input_event_id(
                    user_id=str(access_scope.user_id),
                    session_id=str(access_scope.session_id),
                    run_id=current_run_id,
                    input_hash=input_hash,
                ),
                payload=payload,
            )
            inserted = await uow.workflow_run.add_event_record_if_absent(
                session_id=str(access_scope.session_id),
                run_id=current_run_id,
                event=event,
            )
            if not inserted:
                existing_after_insert = await uow.workflow_run.get_event_record_by_type_and_hash(
                    user_id=str(access_scope.user_id),
                    session_id=str(access_scope.session_id),
                    run_id=current_run_id,
                    event_type="feedback_input",
                    input_hash=input_hash,
                )
                if existing_after_insert is not None and isinstance(existing_after_insert.event_payload, FeedbackInputEvent):
                    return FeedbackInputSourceEventResult(
                        event=existing_after_insert.event_payload,
                        event_record=existing_after_insert,
                        reused=True,
                    )
                raise FeedbackRequiredRecordError("feedback_input source event 写入失败")

            created_record = await uow.workflow_run.get_event_record_by_event_id(
                user_id=str(access_scope.user_id),
                session_id=str(access_scope.session_id),
                run_id=current_run_id,
                event_id=event.id,
            )
            if created_record is None or not isinstance(created_record.event_payload, FeedbackInputEvent):
                raise FeedbackRequiredRecordError("feedback_input source event 回读失败")
            return FeedbackInputSourceEventResult(
                event=created_record.event_payload,
                event_record=created_record,
                reused=False,
            )

    async def _load_owned_wait_event(
            self,
            *,
            access_scope: AccessScopeResult,
            wait_event_id: str,
            run_id: str,
    ) -> WorkflowRunEventRecord:
        normalized_wait_event_id = str(wait_event_id or "").strip()
        normalized_run_id = str(run_id or "").strip()
        if not normalized_wait_event_id or not normalized_run_id:
            raise FeedbackRequiredRecordError("continue_cancelled 用户反馈缺少旧 wait event 或旧 run")
        async with self._uow_factory() as uow:
            run = await uow.workflow_run.get_by_id_for_user_session(
                run_id=normalized_run_id,
                user_id=str(access_scope.user_id),
                session_id=str(access_scope.session_id),
            )
            if run is None:
                raise FeedbackRequiredRecordError("continue_cancelled 旧 run 不属于当前 session")
            record = await uow.workflow_run.get_event_record_by_event_id(
                user_id=str(access_scope.user_id),
                session_id=str(access_scope.session_id),
                run_id=normalized_run_id,
                event_id=normalized_wait_event_id,
            )
            if record is None or record.event_type != "wait":
                raise FeedbackRequiredRecordError("continue_cancelled 旧 wait event 不存在或不属于旧 run")
            return record

    def _build_user_feedback_command(
            self,
            *,
            access_scope: AccessScopeResult,
            source_kind: FeedbackSourceKind,
            source_event_id: str,
            source_run_id: str | None,
            source_summary: str,
            source_record_refs: list[dict[str, str | None]],
            intent: UserFeedbackIntent,
    ) -> UserFeedbackCommand:
        category = self._capture_policy.category_for_intent(intent.intent_kind)
        resolved_scope_kind = self._capture_policy.resolve_scope_kind(intent.intent_kind)
        requested_scope_id = self._resolve_scope_id(access_scope=access_scope, scope_kind=resolved_scope_kind)
        normalized_summary = self._sanitize_user_feedback_summary(intent.summary_hint or source_summary)
        classification = FeedbackClassificationResult(
            privacy_level=PrivacyLevel.PRIVATE,
            retention_policy=RetentionPolicyKind.SESSION_BOUND,
            trust_level=DataTrustLevel.USER_PROVIDED,
            source_confidence=FeedbackSourceConfidence.STRONG,
            data_origin=FeedbackDataOrigin.USER,
        )
        return UserFeedbackCommand(
            access_scope=access_scope,
            source_ref=FeedbackSourceRefResult(
                source_kind=source_kind,
                source_event_id=source_event_id,
                source_record_refs=source_record_refs,
                source_run_id=source_run_id,
                source_step_id=access_scope.current_step_id,
                source_summary=normalized_summary,
            ),
            target_ref=intent.target_ref,
            category=category,
            reason_code=intent.reason_code,
            feedback_summary=FeedbackSummaryResult(
                summary_text=normalized_summary,
                summary_kind=FeedbackSummaryKind.USER_STATED,
                is_truncated=False,
                truncation_reason=None,
                language="zh-CN",
            ),
            prompt_safe_summary=FeedbackPromptSafeSummaryResult(
                summary_text=normalized_summary,
                is_truncated=False,
                sanitization_applied=True,
                sanitization_reasons=["ingress_sanitized"],
                prompt_visible=True,
            ),
            classification=classification,
            requested_feedback_scope_kind=resolved_scope_kind,
            requested_scope_id=requested_scope_id,
            current_run_id_at_record_time=str(access_scope.run_id) if access_scope.run_id else None,
            step_id=access_scope.current_step_id,
            profile_hash=None,
            decay_policy="user_feedback",
            ttl_scope="session_scoped",
            origin=DataOrigin.USER_MESSAGE,
            trust_level=classification.trust_level,
            privacy_level=classification.privacy_level,
            retention_policy=classification.retention_policy,
            intent=intent,
        )

    @staticmethod
    def _build_feedback_input_hash(
            *,
            source_action: str,
            intent_kind: UserFeedbackIntentKind,
            target_ref: FeedbackTargetRefResult,
            reason_code: FeedbackReasonCode,
            summary_hint: str | None,
    ) -> str:
        payload = {
            "source_action": source_action,
            "intent_kind": intent_kind.value,
            "target_ref": target_ref.model_dump(mode="json"),
            "reason_code": reason_code.value,
            "sanitized_summary": UserFeedbackIngressService._sanitize_feedback_input_summary(summary_hint),
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return f"feedback_input:{digest}"

    @staticmethod
    def _build_feedback_input_event_id(
            *,
            user_id: str,
            session_id: str,
            run_id: str,
            input_hash: str,
    ) -> str:
        payload = {
            "user_id": user_id,
            "session_id": session_id,
            "run_id": run_id,
            "event_type": "feedback_input",
            "input_hash": input_hash,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        return f"feedback_input_{digest}"

    @staticmethod
    def _normalize_summary_text(value: str | None) -> str:
        normalized = str(value or "").strip()
        if normalized:
            return normalized
        return "用户提交了结构化反馈"

    @staticmethod
    def _sanitize_user_feedback_summary(value: str | None) -> str:
        return FeedbackSanitizer.sanitize_diagnostic_summary(
            UserFeedbackIngressService._normalize_summary_text(value)
        )

    @staticmethod
    def _sanitize_feedback_input_summary(value: str | None) -> str | None:
        if value is None or not str(value).strip():
            return None
        return FeedbackSanitizer.sanitize_diagnostic_summary(value)

    @staticmethod
    def _resolve_scope_id(
            *,
            access_scope: AccessScopeResult,
            scope_kind: FeedbackScopeKind,
    ) -> str:
        if scope_kind == FeedbackScopeKind.SESSION:
            return str(access_scope.session_id)
        if not access_scope.run_id:
            raise FeedbackRequiredRecordError("run scope 用户反馈缺少 current run")
        return str(access_scope.run_id)

    @staticmethod
    def validate_submit_request(request: SubmitFeedbackRequest) -> None:
        allowed_by_action: dict[str, set[UserFeedbackIntentKind]] = {
            "final_satisfaction": {
                UserFeedbackIntentKind.SATISFACTION,
                UserFeedbackIntentKind.DISSATISFACTION,
                UserFeedbackIntentKind.CORRECTION,
            },
            "artifact_satisfaction": {
                UserFeedbackIntentKind.SATISFACTION,
                UserFeedbackIntentKind.DISSATISFACTION,
                UserFeedbackIntentKind.CORRECTION,
            },
            "explicit_correction": {
                UserFeedbackIntentKind.CORRECTION,
            },
            "explicit_preference": {
                UserFeedbackIntentKind.PREFERENCE,
            },
        }
        if request.source_action not in _HTTP_SOURCE_ACTIONS:
            raise FeedbackRequiredRecordError("不支持的 feedback source_action")
        allowed_intents = allowed_by_action[request.source_action]
        if request.intent_kind not in allowed_intents:
            raise FeedbackRequiredRecordError("feedback source_action 与 intent_kind 不匹配")
        if (
                request.source_action == "final_satisfaction"
                and request.target_ref.target_type.value != "message_event"
        ):
            raise FeedbackRequiredRecordError("final_satisfaction 必须绑定 final message event")
        if (
                request.source_action == "artifact_satisfaction"
                and request.target_ref.target_type.value != "artifact_revision"
        ):
            raise FeedbackRequiredRecordError("artifact_satisfaction 必须绑定 artifact revision")

    @staticmethod
    def validate_feedback_input_source_action(source_action: str) -> None:
        if source_action not in _HTTP_SOURCE_ACTIONS and source_action not in _CONTROL_SOURCE_ACTIONS:
            raise FeedbackRequiredRecordError("不支持的 feedback_input source_action")

    @staticmethod
    def _resolve_wait_resume_feedback_semantics(
            *,
            payload_kind: str,
            resume_value: object,
            payload: dict[str, object],
    ) -> tuple[UserFeedbackIntentKind, FeedbackReasonCode]:
        if payload_kind == "confirm":
            if resume_value == payload.get("confirm_resume_value"):
                return UserFeedbackIntentKind.CONFIRMATION, FeedbackReasonCode.USER_CONFIRMED
            return UserFeedbackIntentKind.CONFIRMATION, FeedbackReasonCode.USER_REJECTED
        if payload_kind == "select":
            return UserFeedbackIntentKind.SELECTION, FeedbackReasonCode.USER_SELECTED_OPTION
        if payload_kind == "input_text":
            return UserFeedbackIntentKind.CLARIFICATION, FeedbackReasonCode.USER_PROVIDED_CLARIFICATION
        raise FeedbackRequiredRecordError("不支持的 wait/resume payload kind")


__all__ = [
    "FeedbackInputSourceEventResult",
    "UserFeedbackCapturePolicy",
    "UserFeedbackIngressService",
]
