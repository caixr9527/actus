#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""持久化 final message 后创建 final_answer_snapshot revision。"""

from __future__ import annotations

import hashlib
from typing import Callable

from app.application.service.artifact_ledger_service import ArtifactLedgerService
from app.domain.models import MessageEvent
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactDeliveryState,
    ArtifactRevisionRegistrationCommand,
    ArtifactRevisionSourceKind,
    ArtifactStorageBackend,
    ArtifactStorageRef,
    ArtifactType,
)
from app.domain.services.runtime.contracts.artifact_governance_ports import FinalMessageArtifactProjectorPort
from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)


class FinalMessageArtifactProjector(FinalMessageArtifactProjectorPort):
    """将已落库 final message 投影为 inline snapshot revision。"""

    def __init__(
            self,
            *,
            uow_factory: Callable[[], IUnitOfWork],
            ledger_service: ArtifactLedgerService,
            user_id: str,
            session_id: str,
    ) -> None:
        self._uow_factory = uow_factory
        self._ledger_service = ledger_service
        self._user_id = str(user_id or "").strip()
        self._session_id = str(session_id or "").strip()

    async def project_final_message(
            self,
            *,
            event: MessageEvent,
            persisted_event_id: str,
            run_id: str,
    ) -> None:
        if event.stage != "final":
            return
        event_id = str(persisted_event_id or "").strip()
        if not event_id:
            raise RuntimeError("final_message_event_missing")
        normalized_run_id = str(run_id or "").strip()
        if not normalized_run_id:
            raise RuntimeError("final_message_event_missing")
        scope = await self._build_scope(run_id=normalized_run_id, event_id=event_id)
        text = str(event.message or "")
        encoded = text.encode("utf-8")
        content_hash = "sha256:" + hashlib.sha256(encoded).hexdigest()
        await self._ledger_service.register_revision(
            command=ArtifactRevisionRegistrationCommand(
                scope=scope,
                path=f"inline://final-answer/{event_id}",
                storage_ref=ArtifactStorageRef(
                    storage_backend=ArtifactStorageBackend.INLINE_SNAPSHOT,
                    missing_fields=["storage_hash", "size_bytes", "mime_type"],
                    reason_code="inline_snapshot_no_materialized_storage",
                ),
                content_hash=content_hash,
                size_bytes=len(encoded),
                mime_type="text/markdown",
                artifact_type=ArtifactType.FINAL_ANSWER_SNAPSHOT,
                delivery_state=ArtifactDeliveryState.CANDIDATE,
                source_kind=ArtifactRevisionSourceKind.FINAL_ANSWER_SNAPSHOT,
                source_event_id=event_id,
                source_run_id=scope.run_id,
                source_message_event_id=event_id,
                source_final_answer_hash=content_hash,
                origin=DataOrigin.AGENT_GENERATED,
                trust_level=DataTrustLevel.AGENT_GENERATED,
                privacy_level=PrivacyLevel.PRIVATE,
                retention_policy=RetentionPolicyKind.SESSION_BOUND,
                metadata={
                    "source_message_event_id": event_id,
                    "source_run_id": scope.run_id,
                    "text_length": len(text),
                    "byte_length": len(encoded),
                },
            )
        )

    async def _build_scope(self, *, run_id: str, event_id: str) -> AccessScopeResult:
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id_without_events(
                session_id=self._session_id,
                user_id=self._user_id,
            )
            if session is None:
                raise RuntimeError("final_message_event_missing")
            workspace = await uow.workspace.get_by_session_id_for_user(
                session_id=self._session_id,
                user_id=self._user_id,
            )
            if workspace is None:
                raise RuntimeError("artifact_scope_mismatch")
            event_record = await uow.workflow_run.get_event_record_by_event_id(
                user_id=self._user_id,
                session_id=self._session_id,
                run_id=run_id,
                event_id=event_id,
            )
            if event_record is None:
                raise RuntimeError("final_message_event_missing")
            if not isinstance(event_record.event_payload, MessageEvent) or event_record.event_payload.stage != "final":
                raise RuntimeError("final_message_event_missing")
        if not str(run_id or "").strip():
            raise RuntimeError("final_message_event_missing")
        return AccessScopeResult(
            tenant_id=self._user_id,
            user_id=self._user_id,
            session_id=self._session_id,
            workspace_id=workspace.id,
            run_id=run_id,
            current_step_id=None,
        )
