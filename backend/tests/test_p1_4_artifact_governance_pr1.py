import asyncio
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql

from app.domain.models import WorkspaceArtifact, WorkspaceArtifactRevision
from app.domain.services.runtime.contracts.artifact_governance_contract import (
    ArtifactDeliveryState,
    ArtifactEventPayload,
    ArtifactRevisionSourceKind,
    ArtifactStorageBackend,
    ArtifactStorageRef,
    ArtifactType,
    SelectedArtifactRevisionResult,
)
from app.domain.services.runtime.contracts.data_access_contract import (
    DataOrigin,
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)
from app.infrastructure.models import WorkspaceArtifactModel
from app.infrastructure.models import WorkspaceArtifactRevisionModel
from app.infrastructure.repositories.db_workspace_artifact_revision_repository import (
    DBWorkspaceArtifactRevisionRepository,
)


def _file_storage_ref(*, file_id: str = "file-1", content_hash: str = "sha256:" + "a" * 64) -> ArtifactStorageRef:
    return ArtifactStorageRef(
        storage_backend=ArtifactStorageBackend.FILE_STORAGE,
        object_key="objects/report.md",
        file_id=file_id,
        sandbox_path=None,
        storage_hash=content_hash,
        size_bytes=128,
        mime_type="text/markdown",
        missing_fields=[],
        reason_code=None,
    )


def _revision(
        *,
        revision_id: str = "revision-1",
        artifact_id: str = "artifact-1",
        revision_no: int = 1,
        content_hash: str = "sha256:" + "a" * 64,
        source_kind: ArtifactRevisionSourceKind = ArtifactRevisionSourceKind.TOOL_WRITE_FILE,
        storage_ref: ArtifactStorageRef | None = None,
) -> WorkspaceArtifactRevision:
    return WorkspaceArtifactRevision(
        revision_id=revision_id,
        artifact_id=artifact_id,
        revision_no=revision_no,
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        step_id="step-1",
        path="/workspace/report.md",
        storage_ref=storage_ref or _file_storage_ref(content_hash=content_hash),
        content_hash=content_hash,
        storage_hash=content_hash,
        hash_algorithm="sha256",
        size_bytes=128,
        mime_type="text/markdown",
        artifact_type=ArtifactType.FILE,
        delivery_state=ArtifactDeliveryState.CANDIDATE,
        source_kind=source_kind,
        source_event_id="event-1",
        source_run_id="run-1",
        source_message_event_id=None,
        source_revision_id=None,
        source_fact_ids=["fact-1"],
        source_evidence_ids=[],
        source_final_answer_hash=content_hash if source_kind == ArtifactRevisionSourceKind.FINAL_ANSWER_SNAPSHOT else None,
        derived_content_hash=None,
        tool_call_id="tool-call-1",
        function_name="write_file",
        profile_hash="sha256:" + "b" * 64,
        profile_status="available",
        origin=DataOrigin.AGENT_GENERATED,
        trust_level=DataTrustLevel.AGENT_GENERATED,
        privacy_level=PrivacyLevel.PRIVATE,
        retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
        metadata={"summary": "report"},
        created_at=datetime(2026, 5, 13, 10, 0, 0),
    )


def test_artifact_storage_ref_should_reject_extra_fields_and_unsafe_values() -> None:
    with pytest.raises(ValidationError):
        ArtifactStorageRef(
            storage_backend="file_storage",
            object_key="objects/report.md",
            file_id="file-1",
            sandbox_path=None,
            storage_hash="sha256:" + "a" * 64,
            size_bytes=128,
            mime_type="text/markdown",
            missing_fields=[],
            reason_code=None,
            public_url="https://example.com/file",
        )

    unsafe_payloads = [
        {"object_key": "https://example.com/file"},
        {"object_key": "file:///tmp/file"},
        {"object_key": "/Users/caixr/private.txt"},
        {"object_key": "Bearer secret-token"},
        {"sandbox_path": "/Users/caixr/private.txt"},
        {"sandbox_path": "https://example.com/file"},
        {"sandbox_path": "file:///tmp/file"},
        {"sandbox_path": "C:\\Users\\secret.txt"},
        {"sandbox_path": "\\\\server\\share\\secret.txt"},
        {"sandbox_path": "../secret.txt"},
    ]
    for payload in unsafe_payloads:
        values = {
            "storage_backend": "sandbox",
            "object_key": None,
            "file_id": None,
            "sandbox_path": payload.get("sandbox_path", "/home/ubuntu/report.md"),
            "storage_hash": "sha256:" + "a" * 64,
            "size_bytes": 128,
            "mime_type": "text/markdown",
            "missing_fields": [],
            "reason_code": None,
        }
        values.update(payload)
        with pytest.raises(ValidationError):
            ArtifactStorageRef(**values)


def test_artifact_storage_ref_should_support_three_backends_with_missing_field_rules() -> None:
    file_storage = _file_storage_ref()
    assert file_storage.storage_backend == ArtifactStorageBackend.FILE_STORAGE
    assert file_storage.file_id == "file-1"

    sandbox = ArtifactStorageRef(
        storage_backend=ArtifactStorageBackend.SANDBOX,
        object_key=None,
        file_id=None,
        sandbox_path="/home/ubuntu/report.md",
        storage_hash="sha256:" + "a" * 64,
        size_bytes=128,
        mime_type="text/markdown",
        missing_fields=["file_id"],
        reason_code="sandbox_storage_not_deliverable",
    )
    assert sandbox.sandbox_path == "/home/ubuntu/report.md"

    for sandbox_path in ["/workspace/report.md", "/tmp/report.md", "relative/report.md"]:
        assert ArtifactStorageRef(
            storage_backend=ArtifactStorageBackend.SANDBOX,
            object_key=None,
            file_id=None,
            sandbox_path=sandbox_path,
            storage_hash="sha256:" + "a" * 64,
            size_bytes=128,
            mime_type="text/markdown",
            missing_fields=["file_id"],
            reason_code="sandbox_storage_not_deliverable",
        ).sandbox_path == sandbox_path

    inline_snapshot = ArtifactStorageRef(
        storage_backend=ArtifactStorageBackend.INLINE_SNAPSHOT,
        object_key=None,
        file_id=None,
        sandbox_path=None,
        storage_hash=None,
        size_bytes=None,
        mime_type=None,
        missing_fields=["storage_hash", "size_bytes", "mime_type"],
        reason_code="inline_snapshot_no_materialized_storage",
    )
    assert inline_snapshot.storage_hash is None
    assert inline_snapshot.size_bytes is None
    assert inline_snapshot.mime_type is None

    with pytest.raises(ValidationError):
        ArtifactStorageRef(
            storage_backend="derived_export",
            object_key=None,
            file_id=None,
            sandbox_path=None,
            storage_hash=None,
            size_bytes=None,
            mime_type=None,
            missing_fields=[],
            reason_code=None,
        )


def test_workspace_artifact_revision_persistence_payload_should_keep_created_at_datetime() -> None:
    revision = _revision()

    insert_values = DBWorkspaceArtifactRevisionRepository._to_insert_values(revision)
    orm_model = WorkspaceArtifactRevisionModel.from_domain(revision)

    assert isinstance(insert_values["created_at"], datetime)
    assert insert_values["created_at"] == revision.created_at
    assert isinstance(orm_model.created_at, datetime)
    assert orm_model.created_at == revision.created_at


def test_workspace_artifact_revision_should_enforce_enums_and_final_answer_hash() -> None:
    source_hash = "sha256:" + "c" * 64
    revision_payload = _revision(content_hash=source_hash).model_dump(mode="json")
    revision = WorkspaceArtifactRevision.model_validate(
        {
            **revision_payload,
            "artifact_type": ArtifactType.FINAL_ANSWER_SNAPSHOT,
            "source_kind": ArtifactRevisionSourceKind.FINAL_ANSWER_SNAPSHOT,
            "storage_ref": {
                "storage_backend": ArtifactStorageBackend.INLINE_SNAPSHOT,
                "object_key": None,
                "file_id": None,
                "sandbox_path": None,
                "storage_hash": None,
                "size_bytes": None,
                "mime_type": None,
                "missing_fields": ["storage_hash", "size_bytes", "mime_type"],
                "reason_code": "inline_snapshot_no_materialized_storage",
            },
            "source_event_id": "final-message-event-1",
            "source_message_event_id": "final-message-event-1",
            "source_final_answer_hash": source_hash,
            "tool_call_id": None,
            "function_name": None,
            "size_bytes": 0,
            "mime_type": "text/markdown",
            "storage_hash": None,
        }
    )
    assert revision.content_hash == revision.source_final_answer_hash
    assert revision.revision_id == "revision-1"
    assert "id" not in revision.model_dump(mode="json")

    with pytest.raises(ValidationError):
        WorkspaceArtifactRevision.model_validate(
            {
                **revision.model_dump(mode="json"),
                "content_hash": "sha256:" + "d" * 64,
            }
        )

    with pytest.raises(ValidationError):
        WorkspaceArtifactRevision.model_validate(
            {
                **_revision().model_dump(mode="json"),
                "artifact_type": "browser_screenshot",
            }
        )


def test_selected_artifact_revision_and_event_payload_should_require_revision_identity() -> None:
    selected = SelectedArtifactRevisionResult(
        artifact_id="artifact-1",
        revision_id="revision-1",
        content_hash="sha256:" + "a" * 64,
        path="/workspace/report.md",
        artifact_type=ArtifactType.FILE,
        delivery_state=ArtifactDeliveryState.SELECTED,
        session_id="session-1",
        run_id="run-1",
        source_run_id="run-1",
        source_step_id="step-1",
        source_event_id="event-1",
        source_kind=ArtifactRevisionSourceKind.TOOL_WRITE_FILE,
        selected_reason="summary_selected",
        selected_at=datetime(2026, 5, 13, 10, 1, 0),
    )
    assert selected.revision_id == "revision-1"
    assert "id" not in selected.model_dump(mode="json")

    event = ArtifactEventPayload(
        artifact_refs=[
            {
                "artifact_id": "artifact-1",
                "path": "/workspace/report.md",
                "artifact_type": "file",
                "delivery_state": "candidate",
                "current_revision_id": "revision-1",
                "latest_content_hash": "sha256:" + "a" * 64,
            }
        ],
        revision_refs=[
            {
                "artifact_id": "artifact-1",
                "revision_id": "revision-1",
                "content_hash": "sha256:" + "a" * 64,
                "path": "/workspace/report.md",
                "artifact_type": "file",
                "delivery_state": "candidate",
                "source_event_id": "event-1",
            }
        ],
        counts={"revision_count": 1},
        summary="registered 1 artifact revision",
        source_event_ids=["event-1"],
        runtime_metadata={},
    )
    assert event.artifact_refs[0].artifact_id == "artifact-1"
    assert event.revision_refs[0].revision_id == "revision-1"

    with pytest.raises(ValidationError):
        ArtifactEventPayload(
            revision_refs=[
                {
                    "artifact_id": "artifact-1",
                    "content_hash": "sha256:" + "a" * 64,
                    "path": "/workspace/report.md",
                    "artifact_type": "file",
                    "delivery_state": "candidate",
                    "source_event_id": "event-1",
                }
            ],
            artifact_refs=[],
            counts={"revision_count": 1},
            summary="invalid",
            source_event_ids=["event-1"],
            runtime_metadata={},
        )

    with pytest.raises(ValidationError):
        ArtifactEventPayload(
            artifact_refs=[
                {
                    "artifact_id": "artifact-1",
                    "path": "/workspace/report.md",
                    "artifact_type": "file",
                    "delivery_state": "candidate",
                    "public_url": "https://example.com/report.md",
                    "metadata": {"raw": "unexpected"},
                }
            ],
            revision_refs=[],
            counts={"artifact_count": 1},
            summary="invalid",
            source_event_ids=[],
            runtime_metadata={},
        )


def test_revision_repository_queries_should_filter_user_and_workspace() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(
            return_value=SimpleNamespace(scalars=lambda: SimpleNamespace(all=lambda: []))
        )
    )
    repository = DBWorkspaceArtifactRevisionRepository(db_session=db_session)

    asyncio.run(
        repository.list_by_user_workspace_artifact_id(
            user_id="user-1",
            workspace_id="workspace-1",
            artifact_id="artifact-1",
        )
    )

    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "workspace_artifact_revisions.user_id" in compiled_sql
    assert "workspace_artifact_revisions.workspace_id" in compiled_sql
    assert "workspace_artifact_revisions.artifact_id" in compiled_sql


def test_revision_repository_idempotent_insert_should_return_existing_revision() -> None:
    existing = _revision(revision_id="revision-existing")
    inserted = _revision(revision_id="revision-new")
    db_session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                SimpleNamespace(scalar_one_or_none=lambda: "artifact-1"),
                SimpleNamespace(scalar_one_or_none=lambda: None),
                SimpleNamespace(scalar_one_or_none=lambda: inserted.revision_id),
                SimpleNamespace(),
                SimpleNamespace(scalar_one_or_none=lambda: inserted),
                SimpleNamespace(scalar_one_or_none=lambda: "artifact-1"),
                SimpleNamespace(scalar_one_or_none=lambda: existing),
            ]
        )
    )
    repository = DBWorkspaceArtifactRevisionRepository(db_session=db_session)

    first = asyncio.run(repository.insert_or_get_existing(inserted))
    second = asyncio.run(repository.insert_or_get_existing(inserted))

    assert first.revision_id == "revision-new"
    assert second.revision_id == "revision-existing"


def test_revision_repository_idempotent_insert_should_return_existing_final_answer_snapshot_revision() -> None:
    source_hash = "sha256:" + "c" * 64
    existing = WorkspaceArtifactRevision(
        revision_id="revision-final-existing",
        artifact_id="artifact-1",
        revision_no=1,
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        step_id="step-1",
        path="/workspace/final.md",
        storage_ref=ArtifactStorageRef(
            storage_backend=ArtifactStorageBackend.INLINE_SNAPSHOT,
            object_key=None,
            file_id=None,
            sandbox_path=None,
            storage_hash=None,
            size_bytes=None,
            mime_type=None,
            missing_fields=["storage_hash", "size_bytes", "mime_type"],
            reason_code="inline_snapshot_no_materialized_storage",
        ),
        content_hash=source_hash,
        storage_hash=None,
        hash_algorithm="sha256",
        size_bytes=None,
        mime_type=None,
        artifact_type=ArtifactType.FINAL_ANSWER_SNAPSHOT,
        delivery_state=ArtifactDeliveryState.CANDIDATE,
        source_kind=ArtifactRevisionSourceKind.FINAL_ANSWER_SNAPSHOT,
        source_event_id="final-message-event-1",
        source_run_id="run-1",
        source_message_event_id="final-message-event-1",
        source_revision_id=None,
        source_fact_ids=[],
        source_evidence_ids=[],
        source_final_answer_hash=source_hash,
        derived_content_hash=None,
        tool_call_id=None,
        function_name=None,
        profile_hash="sha256:" + "b" * 64,
        profile_status="available",
        origin=DataOrigin.AGENT_GENERATED,
        trust_level=DataTrustLevel.AGENT_GENERATED,
        privacy_level=PrivacyLevel.PRIVATE,
        retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
        metadata={"summary": "final snapshot"},
        created_at=datetime(2026, 5, 13, 10, 0, 0),
    )
    inserted = WorkspaceArtifactRevision(
        revision_id="revision-final-new",
        artifact_id="artifact-1",
        revision_no=1,
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        step_id="step-1",
        path="/workspace/final.md",
        storage_ref=ArtifactStorageRef(
            storage_backend=ArtifactStorageBackend.INLINE_SNAPSHOT,
            object_key=None,
            file_id=None,
            sandbox_path=None,
            storage_hash=None,
            size_bytes=None,
            mime_type=None,
            missing_fields=["storage_hash", "size_bytes", "mime_type"],
            reason_code="inline_snapshot_no_materialized_storage",
        ),
        content_hash=source_hash,
        storage_hash=None,
        hash_algorithm="sha256",
        size_bytes=None,
        mime_type=None,
        artifact_type=ArtifactType.FINAL_ANSWER_SNAPSHOT,
        delivery_state=ArtifactDeliveryState.CANDIDATE,
        source_kind=ArtifactRevisionSourceKind.FINAL_ANSWER_SNAPSHOT,
        source_event_id="final-message-event-1",
        source_run_id="run-1",
        source_message_event_id="final-message-event-1",
        source_revision_id=None,
        source_fact_ids=[],
        source_evidence_ids=[],
        source_final_answer_hash=source_hash,
        derived_content_hash=None,
        tool_call_id=None,
        function_name=None,
        profile_hash="sha256:" + "b" * 64,
        profile_status="available",
        origin=DataOrigin.AGENT_GENERATED,
        trust_level=DataTrustLevel.AGENT_GENERATED,
        privacy_level=PrivacyLevel.PRIVATE,
        retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
        metadata={"summary": "final snapshot"},
        created_at=datetime(2026, 5, 13, 10, 0, 0),
    )
    db_session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                SimpleNamespace(scalar_one_or_none=lambda: "artifact-1"),
                SimpleNamespace(scalar_one_or_none=lambda: None),
                SimpleNamespace(scalar_one_or_none=lambda: inserted.revision_id),
                SimpleNamespace(),
                SimpleNamespace(scalar_one_or_none=lambda: inserted),
                SimpleNamespace(scalar_one_or_none=lambda: "artifact-1"),
                SimpleNamespace(scalar_one_or_none=lambda: existing),
            ]
        )
    )
    repository = DBWorkspaceArtifactRevisionRepository(db_session=db_session)

    first = asyncio.run(repository.insert_or_get_existing(inserted))
    second = asyncio.run(repository.insert_or_get_existing(inserted))

    assert first.revision_id == "revision-final-new"
    assert second.revision_id == "revision-final-existing"


def test_revision_repository_idempotent_insert_should_return_existing_derived_export_revision() -> None:
    source_hash = "sha256:" + "e" * 64
    content_hash = "sha256:" + "f" * 64
    existing = WorkspaceArtifactRevision(
        revision_id="revision-derived-existing",
        artifact_id="artifact-1",
        revision_no=1,
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        step_id="step-1",
        path="/workspace/final-export.md",
        storage_ref=ArtifactStorageRef(
            storage_backend=ArtifactStorageBackend.FILE_STORAGE,
            object_key="objects/final-export.md",
            file_id="file-1",
            sandbox_path=None,
            storage_hash=content_hash,
            size_bytes=256,
            mime_type="text/markdown",
            missing_fields=[],
            reason_code=None,
        ),
        content_hash=content_hash,
        storage_hash=content_hash,
        hash_algorithm="sha256",
        size_bytes=256,
        mime_type="text/markdown",
        artifact_type=ArtifactType.REPORT,
        delivery_state=ArtifactDeliveryState.CANDIDATE,
        source_kind=ArtifactRevisionSourceKind.DERIVED_EXPORT,
        source_event_id="export-event-1",
        source_run_id="run-1",
        source_message_event_id=None,
        source_revision_id="revision-source-1",
        source_fact_ids=[],
        source_evidence_ids=[],
        source_final_answer_hash=source_hash,
        derived_content_hash=content_hash,
        tool_call_id=None,
        function_name=None,
        profile_hash="sha256:" + "b" * 64,
        profile_status="available",
        origin=DataOrigin.AGENT_GENERATED,
        trust_level=DataTrustLevel.AGENT_GENERATED,
        privacy_level=PrivacyLevel.PRIVATE,
        retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
        metadata={"summary": "derived export"},
        created_at=datetime(2026, 5, 13, 10, 0, 0),
    )
    inserted = WorkspaceArtifactRevision(
        revision_id="revision-derived-new",
        artifact_id="artifact-1",
        revision_no=1,
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        step_id="step-1",
        path="/workspace/final-export.md",
        storage_ref=ArtifactStorageRef(
            storage_backend=ArtifactStorageBackend.FILE_STORAGE,
            object_key="objects/final-export.md",
            file_id="file-1",
            sandbox_path=None,
            storage_hash=content_hash,
            size_bytes=256,
            mime_type="text/markdown",
            missing_fields=[],
            reason_code=None,
        ),
        content_hash=content_hash,
        storage_hash=content_hash,
        hash_algorithm="sha256",
        size_bytes=256,
        mime_type="text/markdown",
        artifact_type=ArtifactType.REPORT,
        delivery_state=ArtifactDeliveryState.CANDIDATE,
        source_kind=ArtifactRevisionSourceKind.DERIVED_EXPORT,
        source_event_id="export-event-1",
        source_run_id="run-1",
        source_message_event_id=None,
        source_revision_id="revision-source-1",
        source_fact_ids=[],
        source_evidence_ids=[],
        source_final_answer_hash=source_hash,
        derived_content_hash=content_hash,
        tool_call_id=None,
        function_name=None,
        profile_hash="sha256:" + "b" * 64,
        profile_status="available",
        origin=DataOrigin.AGENT_GENERATED,
        trust_level=DataTrustLevel.AGENT_GENERATED,
        privacy_level=PrivacyLevel.PRIVATE,
        retention_policy=RetentionPolicyKind.WORKSPACE_BOUND,
        metadata={"summary": "derived export"},
        created_at=datetime(2026, 5, 13, 10, 0, 0),
    )
    db_session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                SimpleNamespace(scalar_one_or_none=lambda: "artifact-1"),
                SimpleNamespace(scalar_one_or_none=lambda: None),
                SimpleNamespace(scalar_one_or_none=lambda: inserted.revision_id),
                SimpleNamespace(),
                SimpleNamespace(scalar_one_or_none=lambda: inserted),
                SimpleNamespace(scalar_one_or_none=lambda: "artifact-1"),
                SimpleNamespace(scalar_one_or_none=lambda: existing),
            ]
        )
    )
    repository = DBWorkspaceArtifactRevisionRepository(db_session=db_session)

    first = asyncio.run(repository.insert_or_get_existing(inserted))
    second = asyncio.run(repository.insert_or_get_existing(inserted))

    assert first.revision_id == "revision-derived-new"
    assert second.revision_id == "revision-derived-existing"


def test_workspace_artifact_model_should_include_current_revision_query_index() -> None:
    index_names = {index.name for index in WorkspaceArtifactModel.__table__.indexes}
    assert "ix_workspace_artifacts_user_workspace_current_revision" in index_names


def test_workspace_artifact_revision_model_should_include_partial_unique_indexes() -> None:
    index_names = {index.name for index in WorkspaceArtifactRevisionModel.__table__.indexes}
    assert "uq_war_tool_idem" in index_names
    assert "uq_war_final_snapshot_idem" in index_names
    assert "uq_war_derived_export_idem" in index_names


def test_workspace_artifact_model_should_include_current_revision_projection_fields() -> None:
    artifact = WorkspaceArtifact(
        id="artifact-1",
        workspace_id="workspace-1",
        user_id="user-1",
        session_id="session-1",
        run_id="run-1",
        path="/workspace/report.md",
        artifact_type="file",
        current_revision_id="revision-1",
        latest_content_hash="sha256:" + "a" * 64,
        latest_size=128,
        latest_mime_type="text/markdown",
        artifact_status="active",
    )

    assert artifact.current_revision_id == "revision-1"
    assert artifact.latest_content_hash == "sha256:" + "a" * 64
    assert artifact.latest_size == 128
    assert artifact.latest_mime_type == "text/markdown"
    assert artifact.artifact_status == "active"
