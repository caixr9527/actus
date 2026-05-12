import asyncio
from pathlib import Path

from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.application.service.sandbox_fact_document_input_projector import SandboxFactDocumentInputProjector
from app.application.service.sandbox_fact_ledger_service import SandboxFactLedgerService
from app.domain.models import ToolEvent, ToolEventStatus, ToolResult
from app.domain.models.sandbox_fact import SandboxFactKind, SandboxFactProfileRef, SandboxFactRecord
from app.domain.services.runtime.contracts.document_input_contract import (
    DocumentInputKind,
    DocumentInputPart,
    DocumentInputSourceRef,
    DocumentParseStatus,
)
from app.domain.services.runtime.contracts.sandbox_fact_ports import SandboxFactProjectionContext
from app.domain.services.workspace_runtime.projectors import SandboxFactToolEventProjector


class _SandboxFactRepo:
    def __init__(self) -> None:
        self.saved: list[SandboxFactRecord] = []

    async def save_once(self, fact: SandboxFactRecord) -> SandboxFactRecord:
        self.saved.append(fact)
        return fact


class _UoW:
    def __init__(self, repo: _SandboxFactRepo) -> None:
        self.sandbox_fact = repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


def _context(*, source_event_id: str | None = "message-event-1") -> SandboxFactProjectionContext:
    return SandboxFactProjectionContext(
        scope=AccessScopeResult(
            tenant_id="user-1",
            user_id="user-1",
            session_id="session-1",
            workspace_id="workspace-1",
            run_id="run-1",
        ),
        profile_ref=SandboxFactProfileRef(status="missing"),
        sandbox_id="sandbox-1",
        source_event_id=source_event_id,
    )


def _part(
        *,
        text_excerpt: str = "hello document",
        parse_status: DocumentParseStatus = DocumentParseStatus.PARSED,
        reason_code: str | None = None,
        is_truncated: bool = False,
) -> DocumentInputPart:
    return DocumentInputPart(
        kind=DocumentInputKind.TEXT,
        source=DocumentInputSourceRef(
            file_id="file-1",
            user_id="user-1",
            session_id="session-1",
            workspace_id="workspace-1",
            run_id="run-1",
            sandbox_filepath="/home/ubuntu/upload/file-1/note.txt",
            filename="note.txt",
            mime_type="text/plain",
            extension=".txt",
            size=32,
            sha256="sha256:full-file",
        ),
        parse_status=parse_status,
        text_excerpt=text_excerpt,
        reason_code=reason_code,
        is_truncated=is_truncated,
    )


def test_document_context_fact_should_preserve_document_hash_and_parse_status() -> None:
    repo = _SandboxFactRepo()
    projector = SandboxFactDocumentInputProjector(
        ledger_service=SandboxFactLedgerService(uow_factory=lambda: _UoW(repo)),
    )

    asyncio.run(projector.record_document_context(context=_context(), parts=[_part()]))

    assert len(repo.saved) == 1
    fact = repo.saved[0]
    assert fact.fact_kind == SandboxFactKind.DOCUMENT_CONTEXT
    assert fact.source_ref.source_event_id == "message-event-1"
    assert fact.source_ref.source_type.value == "document_input"
    assert fact.source_ref.document_source_id == "file-1"
    assert fact.payload["file_id"] == "file-1"
    assert fact.payload["full_file_sha256"] == "sha256:full-file"
    assert fact.payload["read_content_sha256"].startswith("sha256:")
    assert fact.payload["parse_status"] == "parsed"
    assert fact.payload["excerpt_char_count"] == len("hello document")


def test_document_context_fact_should_use_none_read_hash_without_excerpt() -> None:
    repo = _SandboxFactRepo()
    projector = SandboxFactDocumentInputProjector(
        ledger_service=SandboxFactLedgerService(uow_factory=lambda: _UoW(repo)),
    )

    asyncio.run(
        projector.record_document_context(
            context=_context(),
            parts=[
                _part(
                    text_excerpt="",
                    parse_status=DocumentParseStatus.UNSUPPORTED,
                    reason_code="unsupported_document_format",
                )
            ],
        )
    )

    assert repo.saved[0].payload["read_content_sha256"] is None
    assert repo.saved[0].payload["parse_status"] == "unsupported"
    assert repo.saved[0].payload["reason_code"] == "unsupported_document_format"
    assert repo.saved[0].payload["excerpt_char_count"] == 0


def test_document_context_fact_should_fail_closed_without_source_event_id() -> None:
    repo = _SandboxFactRepo()
    projector = SandboxFactDocumentInputProjector(
        ledger_service=SandboxFactLedgerService(uow_factory=lambda: _UoW(repo)),
    )

    asyncio.run(projector.record_document_context(context=_context(source_event_id=None), parts=[_part()]))

    assert repo.saved == []


def test_document_projector_should_not_use_legacy_environment_summary_without_parts() -> None:
    repo = _SandboxFactRepo()
    projector = SandboxFactDocumentInputProjector(
        ledger_service=SandboxFactLedgerService(uow_factory=lambda: _UoW(repo)),
    )

    asyncio.run(projector.record_document_context(context=_context(), parts=[]))

    assert repo.saved == []


def test_browser_snapshot_fact_should_not_use_environment_summary_as_screenshot_fallback() -> None:
    repo = _SandboxFactRepo()
    projector = SandboxFactToolEventProjector(
        ledger_service=SandboxFactLedgerService(uow_factory=lambda: _UoW(repo)),
    )
    event = ToolEvent(
        id="tool-event-1",
        tool_call_id="call-1",
        tool_name="browser",
        function_name="browser_view",
        function_args={},
        function_result=ToolResult(
            success=True,
            data={
                "url": "https://example.com",
                "title": "Example",
                "environment_summary": {
                    "screenshot_artifact": {
                        "artifact_id": "legacy-artifact",
                        "artifact_path": "/legacy/shot.png",
                    }
                },
            },
        ),
        status=ToolEventStatus.CALLED,
    )

    facts = asyncio.run(projector.record_from_tool_event(context=_context(), event=event))

    assert facts[0].payload["screenshot_artifact_id"] is None
    assert facts[0].payload["screenshot_artifact_path"] is None
    assert "screenshot_artifact" in facts[0].payload["missing_fields"]


def test_fact_related_modules_should_not_reference_environment_summary_fallback() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    fact_related_paths = [
        repo_root / "app/application/service/sandbox_fact_document_input_projector.py",
        repo_root / "app/domain/services/workspace_runtime/projectors/sandbox_fact_tool_event_projector.py",
        repo_root / "app/infrastructure/repositories/db_sandbox_fact_repository.py",
    ]

    for path in fact_related_paths:
        assert "environment_summary" not in path.read_text(encoding="utf-8")
