from app.domain.services.runtime.contracts.document_input_contract import (
    DocumentInputKind,
    DocumentInputPart,
    DocumentInputSourceRef,
    DocumentParseStatus,
    is_document_input_part,
)


def test_document_input_part_should_use_document_as_only_runtime_attachment_type() -> None:
    source = DocumentInputSourceRef(
        file_id="file-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
        sandbox_filepath="/workspace/notes.md",
        filename="notes.md",
        mime_type="text/markdown",
        extension=".md",
        size=128,
        sha256="a" * 64,
    )

    part = DocumentInputPart(
        kind=DocumentInputKind.MARKDOWN,
        source=source,
        parse_status=DocumentParseStatus.ACCEPTED,
        text_excerpt="hello",
    )

    dumped = part.model_dump()
    assert dumped["type"] == "document"
    assert dumped["kind"] == "markdown"
    assert dumped["source"]["file_id"] == "file-1"
    assert is_document_input_part(dumped) is True
    assert is_document_input_part({"type": "image"}) is False
