import asyncio
from io import BytesIO

import pytest

from app.application.service.document_input_service import (
    DocumentAttachmentSource,
    DocumentInputScopeMismatchError,
    DocumentInputService,
    FileStorageDocumentAttachmentReader,
)
from app.domain.services.runtime.contracts.access_scope_contract import AccessScopeResult
from app.domain.models import File


class _Reader:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self.is_truncated = False

    async def open_stream(self):
        return BytesIO(self._data)

    async def compute_sha256(self) -> str:
        return "a" * 64


class _ChunkedStream:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0
        self.read_sizes: list[int] = []

    def read(self, size: int = -1):
        self.read_sizes.append(size)
        if size < 0:
            size = len(self._data) - self._offset
        chunk = self._data[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk


class _FileStorage:
    def __init__(self, stream: _ChunkedStream) -> None:
        self.stream = stream

    async def download_file(self, file_id: str, user_id: str | None = None):
        return self.stream, File(id=file_id)

    def get_file_url(self, file: File) -> str:
        return ""


def _scope() -> AccessScopeResult:
    return AccessScopeResult(
        tenant_id="user-1",
        user_id="user-1",
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
    )


def _source(file: File, data: bytes) -> DocumentAttachmentSource:
    return DocumentAttachmentSource(
        scope=_scope(),
        file=file,
        sandbox_filepath=f"/home/ubuntu/upload/{file.id}/{file.filename}",
        reader=_Reader(data),
    )


def test_document_input_service_should_parse_markdown_as_document_part() -> None:
    service = DocumentInputService()
    file = File(
        id="file-1",
        user_id="user-1",
        filename="notes.md",
        extension=".md",
        mime_type="text/markdown",
        size=12,
    )

    parts = asyncio.run(service.build_input_parts(scope=_scope(), attachments=[_source(file, b"# hello\nbody")]))

    assert len(parts) == 1
    part = parts[0]
    assert part.type == "document"
    assert part.kind.value == "markdown"
    assert part.parse_status.value == "parsed"
    assert part.text_excerpt == "# hello\nbody"
    assert part.source.file_id == "file-1"
    assert part.source.sandbox_filepath == "/home/ubuntu/upload/file-1/notes.md"
    assert part.source.sha256 == "a" * 64


def test_pdf_should_return_unsupported_document_format_until_parser_exists() -> None:
    service = DocumentInputService()
    file = File(
        id="file-pdf",
        user_id="user-1",
        filename="paper.pdf",
        extension=".pdf",
        mime_type="application/pdf",
        size=128,
    )

    parts = asyncio.run(service.build_input_parts(scope=_scope(), attachments=[_source(file, b"%PDF")]))

    assert parts[0].type == "document"
    assert parts[0].kind.value == "pdf"
    assert parts[0].parse_status.value == "unsupported"
    assert parts[0].reason_code == "unsupported_document_format"


def test_document_input_service_should_fail_closed_on_scope_mismatch() -> None:
    service = DocumentInputService()
    file = File(id="file-1", filename="notes.txt", extension=".txt", mime_type="text/plain", size=5)
    mismatched_scope = _scope().model_copy(update={"session_id": "other-session"})
    source = DocumentAttachmentSource(
        scope=mismatched_scope,
        file=file,
        sandbox_filepath="/home/ubuntu/upload/file-1/notes.txt",
        reader=_Reader(b"hello"),
    )

    with pytest.raises(DocumentInputScopeMismatchError):
        asyncio.run(service.build_input_parts(scope=_scope(), attachments=[source]))


def test_document_input_reader_should_limit_reads_and_return_too_large_part() -> None:
    stream = _ChunkedStream(b"abcdef")
    reader = FileStorageDocumentAttachmentReader(
        file_storage=_FileStorage(stream),
        file=File(id="file-1", filename="big.txt", extension=".txt", mime_type="text/plain", size=6),
        user_id="user-1",
        max_bytes=3,
    )
    service = DocumentInputService()
    source = DocumentAttachmentSource(
        scope=_scope(),
        file=File(id="file-1", filename="big.txt", extension=".txt", mime_type="text/plain", size=6),
        sandbox_filepath="/home/ubuntu/upload/file-1/big.txt",
        reader=reader,
    )

    parts = asyncio.run(service.build_input_parts(scope=_scope(), attachments=[source]))

    assert parts[0].parse_status.value == "too_large"
    assert parts[0].reason_code == "too_large"
    assert parts[0].text_excerpt == "abc"
    assert reader.is_truncated is True
    assert -1 not in stream.read_sizes


def test_document_input_service_should_extract_visible_html_text() -> None:
    service = DocumentInputService()
    html_bytes = b"""
    <html><head><title>Doc Title</title><style>.x{display:none}</style></head>
    <body><script>alert('x')</script><h1>Hello</h1><p>Visible &amp; safe</p></body></html>
    """
    file = File(id="file-html", filename="page.html", extension=".html", mime_type="text/html", size=len(html_bytes))

    parts = asyncio.run(service.build_input_parts(scope=_scope(), attachments=[_source(file, html_bytes)]))

    assert parts[0].structured_preview == {"title": "Doc Title"}
    assert "Hello" in parts[0].text_excerpt
    assert "Visible & safe" in parts[0].text_excerpt
    assert "<script" not in parts[0].text_excerpt
    assert "alert" not in parts[0].text_excerpt
    assert "display:none" not in parts[0].text_excerpt
    assert "<h1>" not in parts[0].text_excerpt


def test_document_input_service_should_build_prompt_context_with_status_and_budget() -> None:
    service = DocumentInputService()
    parsed_file = File(id="file-1", filename="notes.md", extension=".md", mime_type="text/markdown", size=120)
    unsupported_file = File(id="file-2", filename="paper.pdf", extension=".pdf", mime_type="application/pdf", size=120)
    parts = [
        *asyncio.run(service.build_input_parts(scope=_scope(), attachments=[_source(parsed_file, b"abcdef")])),
        *asyncio.run(service.build_input_parts(scope=_scope(), attachments=[_source(unsupported_file, b"%PDF")])),
    ]

    context = service.build_prompt_context(parts=parts, max_chars=3)

    assert context.document_count == 2
    assert context.context_char_count == 3
    assert context.documents[0].file_id == "file-1"
    assert context.documents[0].sandbox_filepath == "/home/ubuntu/upload/file-1/notes.md"
    assert context.documents[0].parse_status == "parsed"
    assert context.documents[0].is_truncated is True
    assert context.documents[0].summary == "mar"
    assert context.documents[0].text_excerpt == ""
    assert context.documents[1].parse_status == "unsupported"
    assert context.documents[1].reason_code == "unsupported_document_format"
