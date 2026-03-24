import asyncio
import io

from app.domain.models import File
from app.domain.services.agent_task_runner import AgentTaskRunner


class _FakeFileStorage:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    async def download_file(self, file_id: str, user_id: str | None = None):
        return io.BytesIO(self._payload), File(id=file_id, filename="demo.txt")

    def get_file_url(self, file: File) -> str:
        return f"https://example.com/files/{file.id}"


class _CountingBytesStream(io.BytesIO):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload)
        self.total_read_bytes = 0

    def read(self, size: int = -1) -> bytes:
        data = super().read(size)
        self.total_read_bytes += len(data or b"")
        return data


class _LargeFileStorage:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload
        self.last_stream: _CountingBytesStream | None = None

    async def download_file(self, file_id: str, user_id: str | None = None):
        stream = _CountingBytesStream(self._payload)
        self.last_stream = stream
        return stream, File(id=file_id, filename="large.txt")

    def get_file_url(self, file: File) -> str:
        return f"https://example.com/files/{file.id}"


def test_build_message_input_envelope_should_inline_text_attachment_content() -> None:
    runner = object.__new__(AgentTaskRunner)
    runner._file_storage = _FakeFileStorage("Hello, Actus!".encode("utf-8"))
    runner._user_id = "user-a"

    attachment = File(
        id="file-1",
        filename="demo.txt",
        filepath="/home/ubuntu/upload/demo.txt",
        extension=".txt",
        mime_type="text/plain",
        size=12,
    )

    envelope = asyncio.run(
        runner._build_message_input_envelope(
            message="看下文件内容",
            attachments=[attachment],
        )
    )

    assert envelope.parts[0].type == "text"
    text_file_part = envelope.parts[1]
    assert text_file_part.type == "file_ref"
    assert text_file_part.filepath == "/home/ubuntu/upload/demo.txt"
    assert text_file_part.text_content == "Hello, Actus!"


def test_read_text_attachment_content_should_read_with_byte_limit() -> None:
    payload = b"a" * (AgentTaskRunner._MAX_INLINE_TEXT_ATTACHMENT_BYTES + 4096)
    storage = _LargeFileStorage(payload)
    runner = object.__new__(AgentTaskRunner)
    runner._file_storage = storage
    runner._user_id = "user-a"
    attachment = File(
        id="file-big",
        filename="large.txt",
        filepath="/home/ubuntu/upload/large.txt",
        extension=".txt",
        mime_type="text/plain",
    )

    content = asyncio.run(runner._read_text_attachment_content(attachment))

    assert content is not None
    assert "内容已截断" in content
    assert storage.last_stream is not None
    assert storage.last_stream.total_read_bytes <= AgentTaskRunner._MAX_INLINE_TEXT_ATTACHMENT_BYTES + 1
