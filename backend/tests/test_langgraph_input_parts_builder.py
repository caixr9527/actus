import asyncio
import base64
import io

from app.domain.models import File, Message
from app.infrastructure.runtime.langgraph_run_engine import LangGraphRunEngine


class _FakeFileStorage:
    def __init__(self, payload_by_id: dict[str, bytes]) -> None:
        self._payload_by_id = payload_by_id

    async def download_file(self, file_id: str, user_id: str | None = None):
        payload = self._payload_by_id.get(file_id, b"")
        return io.BytesIO(payload), File(id=file_id, filename=f"{file_id}.bin")


class _SessionRepo:
    def __init__(self, file_by_path: dict[str, File]) -> None:
        self._file_by_path = file_by_path

    async def get_file_by_path(self, session_id: str, filepath: str):
        return self._file_by_path.get(filepath)


class _FakeUoW:
    def __init__(self, file_by_path: dict[str, File]) -> None:
        self.session = _SessionRepo(file_by_path=file_by_path)


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
        return stream, File(id=file_id, filename="large.bin")


def test_build_input_parts_should_use_suffix_and_default_file_type() -> None:
    image_path = "/home/ubuntu/upload/photo.png"
    unknown_path = "/home/ubuntu/upload/readme"
    file_by_path = {
        image_path: File(id="file-image", filepath=image_path, mime_type="image/png"),
        unknown_path: File(id="file-unknown", filepath=unknown_path, mime_type=""),
    }
    storage = _FakeFileStorage(
        payload_by_id={
            "file-image": b"fake-image",
            "file-unknown": b"fake-blob",
        }
    )
    engine = LangGraphRunEngine(
        session_id="session-a",
        llm=object(),
        file_storage=storage,
        user_id="user-a",
    )

    parts = asyncio.run(
        engine._build_input_parts(
            message=Message(message="请处理附件", attachments=[image_path, unknown_path]),
            uow=_FakeUoW(file_by_path=file_by_path),
        )
    )

    assert parts[0] == {"type": "text", "text": "请处理附件"}

    assert parts[1]["type"] == "image"
    assert parts[1]["filepath"] == image_path
    assert parts[1]["mime_type"] == "image/png"
    assert parts[1]["base64"] == base64.b64encode(b"fake-image").decode("utf-8")

    assert parts[2]["type"] == "file"
    assert parts[2]["filepath"] == unknown_path
    assert parts[2]["mime_type"] == "application/octet-stream"
    assert parts[2]["base64"] == base64.b64encode(b"fake-blob").decode("utf-8")


def test_read_attachment_base64_should_read_with_byte_limit() -> None:
    payload = b"a" * (LangGraphRunEngine._MAX_INLINE_BINARY_ATTACHMENT_BYTES + 4096)
    storage = _LargeFileStorage(payload)
    engine = LangGraphRunEngine(
        session_id="session-a",
        llm=object(),
        file_storage=storage,
        user_id="user-a",
    )

    content = asyncio.run(engine._read_attachment_base64(File(id="file-big", filename="large.bin")))

    assert content is None
    assert storage.last_stream is not None
    assert storage.last_stream.total_read_bytes <= LangGraphRunEngine._MAX_INLINE_BINARY_ATTACHMENT_BYTES + 1
