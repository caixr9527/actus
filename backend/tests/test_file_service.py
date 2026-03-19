import asyncio
import io
from typing import Optional

import pytest
from fastapi import UploadFile

from app.application.errors import NotFoundError, error_keys
from app.application.service.file_service import FileService
from app.domain.models import File


class _FakeFileRepository:
    async def get_by_id(self, file_id: str) -> Optional[object]:
        return None

    async def get_by_id_and_user_id(self, file_id: str, user_id: str) -> Optional[object]:
        return None


class _FakeUoW:
    def __init__(self) -> None:
        self.file = _FakeFileRepository()

    async def __aenter__(self) -> "_FakeUoW":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _FakeFileStorage:
    def __init__(self) -> None:
        self.upload_user_ids: list[str | None] = []
        self.upload_payloads: list[object] = []

    async def upload_file(self, upload_file, user_id=None):
        self.upload_user_ids.append(user_id)
        self.upload_payloads.append(upload_file)
        return File(id="file-1", filename="a.txt")

    async def download_file(self, file_id: str, user_id=None):
        raise AssertionError("download_file should not be called in this test")

    def get_file_url(self, file: File) -> str:
        return f"https://example.com/{file.id}"


def test_get_file_info_should_raise_not_found_with_error_key() -> None:
    service = FileService(
        uow_factory=lambda: _FakeUoW(),
        file_storage=_FakeFileStorage(),
    )

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.get_file_info("user-1", "file-1"))

    assert exc.value.error_key == error_keys.FILE_NOT_FOUND
    assert exc.value.error_params == {"file_id": "file-1"}


def test_upload_file_should_convert_fastapi_file_to_domain_payload() -> None:
    storage = _FakeFileStorage()
    service = FileService(
        uow_factory=lambda: _FakeUoW(),
        file_storage=storage,
    )

    upload = UploadFile(filename="a.txt", file=io.BytesIO(b"hello"))
    asyncio.run(service.upload_file(user_id="user-1", upload_file=upload))

    assert storage.upload_user_ids == ["user-1"]
    assert len(storage.upload_payloads) == 1
    payload = storage.upload_payloads[0]
    assert getattr(payload, "filename") == "a.txt"
    assert getattr(payload, "size") == 0
    assert getattr(payload, "file").read() == b"hello"
