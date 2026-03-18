import asyncio
from typing import Optional

import pytest

from app.application.errors import NotFoundError, error_keys
from app.application.service.file_service import FileService


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
    async def upload_file(self, upload_file, user_id=None):
        raise AssertionError("upload_file should not be called in this test")

    async def download_file(self, file_id: str, user_id=None):
        raise AssertionError("download_file should not be called in this test")


def test_get_file_info_should_raise_not_found_with_error_key() -> None:
    service = FileService(
        uow_factory=lambda: _FakeUoW(),
        file_storage=_FakeFileStorage(),
    )

    with pytest.raises(NotFoundError) as exc:
        asyncio.run(service.get_file_info("user-1", "file-1"))

    assert exc.value.error_key == error_keys.FILE_NOT_FOUND
    assert exc.value.error_params == {"file_id": "file-1"}
