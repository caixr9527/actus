import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.domain.models import File
from app.infrastructure.repositories.db_session_repository import DBSessionRepository


def test_add_file_should_only_persist_session_files() -> None:
    record = SimpleNamespace(files=[])
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: record)),
        info={},
    )
    repository = DBSessionRepository(db_session=db_session)
    repository._workflow_run_repository = SimpleNamespace(
        append_file_snapshot=AsyncMock(),
    )

    asyncio.run(
        repository.add_file(
            session_id="session-1",
            file=File(id="file-1", filename="a.txt", filepath="/tmp/a.txt"),
        )
    )

    assert record.files == [
        {
            "id": "file-1",
            "filename": "a.txt",
            "filepath": "/tmp/a.txt",
            "key": "",
            "extension": "",
            "mime_type": "",
            "size": 0,
        }
    ]
    repository._workflow_run_repository.append_file_snapshot.assert_not_awaited()


def test_add_file_should_replace_existing_session_file_with_same_filepath() -> None:
    record = SimpleNamespace(
        files=[
            {
                "id": "file-old",
                "filename": "a.txt",
                "filepath": "/tmp/a.txt",
                "mime_type": "text/plain",
            },
            {
                "id": "file-keep",
                "filename": "b.txt",
                "filepath": "/tmp/b.txt",
                "mime_type": "text/plain",
            },
        ]
    )
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: record)),
        info={},
    )
    repository = DBSessionRepository(db_session=db_session)

    asyncio.run(
        repository.add_file(
            session_id="session-1",
            file=File(id="file-new", filename="a.txt", filepath="/tmp/a.txt", mime_type="text/markdown"),
        )
    )

    assert [item["id"] for item in record.files] == ["file-keep", "file-new"]
    assert record.files[-1]["mime_type"] == "text/markdown"


def test_get_file_by_path_should_only_read_session_files() -> None:
    record = SimpleNamespace(
        current_run_id="run-1",
        files=[
            {
                "id": "file-1",
                "filename": "a.txt",
                "filepath": "/tmp/a.txt",
            }
        ],
    )
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: record)),
        info={},
    )
    repository = DBSessionRepository(db_session=db_session)
    repository._workflow_run_repository = SimpleNamespace(
        get_by_id=AsyncMock(),
    )

    file = asyncio.run(repository.get_file_by_path(session_id="session-1", filepath="/tmp/a.txt"))

    assert file is not None
    assert file.id == "file-1"
    repository._workflow_run_repository.get_by_id.assert_not_awaited()


def test_get_file_by_path_should_prefer_latest_matching_file() -> None:
    record = SimpleNamespace(
        current_run_id="run-1",
        files=[
            {
                "id": "file-1",
                "filename": "a.txt",
                "filepath": "/tmp/a.txt",
            },
            {
                "id": "file-2",
                "filename": "a.txt",
                "filepath": "/tmp/a.txt",
            },
        ],
    )
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: record)),
        info={},
    )
    repository = DBSessionRepository(db_session=db_session)

    file = asyncio.run(repository.get_file_by_path(session_id="session-1", filepath="/tmp/a.txt"))

    assert file is not None
    assert file.id == "file-2"
