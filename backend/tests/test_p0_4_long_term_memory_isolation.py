import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from sqlalchemy.dialects import postgresql

from app.domain.models import LongTermMemory
from app.domain.models.long_term_memory import LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS
from app.infrastructure.models.long_term_memory import LongTermMemoryModel
from app.infrastructure.repositories.db_long_term_memory_repository import DBLongTermMemoryRepository


def _compile_statement(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def test_long_term_memory_upsert_id_lookup_should_include_user_id() -> None:
    execute_results = [
        SimpleNamespace(scalar_one_or_none=lambda: None),
        SimpleNamespace(scalar_one_or_none=lambda: None),
    ]
    db_session = SimpleNamespace(
        execute=AsyncMock(side_effect=execute_results),
        add=lambda record: None,
        flush=AsyncMock(),
    )
    repository = DBLongTermMemoryRepository(
        db_session=db_session,
        embedding_service=SimpleNamespace(embed_texts=AsyncMock()),
    )

    asyncio.run(
        repository.upsert(
            LongTermMemory(
                user_id="user-1",
                tenant_id="user-1",
                id="mem-1",
                namespace="shared/profile",
                memory_type="profile",
                summary="用户偏好中文",
                content_text="用户偏好中文",
                embedding=[0.1] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS,
                dedupe_key="dedupe-1",
            )
        )
    )

    first_lookup = db_session.execute.await_args_list[0].args[0]
    compiled_sql = _compile_statement(first_lookup)
    assert "long_term_memories.id =" in compiled_sql
    assert "long_term_memories.user_id =" in compiled_sql


def test_long_term_memory_upsert_dedupe_lookup_should_include_user_id_namespace_and_key() -> None:
    execute_results = [
        SimpleNamespace(scalar_one_or_none=lambda: None),
    ]
    db_session = SimpleNamespace(
        execute=AsyncMock(side_effect=execute_results),
        add=lambda record: None,
        flush=AsyncMock(),
    )
    repository = DBLongTermMemoryRepository(
        db_session=db_session,
        embedding_service=SimpleNamespace(embed_texts=AsyncMock()),
    )

    asyncio.run(
        repository.upsert(
            LongTermMemory(
                user_id="user-1",
                tenant_id="user-1",
                id="",
                namespace="shared/profile",
                memory_type="profile",
                summary="用户偏好中文",
                content_text="用户偏好中文",
                embedding=[0.1] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS,
                dedupe_key="dedupe-1",
            )
        )
    )

    dedupe_lookup = db_session.execute.await_args_list[0].args[0]
    compiled_sql = _compile_statement(dedupe_lookup)
    assert "long_term_memories.user_id =" in compiled_sql
    assert "long_term_memories.namespace =" in compiled_sql
    assert "long_term_memories.dedupe_key =" in compiled_sql


def test_long_term_memory_upsert_should_not_update_cross_user_same_namespace_dedupe_record() -> None:
    foreign_memory = LongTermMemory(
        user_id="user-2",
        tenant_id="user-2",
        id="mem-foreign",
        namespace="shared/profile",
        memory_type="profile",
        summary="外部用户偏好",
        content_text="外部用户偏好",
        embedding=[0.2] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS,
        dedupe_key="dedupe-1",
    )
    foreign_record = LongTermMemoryModel.from_domain(foreign_memory)
    added_records: list[LongTermMemoryModel] = []
    db_session = SimpleNamespace(
        execute=AsyncMock(
            side_effect=[
                SimpleNamespace(scalar_one_or_none=lambda: None),
                SimpleNamespace(scalar_one_or_none=lambda: None),
            ]
        ),
        add=lambda record: added_records.append(record),
        flush=AsyncMock(),
    )
    repository = DBLongTermMemoryRepository(
        db_session=db_session,
        embedding_service=SimpleNamespace(embed_texts=AsyncMock()),
    )

    persisted = asyncio.run(
        repository.upsert(
            LongTermMemory(
                user_id="user-1",
                tenant_id="user-1",
                id="",
                namespace="shared/profile",
                memory_type="profile",
                summary="当前用户偏好",
                content_text="当前用户偏好",
                embedding=[0.3] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS,
                dedupe_key="dedupe-1",
            )
        )
    )

    assert foreign_record.summary == "外部用户偏好"
    assert len(added_records) == 1
    assert added_records[0].user_id == "user-1"
    assert persisted.user_id == "user-1"
    assert persisted.summary == "当前用户偏好"
