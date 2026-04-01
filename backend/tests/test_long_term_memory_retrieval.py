import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql
from pydantic import ValidationError

from app.domain.models import LongTermMemory, LongTermMemorySearchMode, LongTermMemorySearchQuery
from app.domain.models.long_term_memory import LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS
from app.infrastructure.repositories.db_long_term_memory_repository import DBLongTermMemoryRepository
from app.infrastructure.runtime.langgraph_long_term_memory_repository import LangGraphLongTermMemoryRepository
from core.config import Settings


def test_db_long_term_memory_repository_should_build_hybrid_search_statement() -> None:
    persisted_memory = LongTermMemory(
        id="mem-1",
        namespace="user/user-1/profile",
        memory_type="profile",
        summary="用户偏好中文",
        content={"language": "zh"},
    )
    selected_result = SimpleNamespace(
        scalars=lambda: SimpleNamespace(
            all=lambda: [SimpleNamespace(to_domain=lambda: persisted_memory, last_accessed_at=None)]
        )
    )
    db_session = SimpleNamespace(execute=AsyncMock(return_value=selected_result))
    repository = DBLongTermMemoryRepository(
        db_session=db_session,
        embedding_service=SimpleNamespace(embed_texts=AsyncMock(return_value=[[0.1] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS])),
    )

    result = asyncio.run(
        repository.search(
            LongTermMemorySearchQuery(
                namespace_prefixes=["user/user-1/"],
                query_text="中文 偏好",
                query_embedding=[0.1] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS,
                memory_types=["profile"],
                mode=LongTermMemorySearchMode.HYBRID,
                limit=5,
            )
        )
    )

    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "websearch_to_tsquery" in compiled_sql
    assert "<=>" in compiled_sql
    assert "long_term_memories" in compiled_sql
    assert result[0].id == "mem-1"


def test_db_long_term_memory_repository_should_build_recent_search_statement_without_text_match() -> None:
    persisted_memory = LongTermMemory(
        id="mem-1",
        namespace="user/user-1/profile",
        memory_type="profile",
        summary="用户偏好中文",
        content={"language": "zh"},
    )
    selected_result = SimpleNamespace(
        scalars=lambda: SimpleNamespace(
            all=lambda: [SimpleNamespace(to_domain=lambda: persisted_memory, last_accessed_at=None)]
        )
    )
    db_session = SimpleNamespace(execute=AsyncMock(return_value=selected_result))
    repository = DBLongTermMemoryRepository(
        db_session=db_session,
        embedding_service=SimpleNamespace(embed_texts=AsyncMock()),
    )

    result = asyncio.run(
        repository.search(
            LongTermMemorySearchQuery(
                namespace_prefixes=["user/user-1/"],
                memory_types=["profile"],
                mode=LongTermMemorySearchMode.RECENT,
                limit=5,
            )
        )
    )

    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "websearch_to_tsquery" not in compiled_sql
    assert "<=>" not in compiled_sql
    assert "long_term_memories" in compiled_sql
    assert result[0].id == "mem-1"


def test_db_long_term_memory_repository_should_filter_semantic_search_to_embedded_candidates() -> None:
    persisted_memory = LongTermMemory(
        id="mem-1",
        namespace="user/user-1/fact",
        memory_type="fact",
        summary="用户偏好中文",
        content={"language": "zh"},
    )
    selected_result = SimpleNamespace(
        scalars=lambda: SimpleNamespace(
            all=lambda: [SimpleNamespace(to_domain=lambda: persisted_memory, last_accessed_at=None)]
        )
    )
    db_session = SimpleNamespace(execute=AsyncMock(return_value=selected_result))
    repository = DBLongTermMemoryRepository(
        db_session=db_session,
        embedding_service=SimpleNamespace(embed_texts=AsyncMock(return_value=[[0.1] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS])),
    )

    result = asyncio.run(
        repository.search(
                LongTermMemorySearchQuery(
                    namespace_prefixes=["user/user-1/"],
                    query_embedding=[0.1] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS,
                    memory_types=["fact"],
                    mode=LongTermMemorySearchMode.SEMANTIC,
                    limit=5,
            )
        )
    )

    statement = db_session.execute.call_args.args[0]
    compiled_sql = str(statement.compile(dialect=postgresql.dialect()))
    assert "long_term_memories.embedding IS NOT NULL" in compiled_sql
    assert "<=>" in compiled_sql
    assert result[0].id == "mem-1"


def test_long_term_memory_search_query_should_reject_semantic_query_without_text_and_embedding() -> None:
    with pytest.raises(ValidationError):
        LongTermMemorySearchQuery(
            namespace_prefixes=["user/user-1/"],
            mode=LongTermMemorySearchMode.SEMANTIC,
            limit=5,
        )


def test_long_term_memory_search_query_should_reject_hybrid_query_without_text() -> None:
    with pytest.raises(ValidationError):
        LongTermMemorySearchQuery(
            namespace_prefixes=["user/user-1/"],
            mode=LongTermMemorySearchMode.HYBRID,
            limit=5,
        )


def test_db_long_term_memory_repository_should_inject_embedding_for_hybrid_query() -> None:
    persisted_memory = LongTermMemory(
        id="mem-1",
        namespace="user/user-1/fact",
        memory_type="fact",
        summary="用户偏好中文",
        content={"language": "zh"},
    )
    selected_result = SimpleNamespace(
        scalars=lambda: SimpleNamespace(
            all=lambda: [SimpleNamespace(to_domain=lambda: persisted_memory, last_accessed_at=None)]
        )
    )
    embedding_service = SimpleNamespace(
        embed_texts=AsyncMock(return_value=[[0.1] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS])
    )
    db_session = SimpleNamespace(execute=AsyncMock(return_value=selected_result))
    repository = DBLongTermMemoryRepository(
        db_session=db_session,
        embedding_service=embedding_service,
    )

    result = asyncio.run(
        repository.search(
            LongTermMemorySearchQuery(
                namespace_prefixes=["user/user-1/"],
                query_text="中文 偏好",
                memory_types=["fact"],
                mode=LongTermMemorySearchMode.HYBRID,
                limit=3,
            )
        )
    )

    embedding_service.embed_texts.assert_awaited_once_with(["中文 偏好"])
    assert result[0].id == "mem-1"


def test_db_long_term_memory_repository_should_reject_invalid_query_embedding_dimensions() -> None:
    repository = DBLongTermMemoryRepository(
        db_session=SimpleNamespace(execute=AsyncMock()),
        embedding_service=SimpleNamespace(embed_texts=AsyncMock()),
    )

    with pytest.raises(ValueError, match="向量维度必须为 1536"):
        asyncio.run(
            repository.search(
                LongTermMemorySearchQuery(
                    namespace_prefixes=["user/user-1/"],
                    query_text="中文 偏好",
                    query_embedding=[0.1, 0.2, 0.3],
                    memory_types=["fact"],
                    mode=LongTermMemorySearchMode.HYBRID,
                    limit=3,
                )
            )
        )


def test_langgraph_long_term_memory_repository_should_forward_structured_query() -> None:
    query = LongTermMemorySearchQuery(
        namespace_prefixes=["user/user-1/"],
        query_text="中文 偏好",
        memory_types=["profile"],
        mode=LongTermMemorySearchMode.KEYWORD,
        limit=3,
    )
    expected = [
        LongTermMemory(
            id="mem-1",
            namespace="user/user-1/profile",
            memory_type="profile",
            summary="用户偏好中文",
            content={"language": "zh"},
        )
    ]
    uow = SimpleNamespace(
        long_term_memory=SimpleNamespace(search=AsyncMock(return_value=expected), upsert=AsyncMock())
    )

    class _FakeUoW:
        async def __aenter__(self):
            return uow

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            return None

    repository = LangGraphLongTermMemoryRepository(uow_factory=lambda: _FakeUoW())

    result = asyncio.run(repository.search(query))

    uow.long_term_memory.search.assert_awaited_once_with(query)
    assert result == expected


def test_db_long_term_memory_repository_should_inject_memory_embedding_on_upsert() -> None:
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: None)),
        add=lambda record: None,
        flush=AsyncMock(),
    )
    embedding_service = SimpleNamespace(
        embed_texts=AsyncMock(return_value=[[0.2] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS])
    )
    repository = DBLongTermMemoryRepository(
        db_session=db_session,
        embedding_service=embedding_service,
    )

    persisted = asyncio.run(
        repository.upsert(
            LongTermMemory(
                id="mem-1",
                namespace="user/user-1/fact",
                memory_type="fact",
                summary="用户偏好中文",
                content={"language": "zh"},
            )
        )
    )

    embedding_service.embed_texts.assert_awaited_once()
    assert persisted.content_text == "用户偏好中文\nlanguage: zh"
    assert persisted.embedding == [0.2] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS


def test_settings_should_reject_embedding_dimensions_mismatch() -> None:
    with pytest.raises(ValueError, match="EMBEDDING_DIMENSIONS 必须为 1536"):
        Settings(
            embedding_dimensions=1024,
        )


def test_settings_should_normalize_empty_embedding_dimensions_to_none() -> None:
    settings = Settings(
        embedding_dimensions="",
    )

    assert settings.embedding_dimensions is None


def test_settings_should_treat_empty_embedding_dimensions_as_none() -> None:
    settings = Settings(
        embedding_dimensions="",
    )

    assert settings.embedding_dimensions is None
