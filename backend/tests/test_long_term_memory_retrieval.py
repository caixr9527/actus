import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql
from pydantic import ValidationError

from app.domain.models import LongTermMemory, LongTermMemorySearchMode, LongTermMemorySearchQuery
from app.infrastructure.repositories.db_long_term_memory_repository import DBLongTermMemoryRepository
from app.infrastructure.runtime.langgraph_long_term_memory_repository import LangGraphLongTermMemoryRepository


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
    repository = DBLongTermMemoryRepository(db_session=db_session)

    result = asyncio.run(
        repository.search(
            LongTermMemorySearchQuery(
                namespace_prefixes=["user/user-1/"],
                query_text="中文 偏好",
                query_embedding=[0.1, 0.2, 0.3],
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
    repository = DBLongTermMemoryRepository(db_session=db_session)

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
    repository = DBLongTermMemoryRepository(db_session=db_session)

    result = asyncio.run(
        repository.search(
            LongTermMemorySearchQuery(
                namespace_prefixes=["user/user-1/"],
                query_embedding=[0.1, 0.2, 0.3],
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


def test_long_term_memory_search_query_should_reject_semantic_query_without_embedding() -> None:
    with pytest.raises(ValidationError):
        LongTermMemorySearchQuery(
            namespace_prefixes=["user/user-1/"],
            mode=LongTermMemorySearchMode.SEMANTIC,
            limit=5,
        )


def test_long_term_memory_search_query_should_reject_hybrid_query_without_embedding() -> None:
    with pytest.raises(ValidationError):
        LongTermMemorySearchQuery(
            namespace_prefixes=["user/user-1/"],
            query_text="中文 偏好",
            mode=LongTermMemorySearchMode.HYBRID,
            limit=5,
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
