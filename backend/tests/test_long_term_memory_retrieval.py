import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy import UniqueConstraint
from pydantic import ValidationError

from app.domain.models import LongTermMemory, LongTermMemorySearchMode, LongTermMemorySearchQuery
from app.domain.models.long_term_memory import LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS
from app.domain.services.runtime.contracts.data_access_contract import (
    DataTrustLevel,
    PrivacyLevel,
    RetentionPolicyKind,
)
from app.infrastructure.repositories.db_long_term_memory_repository import DBLongTermMemoryRepository
from app.infrastructure.models.long_term_memory import LongTermMemoryModel
from app.infrastructure.runtime.langgraph.memory.long_term_memory_repository import LangGraphLongTermMemoryRepository
from core.config import Settings


def _compile_statement(statement) -> str:
    return str(statement.compile(dialect=postgresql.dialect()))


def _assert_memory_search_filters_user_id(compiled_sql: str) -> None:
    assert "long_term_memories.user_id =" in compiled_sql
    assert "long_term_memories.user_id IS NULL" not in compiled_sql


def test_db_long_term_memory_repository_should_build_hybrid_search_statement() -> None:
    persisted_memory = LongTermMemory(
        user_id="user-1",
        tenant_id="user-1",
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
                user_id="user-1",
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
    compiled_sql = _compile_statement(statement)
    _assert_memory_search_filters_user_id(compiled_sql)
    assert "websearch_to_tsquery" in compiled_sql
    assert "<=>" in compiled_sql
    assert "long_term_memories" in compiled_sql
    assert result[0].id == "mem-1"


def test_db_long_term_memory_repository_should_build_recent_search_statement_without_text_match() -> None:
    persisted_memory = LongTermMemory(
        user_id="user-1",
        tenant_id="user-1",
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
                user_id="user-1",
                namespace_prefixes=["user/user-1/"],
                memory_types=["profile"],
                mode=LongTermMemorySearchMode.RECENT,
                limit=5,
            )
        )
    )

    statement = db_session.execute.call_args.args[0]
    compiled_sql = _compile_statement(statement)
    _assert_memory_search_filters_user_id(compiled_sql)
    assert "websearch_to_tsquery" not in compiled_sql
    assert "<=>" not in compiled_sql
    assert "long_term_memories" in compiled_sql
    assert result[0].id == "mem-1"


def test_db_long_term_memory_repository_should_filter_semantic_search_to_embedded_candidates() -> None:
    persisted_memory = LongTermMemory(
        user_id="user-1",
        tenant_id="user-1",
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
                user_id="user-1",
                namespace_prefixes=["user/user-1/"],
                query_embedding=[0.1] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS,
                memory_types=["fact"],
                mode=LongTermMemorySearchMode.SEMANTIC,
                limit=5,
            )
        )
    )

    statement = db_session.execute.call_args.args[0]
    compiled_sql = _compile_statement(statement)
    _assert_memory_search_filters_user_id(compiled_sql)
    assert "long_term_memories.embedding IS NOT NULL" in compiled_sql
    assert "<=>" in compiled_sql
    assert result[0].id == "mem-1"


def test_db_long_term_memory_repository_should_not_recall_orphan_memory() -> None:
    selected_result = SimpleNamespace(
        scalars=lambda: SimpleNamespace(all=lambda: [])
    )
    db_session = SimpleNamespace(execute=AsyncMock(return_value=selected_result))
    repository = DBLongTermMemoryRepository(
        db_session=db_session,
        embedding_service=SimpleNamespace(embed_texts=AsyncMock()),
    )

    result = asyncio.run(
        repository.search(
            LongTermMemorySearchQuery(
                user_id="user-1",
                namespace_prefixes=["user/user-1/"],
                memory_types=["profile"],
                mode=LongTermMemorySearchMode.RECENT,
                limit=5,
            )
        )
    )

    statement = db_session.execute.call_args.args[0]
    compiled_sql = _compile_statement(statement)
    _assert_memory_search_filters_user_id(compiled_sql)
    assert result == []


def test_long_term_memory_search_query_should_reject_semantic_query_without_text_and_embedding() -> None:
    with pytest.raises(ValidationError):
        LongTermMemorySearchQuery(
            user_id="user-1",
            namespace_prefixes=["user/user-1/"],
            mode=LongTermMemorySearchMode.SEMANTIC,
            limit=5,
        )


def test_long_term_memory_search_query_should_reject_hybrid_query_without_text() -> None:
    with pytest.raises(ValidationError):
        LongTermMemorySearchQuery(
            user_id="user-1",
            namespace_prefixes=["user/user-1/"],
            mode=LongTermMemorySearchMode.HYBRID,
            limit=5,
        )


def test_long_term_memory_search_query_should_require_user_id() -> None:
    with pytest.raises(ValidationError):
        LongTermMemorySearchQuery(
            namespace_prefixes=["user/user-1/"],
            memory_types=["profile"],
            mode=LongTermMemorySearchMode.RECENT,
            limit=5,
        )


def test_db_long_term_memory_repository_should_inject_embedding_for_hybrid_query() -> None:
    persisted_memory = LongTermMemory(
        user_id="user-1",
        tenant_id="user-1",
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
                user_id="user-1",
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


def test_pr2_migration_should_backfill_memory_user_id_from_user_namespace() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "0f1e2d3c4b5a_p0_4_pr2_owner_fields_and_filters.py"
    )
    migration_source = migration_path.read_text(encoding="utf-8")

    assert "UPDATE long_term_memories" in migration_source
    assert "namespace LIKE 'user/%'" in migration_source
    assert "split_part(namespace, '/', 2)" in migration_source
    assert "AND user_id IS NULL" in migration_source


def test_pr4_migration_should_scope_memory_dedupe_constraint_by_user_id() -> None:
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "d7cfc3374fa2_.py"
    )
    migration_source = migration_path.read_text(encoding="utf-8")

    assert "op.drop_constraint(op.f('uq_long_term_memories_namespace_dedupe_key')" in migration_source
    assert "op.create_unique_constraint('uq_long_term_memories_user_namespace_dedupe_key'" in migration_source
    assert "['user_id', 'namespace', 'dedupe_key']" in migration_source


def test_long_term_memory_model_should_scope_dedupe_constraint_by_user_id() -> None:
    constraints = [
        constraint
        for constraint in LongTermMemoryModel.__table__.constraints
        if isinstance(constraint, UniqueConstraint)
    ]
    target = next(
        constraint
        for constraint in constraints
        if constraint.name == "uq_long_term_memories_user_namespace_dedupe_key"
    )

    assert [column.name for column in target.columns] == ["user_id", "namespace", "dedupe_key"]


def test_db_long_term_memory_repository_should_reject_invalid_query_embedding_dimensions() -> None:
    repository = DBLongTermMemoryRepository(
        db_session=SimpleNamespace(execute=AsyncMock()),
        embedding_service=SimpleNamespace(embed_texts=AsyncMock()),
    )

    with pytest.raises(ValueError, match="向量维度必须为 1536"):
        asyncio.run(
            repository.search(
                LongTermMemorySearchQuery(
                    user_id="user-1",
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
        user_id="user-1",
        namespace_prefixes=["user/user-1/"],
        query_text="中文 偏好",
        memory_types=["profile"],
        mode=LongTermMemorySearchMode.KEYWORD,
        limit=3,
    )
    expected = [
        LongTermMemory(
            user_id="user-1",
            tenant_id="user-1",
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
                user_id="user-1",
                tenant_id="user-1",
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


def test_db_long_term_memory_repository_should_persist_memory_governance_fields() -> None:
    captured_records = []
    db_session = SimpleNamespace(
        execute=AsyncMock(return_value=SimpleNamespace(scalar_one_or_none=lambda: None)),
        add=lambda record: captured_records.append(record),
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
                id="mem-1",
                namespace="user/user-1/fact",
                memory_type="fact",
                summary="用户偏好中文",
                content={"language": "zh"},
                content_text="用户偏好中文",
                embedding=[0.3] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS,
                source={"kind": "memory_consolidation"},
                trust_level=DataTrustLevel.SYSTEM_GENERATED,
                privacy_level=PrivacyLevel.SENSITIVE,
                retention_policy=RetentionPolicyKind.USER_MEMORY,
            )
        )
    )

    assert len(captured_records) == 1
    record = captured_records[0]
    assert record.source == {"kind": "memory_consolidation"}
    assert record.trust_level == DataTrustLevel.SYSTEM_GENERATED.value
    assert record.privacy_level == PrivacyLevel.SENSITIVE.value
    assert record.retention_policy == RetentionPolicyKind.USER_MEMORY.value
    assert persisted.source == {"kind": "memory_consolidation"}
    assert persisted.trust_level == DataTrustLevel.SYSTEM_GENERATED
    assert persisted.privacy_level == PrivacyLevel.SENSITIVE
    assert persisted.retention_policy == RetentionPolicyKind.USER_MEMORY


def test_db_long_term_memory_repository_should_reject_empty_user_id_on_upsert() -> None:
    repository = DBLongTermMemoryRepository(
        db_session=SimpleNamespace(execute=AsyncMock()),
        embedding_service=SimpleNamespace(embed_texts=AsyncMock()),
    )

    with pytest.raises(ValueError, match="长期记忆写入必须提供 user_id"):
        asyncio.run(
            repository.upsert(
                LongTermMemory.model_construct(
                    user_id="",
                    namespace="user/user-1/fact",
                    memory_type="fact",
                    summary="用户偏好中文",
                    content={"language": "zh"},
                )
            )
        )


def test_db_long_term_memory_repository_should_scope_upsert_lookup_by_user_id() -> None:
    existing_memory = LongTermMemory(
        user_id="user-1",
        tenant_id="user-1",
        id="mem-existing",
        namespace="shared/profile",
        memory_type="profile",
        summary="旧偏好",
        content={"language": "zh"},
        content_text="旧偏好",
        embedding=[0.4] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS,
        dedupe_key="shared-dedupe",
    )
    existing_record = LongTermMemoryModel.from_domain(existing_memory)
    execute_results = [
        SimpleNamespace(scalar_one_or_none=lambda: None),
        SimpleNamespace(scalar_one_or_none=lambda: None),
        SimpleNamespace(scalar_one_or_none=lambda: existing_record),
    ]
    added_records = []
    db_session = SimpleNamespace(
        execute=AsyncMock(side_effect=execute_results),
        add=lambda record: added_records.append(record),
        flush=AsyncMock(),
    )
    repository = DBLongTermMemoryRepository(
        db_session=db_session,
        embedding_service=SimpleNamespace(embed_texts=AsyncMock()),
    )

    user2_memory = LongTermMemory(
        user_id="user-2",
        tenant_id="user-2",
        id="mem-user-2",
        namespace="shared/profile",
        memory_type="profile",
        summary="用户2偏好",
        content={"language": "en"},
        content_text="用户2偏好",
        embedding=[0.5] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS,
        dedupe_key="shared-dedupe",
    )
    user1_update = LongTermMemory(
        user_id="user-1",
        tenant_id="user-1",
        id="",
        namespace="shared/profile",
        memory_type="profile",
        summary="新偏好",
        content={"language": "zh", "style": "concise"},
        content_text="新偏好",
        embedding=[0.6] * LONG_TERM_MEMORY_EMBEDDING_DIMENSIONS,
        dedupe_key="shared-dedupe",
    )

    persisted_user2 = asyncio.run(repository.upsert(user2_memory))
    persisted_user1 = asyncio.run(repository.upsert(user1_update))

    assert len(added_records) == 1
    assert added_records[0].user_id == "user-2"
    assert persisted_user2.user_id == "user-2"
    assert persisted_user1.id == "mem-existing"
    assert persisted_user1.user_id == "user-1"
    assert persisted_user1.summary == "新偏好"


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
