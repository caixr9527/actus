import asyncio
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.application.errors import NotFoundError, error_keys
from app.application.service.agent_service import AgentService
from app.application.service.document_input_preflight_policy import DocumentInputPreflightPolicy
from app.application.service.document_input_service import (
    DocumentAttachmentSource,
    DocumentInputService,
    FileStorageDocumentAttachmentReader,
)
from app.application.service.runtime_access_control_service import AccessScopeResult
from app.domain.models import (
    ErrorEvent,
    ExecutionStatus,
    File,
    Message,
    MessageEvent,
    Plan,
    Session,
    SessionStatus,
    Step,
    StepEvent,
    StepOutcome,
    WorkflowRun,
    WorkflowRunStatus,
)
from app.domain.services.runtime.contracts.data_access_contract import DataAccessAction
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.infrastructure.runtime.langgraph.engine.run_engine import DocumentInputContractError, LangGraphRunEngine
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes import (
    guard_step_reuse_node,
    wait_for_human_node,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.execute_nodes import execute_step_node
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.execute_helpers import (
    _build_message as _build_execute_message,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.planner_nodes import (
    _build_message as _build_planner_message,
)


LEGACY_INPUT_TYPES = {"image", "audio", "video", "file", "file_ref"}
LEGACY_NATIVE_BLOCK_TYPES = {"image_url", "video", "video_url", "audio", "input_audio", "file", "file_ref"}
_TEST_RUNTIME_CONTEXT_SERVICE = RuntimeContextService()


class _FakeGraph:
    async def ainvoke(self, state, config=None):
        return {"emitted_events": []}


class _NoopCoordinator:
    def __init__(self) -> None:
        self.accept_user_message_calls = []
        self.mark_resume_requested_calls = []

    async def reconcile_current_run(self, session_id: str, *, reason: str):
        return SimpleNamespace(warnings=[], snapshot_after=None)

    async def accept_user_message(self, **kwargs) -> None:
        self.accept_user_message_calls.append(kwargs)

    async def mark_resume_requested(self, **kwargs) -> None:
        self.mark_resume_requested_calls.append(kwargs)


class _SessionRepo:
    def __init__(self) -> None:
        self.events = []
        self.unread_updates = []

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        return Session(id=session_id, user_id=user_id or "user-1", status=SessionStatus.COMPLETED)

    async def add_event(self, session_id: str, event):
        self.events.append(event)

    async def update_unread_message_count(self, session_id: str, count: int):
        self.unread_updates.append((session_id, count))


class _FileRepo:
    def __init__(self, file: File | None) -> None:
        self.file = file
        self.calls = []

    async def get_by_id_and_user_id(self, *, file_id: str, user_id: str):
        self.calls.append((file_id, user_id))
        if self.file is None:
            return None
        return self.file.model_copy(update={"id": file_id, "user_id": user_id})


class _UoW:
    def __init__(self, *, file_repo: _FileRepo, session_repo: _SessionRepo) -> None:
        self.file = file_repo
        self.session = session_repo

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _AccessControl:
    def __init__(self, *, reject_file: bool = False) -> None:
        self.reject_file = reject_file
        self.file_access_calls = []
        self.session_scope_calls = []

    async def resolve_session_scope(self, *, user_id: str, session_id: str):
        self.session_scope_calls.append((user_id, session_id))
        return AccessScopeResult(tenant_id=user_id, user_id=user_id, session_id=session_id)

    async def assert_file_access(self, *, user_id: str, file_id: str, action: DataAccessAction):
        self.file_access_calls.append((user_id, file_id, action))
        if self.reject_file:
            raise NotFoundError(
                msg=f"该文件[{file_id}]不存在",
                error_key=error_keys.FILE_NOT_FOUND,
                error_params={"file_id": file_id},
            )
        return AccessScopeResult(tenant_id=user_id, user_id=user_id)


class _GraphRuntime:
    def __init__(self) -> None:
        self.create_task_calls = []
        self.resume_task_calls = []

    async def get_task(self, session):
        return None

    async def create_task(self, **kwargs):
        self.create_task_calls.append(kwargs)
        raise AssertionError("preflight 失败时不应创建 graph task")

    async def resume_task(self, **kwargs):
        self.resume_task_calls.append(kwargs)
        raise AssertionError("preflight 失败时不应恢复 graph task")


class _WorkspaceManager:
    def __init__(self) -> None:
        self.resolve_current_run_id_calls = []

    async def resolve_current_run_id(self, *, session):
        self.resolve_current_run_id_calls.append(session.id)
        return None


def _build_agent_service_with_attachment(file: File | None, *, reject_file: bool = False):
    session_repo = _SessionRepo()
    file_repo = _FileRepo(file)
    graph_runtime = _GraphRuntime()
    coordinator = _NoopCoordinator()
    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW(file_repo=file_repo, session_repo=session_repo)
    service._access_control_service = _AccessControl(reject_file=reject_file)
    service._document_input_preflight_policy = DocumentInputPreflightPolicy()
    service._runtime_state_coordinator = coordinator
    service._graph_runtime = graph_runtime
    service._workspace_manager = _WorkspaceManager()
    service._task_cls = object
    return service, session_repo, graph_runtime, coordinator


async def _collect_chat_events(service: AgentService, *, attachments: list[str]):
    events = []
    async for event in service.chat(
            session_id="session-1",
            user_id="user-1",
            message="读取附件",
            attachments=attachments,
    ):
        events.append(event)
    return events


def _assert_no_runtime_side_effects(
        *,
        session_repo: _SessionRepo,
        graph_runtime: _GraphRuntime,
        coordinator: _NoopCoordinator,
) -> None:
    assert graph_runtime.create_task_calls == []
    assert graph_runtime.resume_task_calls == []
    assert coordinator.accept_user_message_calls == []
    assert session_repo.unread_updates == []
    assert not any(isinstance(event, MessageEvent) for event in session_repo.events)


def _scope(user_id: str = "user-1") -> AccessScopeResult:
    return AccessScopeResult(
        tenant_id=user_id,
        user_id=user_id,
        session_id="session-1",
        workspace_id="workspace-1",
        run_id="run-1",
    )


class _Reader:
    def __init__(self, data: bytes, *, raise_on_open: bool = False) -> None:
        self._data = data
        self._raise_on_open = raise_on_open
        self.is_truncated = False

    async def open_stream(self):
        if self._raise_on_open:
            raise RuntimeError("storage read failed")
        return io.BytesIO(self._data)

    async def compute_sha256(self) -> str:
        return "a" * 64


class _ChunkedStream:
    def __init__(self, data: bytes) -> None:
        self._data = data
        self._offset = 0
        self.read_sizes = []

    def read(self, size: int = -1):
        self.read_sizes.append(size)
        if size < 0:
            size = len(self._data) - self._offset
        chunk = self._data[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk


class _FileStorage:
    def __init__(self, payloads_by_id: dict[str, bytes | _ChunkedStream]) -> None:
        self.payloads_by_id = dict(payloads_by_id)

    async def download_file(self, file_id: str, user_id: str | None = None):
        payload = self.payloads_by_id[file_id]
        stream = payload if isinstance(payload, _ChunkedStream) else io.BytesIO(payload)
        return stream, File(id=file_id)

    def get_file_url(self, file: File) -> str:
        return f"https://cdn.example.com/{file.id}"


class _InputPartsSessionRepo:
    def __init__(self, files_by_path: dict[str, File]) -> None:
        self.files_by_path = {path: file.model_copy(deep=True) for path, file in files_by_path.items()}

    async def get_file_by_path(self, session_id: str, filepath: str):
        file = self.files_by_path.get(filepath)
        return file.model_copy(deep=True) if file is not None else None


class _GraphStateUoW:
    def __init__(self, *, session_repo, workflow_run_repo=None) -> None:
        self.session = session_repo
        self.workflow_run = workflow_run_repo or SimpleNamespace(get_by_id=AsyncMock(return_value=None))
        self.workflow_run_summary = SimpleNamespace(list_by_session_id=AsyncMock(side_effect=[[], []]))
        self.session_context_snapshot = SimpleNamespace(get_by_session_id=AsyncMock(return_value=None))
        self.workspace = SimpleNamespace(
            get_by_id=AsyncMock(return_value=None),
            get_by_id_for_user=AsyncMock(return_value=None),
            get_by_session_id=AsyncMock(return_value=None),
        )

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return None


def _build_run_engine(**kwargs) -> LangGraphRunEngine:
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return LangGraphRunEngine(**kwargs)


def _stage_llms(llm=object()) -> dict[str, object]:
    return {
        "router": llm,
        "planner": llm,
        "executor": llm,
        "replan": llm,
        "summary": llm,
    }


def _assert_no_legacy_input_parts(parts: list[dict]) -> None:
    assert parts
    assert all(part.get("type") == "document" for part in parts)
    assert not any(part.get("type") in LEGACY_INPUT_TYPES for part in parts)
    assert all("base64_payload" not in part for part in parts)
    assert all("file_url" not in part for part in parts)


def _assert_no_native_media_blocks(content: list[dict]) -> None:
    assert content
    assert all(part.get("type") == "text" for part in content)
    assert not any(part.get("type") in LEGACY_NATIVE_BLOCK_TYPES for part in content)


def test_media_preflight_rejection_should_not_create_message_event_task_or_graph_state() -> None:
    service, session_repo, graph_runtime, coordinator = _build_agent_service_with_attachment(
        File(
            id="file-image",
            user_id="user-1",
            filename="photo.png",
            extension=".png",
            mime_type="image/png",
            size=128,
        )
    )

    events = asyncio.run(_collect_chat_events(service, attachments=["file-image"]))

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ErrorEvent)
    assert event.error_key == error_keys.DOCUMENT_INPUT_UNSUPPORTED_MEDIA_IMAGE
    assert event.error_params["file_id"] == "file-image"
    assert event.error_params["reason_code"] == "unsupported_media_image"
    _assert_no_runtime_side_effects(
        session_repo=session_repo,
        graph_runtime=graph_runtime,
        coordinator=coordinator,
    )


def test_cross_user_file_should_not_enter_runtime() -> None:
    service, session_repo, graph_runtime, coordinator = _build_agent_service_with_attachment(
        None,
        reject_file=True,
    )

    events = asyncio.run(_collect_chat_events(service, attachments=["file-owned-by-other-user"]))

    assert len(events) == 1
    event = events[0]
    assert isinstance(event, ErrorEvent)
    assert event.error_key == error_keys.FILE_NOT_FOUND
    assert event.error_params == {"file_id": "file-owned-by-other-user"}
    _assert_no_runtime_side_effects(
        session_repo=session_repo,
        graph_runtime=graph_runtime,
        coordinator=coordinator,
    )


def test_graph_input_parts_should_use_document_as_only_attachment_shape() -> None:
    engine = _build_run_engine(
        session_id="session-1",
        stage_llms=_stage_llms(),
        user_id="user-1",
        file_storage=_FileStorage(
            {
                "file-1": b"# note\nbody",
                "file-2": b'{"ok": true}',
            }
        ),
    )
    uow = SimpleNamespace(
        session=_InputPartsSessionRepo(
            {
                "/home/ubuntu/upload/file-1/note.md": File(
                    id="file-1",
                    filename="note.md",
                    filepath="/home/ubuntu/upload/file-1/note.md",
                    mime_type="text/markdown",
                    extension=".md",
                    size=11,
                ),
                "/home/ubuntu/upload/file-2/data.json": File(
                    id="file-2",
                    filename="data.json",
                    filepath="/home/ubuntu/upload/file-2/data.json",
                    mime_type="application/json",
                    extension=".json",
                    size=12,
                ),
            }
        )
    )

    parts = asyncio.run(
        engine._build_input_parts(
            Message(
                message="读取附件",
                attachments=[
                    "/home/ubuntu/upload/file-1/note.md",
                    "/home/ubuntu/upload/file-2/data.json",
                ],
            ),
            uow=uow,
            scope=_scope(),
            request_id="request-1",
        )
    )

    _assert_no_legacy_input_parts(parts)
    assert [part["source"]["file_id"] for part in parts] == ["file-1", "file-2"]
    assert [part["parse_status"] for part in parts] == ["parsed", "parsed"]


def test_graph_input_state_should_fail_closed_for_document_contract_error_without_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.engine.run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: _FakeGraph(),
    )
    session_repo = SimpleNamespace(
        get_by_id=AsyncMock(
            return_value=Session(
                id="session-1",
                user_id="user-1",
                workspace_id="workspace-1",
                current_run_id="run-1",
            )
        ),
        get_file_by_path=AsyncMock(return_value=None),
    )
    workflow_run_repo = SimpleNamespace(
        get_by_id=AsyncMock(
            return_value=WorkflowRun(
                id="run-1",
                session_id="session-1",
                user_id="user-1",
                status=WorkflowRunStatus.RUNNING,
            )
        )
    )
    engine = _build_run_engine(
        session_id="session-1",
        stage_llms=_stage_llms(),
        user_id="user-1",
        file_storage=_FileStorage({"file-1": b"content"}),
        uow_factory=lambda: _GraphStateUoW(
            session_repo=session_repo,
            workflow_run_repo=workflow_run_repo,
        ),
        access_control_service=None,
    )
    engine._access_control_service = None

    with pytest.raises(DocumentInputContractError, match="scope"):
        asyncio.run(
            engine._build_graph_input_state(
                message=Message(
                    message="读取附件",
                    attachments=["/home/ubuntu/upload/file-1/notes.txt"],
                ),
                run_id="run-1",
                invoke_config={"configurable": {"thread_id": "thread-1"}},
            )
        )


def test_graph_input_state_should_keep_non_attachment_fallback_behavior(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.engine.run_engine.build_planner_react_langgraph_graph",
        lambda **kwargs: _FakeGraph(),
    )
    broken_uow = SimpleNamespace(
        __aenter__=AsyncMock(side_effect=RuntimeError("snapshot unavailable")),
        __aexit__=AsyncMock(return_value=None),
    )
    engine = _build_run_engine(
        session_id="session-1",
        stage_llms=_stage_llms(),
        user_id="user-1",
        uow_factory=lambda: broken_uow,
    )

    state = asyncio.run(
        engine._build_graph_input_state(
            message=Message(message="普通消息", attachments=[]),
            run_id="run-1",
            invoke_config={"configurable": {"thread_id": "thread-1"}},
        )
    )

    assert state["input_parts"] == []
    assert state["user_message"] == "普通消息"
    assert state["thread_id"] == "thread-1"


def _source(file: File, reader: _Reader | FileStorageDocumentAttachmentReader) -> DocumentAttachmentSource:
    return DocumentAttachmentSource(
        scope=_scope(),
        file=file,
        sandbox_filepath=f"/home/ubuntu/upload/{file.id}/{file.filename}",
        reader=reader,
    )


def test_parse_failures_should_not_be_disguised_as_parsed() -> None:
    service = DocumentInputService()
    large_stream = _ChunkedStream(b"abcdef")
    large_file = File(id="file-large", filename="large.txt", extension=".txt", mime_type="text/plain", size=6)
    cases = [
        _source(
            File(id="file-pdf", filename="paper.pdf", extension=".pdf", mime_type="application/pdf", size=4),
            _Reader(b"%PDF"),
        ),
        _source(
            large_file,
            FileStorageDocumentAttachmentReader(
                file_storage=_FileStorage({"file-large": large_stream}),
                file=large_file,
                user_id="user-1",
                max_bytes=3,
            ),
        ),
        _source(
            File(id="file-empty", filename="empty.txt", extension=".txt", mime_type="text/plain", size=0),
            _Reader(b""),
        ),
        _source(
            File(id="file-failed", filename="broken.txt", extension=".txt", mime_type="text/plain", size=5),
            _Reader(b"hello", raise_on_open=True),
        ),
    ]

    parts = asyncio.run(service.build_input_parts(scope=_scope(), attachments=cases))
    statuses_by_file = {part.source.file_id: part.parse_status.value for part in parts}
    reason_by_file = {part.source.file_id: part.reason_code for part in parts}

    assert statuses_by_file == {
        "file-pdf": "unsupported",
        "file-large": "too_large",
        "file-empty": "empty",
        "file-failed": "failed",
    }
    assert reason_by_file == {
        "file-pdf": "unsupported_document_format",
        "file-large": "too_large",
        "file-empty": "empty",
        "file-failed": "parse_failed",
    }
    assert "parsed" not in statuses_by_file.values()
    assert -1 not in large_stream.read_sizes


class _ForbiddenMultiplexLLM:
    async def format_multiplexed_message(self, text: str, input_parts: list[dict]):
        raise AssertionError("planner/execute 不应调用旧 multimodal 原生格式化路径")


def test_planner_and_execute_user_content_should_not_emit_native_media_blocks() -> None:
    legacy_input_parts = [
        {"type": "image", "url": "https://cdn.example.com/image.png"},
        {"type": "audio", "url": "https://cdn.example.com/audio.mp3"},
        {"type": "video", "url": "https://cdn.example.com/video.mp4"},
        {"type": "file", "url": "https://cdn.example.com/raw.bin"},
        {"type": "file_ref", "file_id": "file-1"},
        {"type": "document", "text_excerpt": "合法文档上下文只走 prompt"},
    ]

    planner_content = asyncio.run(
        _build_planner_message(_ForbiddenMultiplexLLM(), "planner prompt", legacy_input_parts)
    )
    execute_content = asyncio.run(
        _build_execute_message(_ForbiddenMultiplexLLM(), "execute prompt", legacy_input_parts)
    )

    _assert_no_native_media_blocks(planner_content)
    _assert_no_native_media_blocks(execute_content)
    assert planner_content == [{"type": "text", "text": "planner prompt"}]
    assert execute_content == [{"type": "text", "text": "execute prompt"}]


class _LeakyExecuteLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "步骤摘要",
                    "attachments": ["/tmp/step-output.md"],
                    "final_answer_text": "不能由 execute 写入的最终正文",
                    "final_message": "不能由 execute 写入的最终消息",
                    "selected_artifacts": ["/tmp/final.md"],
                },
                ensure_ascii=False,
            )
        }


async def _execute_step_node(state, llm, runtime_tools=None):
    return await execute_step_node(
        state,
        llm,
        runtime_tools=runtime_tools or [],
        runtime_context_service=_TEST_RUNTIME_CONTEXT_SERVICE,
    )


def _document_part() -> dict:
    return {
        "type": "document",
        "kind": "text",
        "source": {
            "file_id": "file-1",
            "user_id": "user-1",
            "session_id": "session-1",
            "workspace_id": "workspace-1",
            "run_id": "run-1",
            "sandbox_filepath": "/home/ubuntu/upload/file-1/notes.txt",
            "filename": "notes.txt",
            "mime_type": "text/plain",
            "extension": ".txt",
            "size": 12,
            "sha256": "a" * 64,
        },
        "parse_status": "parsed",
        "text_excerpt": "文档正文",
        "summary": "文本文件，约 1 行。",
        "structured_preview": {},
        "reason_code": None,
        "is_truncated": False,
    }


def test_execute_wait_and_reuse_should_not_write_final_outputs_from_document_input(monkeypatch) -> None:
    plan = Plan(
        title="文档输入纯执行测试",
        goal="验证文档输入不会绕过 summary 写最终输出",
        language="zh",
        steps=[
            Step(
                id="step-1",
                title="读取文档",
                description="读取文档并沉淀步骤事实",
                status=ExecutionStatus.PENDING,
            )
        ],
    )
    base_state = {
        "session_id": "session-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": "读取文档",
        "plan": plan,
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {},
        "execution_count": 0,
        "input_parts": [_document_part()],
        "selected_artifacts": ["/tmp/existing-final.md"],
        "step_states": [],
        "pending_interrupt": {},
        "retrieved_memories": [],
        "recent_run_briefs": [],
        "recent_attempt_briefs": [],
        "session_open_questions": [],
        "session_blockers": [],
        "historical_artifact_paths": [],
        "emitted_events": [],
        "current_step_id": None,
        "final_message": "已有轻量总结",
        "final_answer_text": "已有最终正文",
        "conversation_summary": "",
    }

    executed_state = asyncio.run(_execute_step_node(base_state, _LeakyExecuteLLM()))
    assert executed_state["last_executed_step"].outcome is not None
    assert executed_state["last_executed_step"].outcome.summary == "步骤摘要"
    assert executed_state["last_executed_step"].outcome.produced_artifacts == ["/tmp/step-output.md"]
    assert executed_state["final_message"] == "已有轻量总结"
    assert executed_state["final_answer_text"] == "已有最终正文"
    assert executed_state["selected_artifacts"] == ["/tmp/existing-final.md"]
    assert not any(isinstance(event, MessageEvent) and event.stage == "final" for event in executed_state["emitted_events"])

    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.interrupt",
        lambda payload: "继续",
    )
    waiting_plan = Plan(
        title="文档等待测试",
        goal="验证等待恢复不写最终输出",
        language="zh",
        steps=[
            Step(
                id="wait-step",
                title="等待确认",
                description="等待确认",
                status=ExecutionStatus.RUNNING,
            )
        ],
    )
    wait_state = {
        "plan": waiting_plan,
        "current_step_id": "wait-step",
        "pending_interrupt": {"kind": "input_text", "prompt": "请确认"},
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {},
        "execution_count": 0,
        "input_parts": [_document_part()],
        "final_message": "已有轻量总结",
        "final_answer_text": "已有最终正文",
        "selected_artifacts": ["/tmp/existing-final.md"],
    }
    resumed_state = asyncio.run(wait_for_human_node(wait_state))
    assert resumed_state["last_executed_step"].outcome is not None
    assert resumed_state["final_message"] == "已有轻量总结"
    assert resumed_state["final_answer_text"] == "已有最终正文"
    assert resumed_state["selected_artifacts"] == ["/tmp/existing-final.md"]
    assert not any(isinstance(event, MessageEvent) and event.stage == "final" for event in resumed_state["emitted_events"])

    completed_step = Step(
        id="step-a",
        title="读取文档",
        description="读取文档",
        objective_key="objective-document",
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(
            done=True,
            summary="历史步骤摘要",
            produced_artifacts=["/tmp/reused-candidate.md"],
        ),
    )
    duplicate_step = Step(
        id="step-b",
        title="再次读取文档",
        description="再次读取文档",
        objective_key="objective-document",
        status=ExecutionStatus.PENDING,
    )
    reuse_state = {
        "run_id": "run-1",
        "plan": Plan(
            title="文档复用测试",
            goal="验证复用不写最终输出",
            language="zh",
            steps=[completed_step, duplicate_step],
        ),
        "working_memory": {},
        "graph_metadata": {},
        "input_parts": [_document_part()],
        "selected_artifacts": ["/tmp/existing-final.md"],
        "emitted_events": [],
        "execution_count": 0,
        "final_message": "已有轻量总结",
        "final_answer_text": "已有最终正文",
    }
    reused_state = asyncio.run(guard_step_reuse_node(reuse_state))
    assert reused_state["plan"].steps[1].outcome is not None
    assert reused_state["plan"].steps[1].outcome.produced_artifacts == ["/tmp/reused-candidate.md"]
    assert reused_state["final_message"] == "已有轻量总结"
    assert reused_state["final_answer_text"] == "已有最终正文"
    assert reused_state["selected_artifacts"] == ["/tmp/existing-final.md"]
    assert all(isinstance(event, StepEvent) for event in reused_state["emitted_events"])
