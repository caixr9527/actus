import asyncio

from app.domain.models import Step
from app.domain.models import Workspace
from app.domain.services.runtime.contracts.sandbox_capability_profile_contract import (
    SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY,
)
from app.domain.services.workspace_runtime import WorkspaceEnvironmentSnapshot
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.execute_helpers import (
    _build_message as _build_execute_message,
    prepare_execute_step_input,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.planner_nodes import (
    _build_message as _build_planner_message,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.prompt_context_helpers import (
    _build_prompt_context_packet_async,
    extract_document_attachment_paths,
)


class _NoNativeContentLLM:
    multimodal = True

    async def format_multiplexed_message(self, input_parts):
        raise AssertionError("format_multiplexed_message should not be called")


def _document_part(**overrides):
    part = {
        "type": "document",
        "kind": "markdown",
        "source": {
            "file_id": "file-1",
            "user_id": "user-1",
            "session_id": "session-1",
            "workspace_id": "workspace-1",
            "run_id": "run-1",
            "sandbox_filepath": "/home/ubuntu/upload/file-1/notes.md",
            "filename": "notes.md",
            "mime_type": "text/markdown",
            "extension": ".md",
            "size": 12,
            "sha256": "a" * 64,
        },
        "parse_status": "parsed",
        "text_excerpt": "文档正文",
        "summary": "markdown document, 1 non-empty lines",
        "structured_preview": {},
        "reason_code": None,
        "is_truncated": False,
    }
    part.update(overrides)
    return part


class _WorkspaceRuntimeService:
    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def build_environment_snapshot(self):
        return WorkspaceEnvironmentSnapshot(workspace=self._workspace, artifacts=[])


def _context_service_with_sandbox_profile() -> RuntimeContextService:
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        environment_summary={
            SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: {
                "prompt_summary": {
                    "health_status": "available",
                    "cwd": "/workspace",
                    "available_tools": ["shell"],
                    "sandbox_profile_stale": False,
                }
            }
        },
    )
    return RuntimeContextService(
        workspace_runtime_service=_WorkspaceRuntimeService(workspace),
    )


def test_document_prompt_context_should_be_added_to_unified_context_packet() -> None:
    packet = asyncio.run(
        _build_prompt_context_packet_async(
            stage="planner",
            state={
                "user_message": "读取附件",
                "input_parts": [_document_part()],
                "message_window": [],
                "conversation_summary": "",
            },
            runtime_context_service=RuntimeContextService(),
        )
    )

    assert "document_context" in packet
    assert packet["document_context"]["document_count"] == 1
    document = packet["document_context"]["documents"][0]
    assert document["filename"] == "notes.md"
    assert document["sandbox_filepath"] == "/home/ubuntu/upload/file-1/notes.md"
    assert document["parse_status"] == "parsed"
    assert document["text_excerpt"] == "文档正文"
    assert "document_context" in packet["prompt_visible_fields"]


def test_document_prompt_context_should_preserve_unsupported_document_status_for_planner() -> None:
    unsupported_part = _document_part(
        kind="pdf",
        source={
            "file_id": "file-2",
            "user_id": "user-1",
            "session_id": "session-1",
            "workspace_id": "workspace-1",
            "run_id": "run-1",
            "sandbox_filepath": "/home/ubuntu/upload/file-2/paper.pdf",
            "filename": "paper.pdf",
            "mime_type": "application/pdf",
            "extension": ".pdf",
            "size": 120,
            "sha256": "b" * 64,
        },
        parse_status="unsupported",
        reason_code="unsupported_document_format",
        is_truncated=True,
        summary="pdf document parsing is not available in PR3",
        text_excerpt="",
    )

    packet = asyncio.run(
        _build_prompt_context_packet_async(
            stage="planner",
            state={
                "user_message": "读取 PDF",
                "input_parts": [unsupported_part],
                "message_window": [],
                "conversation_summary": "",
            },
            runtime_context_service=RuntimeContextService(),
        )
    )

    document = packet["document_context"]["documents"][0]
    assert document["filename"] == "paper.pdf"
    assert document["kind"] == "pdf"
    assert document["parse_status"] == "unsupported"
    assert document["reason_code"] == "unsupported_document_format"
    assert document["is_truncated"] is True


def test_document_prompt_context_should_be_added_to_execute_stage() -> None:
    packet = asyncio.run(
        _build_prompt_context_packet_async(
            stage="execute",
            state={
                "user_message": "读取附件",
                "input_parts": [_document_part()],
                "message_window": [],
                "conversation_summary": "",
            },
            runtime_context_service=RuntimeContextService(),
        )
    )

    assert "document_context" in packet
    assert packet["document_context"]["documents"][0]["text_excerpt"] == "文档正文"


def test_document_prompt_context_should_not_be_added_to_summary_or_replan_stage() -> None:
    base_state = {
        "user_message": "读取附件",
        "input_parts": [_document_part()],
        "message_window": [],
        "conversation_summary": "",
    }

    summary_packet = asyncio.run(
        _build_prompt_context_packet_async(
            stage="summary",
            state=base_state,
            runtime_context_service=RuntimeContextService(),
        )
    )
    replan_packet = asyncio.run(
        _build_prompt_context_packet_async(
            stage="replan",
            state=base_state,
            runtime_context_service=RuntimeContextService(),
        )
    )

    assert "document_context" not in summary_packet
    assert "document_context" not in summary_packet.get("prompt_visible_fields", [])
    assert "document_context" not in replan_packet
    assert "document_context" not in replan_packet.get("prompt_visible_fields", [])


def test_summary_prompt_context_should_not_include_sandbox_profile_by_default() -> None:
    packet = asyncio.run(
        _build_prompt_context_packet_async(
            stage="summary",
            state={"user_message": "总结结果"},
            runtime_context_service=_context_service_with_sandbox_profile(),
        )
    )

    assert "sandbox_capability_profile" not in packet.get("environment_digest", {})


def test_summary_prompt_context_should_include_sandbox_profile_when_explicitly_requested() -> None:
    packet = asyncio.run(
        _build_prompt_context_packet_async(
            stage="summary",
            state={"user_message": "总结结果"},
            runtime_context_service=_context_service_with_sandbox_profile(),
            include_sandbox_profile_for_summary=True,
        )
    )

    assert packet["environment_digest"]["sandbox_capability_profile"]["cwd"] == "/workspace"


def test_extract_document_attachment_paths_should_read_source_sandbox_filepath() -> None:
    paths = extract_document_attachment_paths(
        [
            _document_part(),
            {"type": "image", "sandbox_filepath": "/legacy/image.png"},
            _document_part(source={"sandbox_filepath": "/home/ubuntu/upload/file-2/other.md"}),
        ]
    )

    assert paths == [
        "/home/ubuntu/upload/file-1/notes.md",
        "/home/ubuntu/upload/file-2/other.md",
    ]


def test_planner_and_execute_build_message_should_only_emit_text_content() -> None:
    llm = _NoNativeContentLLM()
    input_parts = [_document_part()]

    planner_content = asyncio.run(_build_planner_message(llm, "planner prompt", input_parts))
    execute_content = asyncio.run(_build_execute_message(llm, "execute prompt", input_parts))

    assert planner_content == [{"type": "text", "text": "planner prompt"}]
    assert execute_content == [{"type": "text", "text": "execute prompt"}]


def test_execute_prepare_input_should_use_document_context_and_source_attachment_paths() -> None:
    prepared = asyncio.run(
        prepare_execute_step_input(
            state={
                "user_message": "读取附件",
                "input_parts": [_document_part()],
                "message_window": [],
                "conversation_summary": "",
                "working_memory": {},
            },
            step=Step(id="step-1", description="读取附件内容"),
            llm=_NoNativeContentLLM(),
            runtime_context_service=RuntimeContextService(),
            task_mode="file_processing",
            user_message="读取附件",
        )
    )

    assert prepared.attachments == ["/home/ubuntu/upload/file-1/notes.md"]
    assert prepared.user_content[0]["type"] == "text"
    prompt = prepared.user_content[0]["text"]
    assert "已知上下文" in prompt
    assert "document_context" in prompt
    assert "文档正文" in prompt
