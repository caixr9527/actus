import asyncio

from app.domain.models import Step, ToolEventStatus
from app.domain.services.tools.base import BaseTool, tool
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.parsers import (
    normalize_supported_input_types,
    resolve_model_input_policy,
)
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.tools import (
    execute_step_with_prompt,
)


class _FakeLLM:
    def __init__(self, *, multimodal: bool, supported: list[str]) -> None:
        self._multimodal = multimodal
        self._supported = supported

    @property
    def multimodal(self) -> bool:
        return self._multimodal

    @property
    def supported(self) -> list[str]:
        return self._supported


def test_normalize_supported_input_types_should_always_include_text() -> None:
    assert normalize_supported_input_types(["image"]) == ["text", "image"]
    assert normalize_supported_input_types(["text", "image"]) == ["text", "image"]
    assert normalize_supported_input_types("audio") == ["text", "audio"]
    assert normalize_supported_input_types("invalid") == ["text"]


def test_resolve_model_input_policy_should_downgrade_text_file_when_model_not_multimodal() -> None:
    policy = resolve_model_input_policy(
        llm=_FakeLLM(multimodal=False, supported=["text"]),
        input_parts=[
            {"type": "text", "text": "请读取附件"},
            {
                "type": "file_ref",
                "filepath": "/home/ubuntu/upload/notes.md",
                "mime_type": "text/markdown",
                "extension": "md",
            },
            {
                "type": "image",
                "filepath": "/home/ubuntu/upload/image.png",
                "uri": "https://cdn.example.com/image.png",
                "mime_type": "image/png",
            },
        ],
    )

    assert policy["text_attachment_paths"] == ["/home/ubuntu/upload/notes.md"]
    assert policy["native_user_content_parts"] == []
    assert policy["unsupported_parts"] == [
        {
            "type": "image",
            "filepath": "/home/ubuntu/upload/image.png",
            "reason": "model_multimodal_disabled",
        }
    ]
    assert policy["inline_text_from_attachments"] is False


def test_resolve_model_input_policy_should_map_native_image_when_supported() -> None:
    policy = resolve_model_input_policy(
        llm=_FakeLLM(multimodal=True, supported=["text", "image"]),
        input_parts=[
            {
                "type": "image",
                "filepath": "/home/ubuntu/upload/image.png",
                "uri": "https://cdn.example.com/image.png",
                "mime_type": "image/png",
            },
            {
                "type": "audio",
                "filepath": "/home/ubuntu/upload/audio.mp3",
                "uri": "https://cdn.example.com/audio.mp3",
                "mime_type": "audio/mpeg",
            },
        ],
    )

    assert policy["native_user_content_parts"] == [
        {
            "type": "image_url",
            "image_url": {"url": "https://cdn.example.com/image.png"},
        }
    ]
    assert policy["unsupported_parts"] == [
        {
            "type": "audio",
            "filepath": "/home/ubuntu/upload/audio.mp3",
            "reason": "model_supported_not_contains_type",
        }
    ]
    assert policy["inline_text_from_attachments"] is False


def test_resolve_model_input_policy_should_prefer_native_text_file_ref_when_multimodal_enabled() -> None:
    policy = resolve_model_input_policy(
        llm=_FakeLLM(multimodal=True, supported=["text", "image"]),
        input_parts=[
            {"type": "text", "text": "请读取文本附件"},
            {
                "type": "file_ref",
                "file_id": "file-1",
                "filepath": "/home/ubuntu/upload/notes.md",
                "uri": "https://cdn.example.com/notes.md",
                "mime_type": "text/markdown",
                "extension": "md",
                "text_content": "这是一个测试文本附件内容。",
            },
        ],
    )

    assert policy["text_attachment_paths"] == []
    assert len(policy["native_user_content_parts"]) == 1
    native_part = policy["native_user_content_parts"][0]
    assert native_part["type"] == "text"
    assert "source: file-1" in native_part["text"]
    assert "这是一个测试文本附件内容。" in native_part["text"]
    assert policy["unsupported_parts"] == []
    assert policy["inline_text_from_attachments"] is True


def test_resolve_model_input_policy_should_fallback_to_sandbox_when_text_content_unavailable() -> None:
    policy = resolve_model_input_policy(
        llm=_FakeLLM(multimodal=True, supported=["text", "image"]),
        input_parts=[
            {
                "type": "file_ref",
                "file_id": "file-1",
                "filepath": "/home/ubuntu/upload/notes.md",
                "mime_type": "text/markdown",
                "extension": "md",
            },
        ],
    )

    assert policy["text_attachment_paths"] == ["/home/ubuntu/upload/notes.md"]
    assert policy["native_user_content_parts"] == []
    assert policy["unsupported_parts"] == [
        {
            "type": "file_ref",
            "filepath": "/home/ubuntu/upload/notes.md",
            "reason": "text_content_unavailable_fallback_to_sandbox",
        }
    ]
    assert policy["inline_text_from_attachments"] is False


def test_execute_step_with_prompt_should_send_multimodal_user_content_when_parts_provided() -> None:
    class _CaptureLLM(_FakeLLM):
        def __init__(self) -> None:
            super().__init__(multimodal=True, supported=["text", "image"])
            self.last_messages = None

        async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
            self.last_messages = messages
            return {"role": "assistant", "content": '{"success": true, "result": "ok"}'}

    llm = _CaptureLLM()

    async def _run():
        payload, _ = await execute_step_with_prompt(
            llm=llm,
            execution_prompt="请识别图片内容",
            step=Step(description="识别图片"),
            runtime_tools=None,
            extra_user_content_parts=[
                {
                    "type": "image_url",
                    "image_url": {"url": "https://cdn.example.com/image.png"},
                }
            ],
        )
        return payload

    payload = asyncio.run(_run())
    first_user_message = llm.last_messages[0]
    assert first_user_message["role"] == "user"
    assert first_user_message["content"][0] == {"type": "text", "text": "请识别图片内容"}
    assert first_user_message["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://cdn.example.com/image.png"},
    }
    assert payload["result"] == "ok"


def test_execute_step_with_prompt_should_block_disallowed_read_file_call() -> None:
    class _FakeFileTool(BaseTool):
        name = "file"

        def __init__(self) -> None:
            super().__init__()
            self.invoked = False

        @tool(
            name="read_file",
            description="Read file content",
            parameters={"filepath": {"type": "string", "description": "path"}},
            required=["filepath"],
        )
        async def read_file(self, filepath: str):
            self.invoked = True
            return {"success": True, "data": {"content": "should-not-run"}}

    class _ToolCallLLM(_FakeLLM):
        def __init__(self) -> None:
            super().__init__(multimodal=True, supported=["text"])
            self.call_index = 0

        async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
            self.call_index += 1
            if self.call_index == 1:
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"filepath":"/home/ubuntu/upload/1.txt"}',
                            },
                        }
                    ],
                }
            return {"role": "assistant", "content": '{"success": true, "result": "ok"}'}

    class _FakeSearchTool(BaseTool):
        name = "search"

        @tool(
            name="web_search",
            description="Search web",
            parameters={"query": {"type": "string", "description": "query"}},
            required=["query"],
        )
        async def web_search(self, query: str):
            return {"success": True, "data": {"items": []}}

    llm = _ToolCallLLM()
    file_tool = _FakeFileTool()
    search_tool = _FakeSearchTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            execution_prompt="读取文本内容",
            step=Step(description="读取文本"),
            runtime_tools=[file_tool, search_tool],
            disallowed_function_names=["read_file"],
        )

    payload, events = asyncio.run(_run())
    assert payload["result"] == "ok"
    assert file_tool.invoked is False
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "read_file"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "工具已禁用" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_hide_disallowed_tool_from_schema() -> None:
    class _FakeFileTool(BaseTool):
        name = "file"

        def __init__(self) -> None:
            super().__init__()
            self.invoked = False

        @tool(
            name="read_file",
            description="Read file content",
            parameters={"filepath": {"type": "string", "description": "path"}},
            required=["filepath"],
        )
        async def read_file(self, filepath: str):
            self.invoked = True
            return {"success": True, "data": {"content": "should-not-run"}}

    class _CaptureToolSchemaLLM(_FakeLLM):
        def __init__(self) -> None:
            super().__init__(multimodal=True, supported=["text"])
            self.last_tools = None

        async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
            self.last_tools = list(tools or [])
            return {"role": "assistant", "content": '{"success": true, "result": "ok"}'}

    llm = _CaptureToolSchemaLLM()
    file_tool = _FakeFileTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            execution_prompt="读取文本内容",
            step=Step(description="读取文本"),
            runtime_tools=[file_tool],
            disallowed_function_names=["read_file"],
        )

    payload, events = asyncio.run(_run())
    assert payload["result"] == "ok"
    assert events == []
    assert file_tool.invoked is False
    assert llm.last_tools is not None
    exposed_function_names = {
        str((schema.get("function") or {}).get("name") or "").strip().lower()
        for schema in llm.last_tools
        if isinstance(schema, dict)
    }
    assert "read_file" not in exposed_function_names
