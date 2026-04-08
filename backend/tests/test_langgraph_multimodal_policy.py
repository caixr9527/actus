import asyncio

from app.application.service.model_input_policy import (
    normalize_supported_input_types,
    resolve_model_input_policy,
)
from app.domain.models import Step, ToolEventStatus
from app.domain.services.tools.base import BaseTool, tool
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


def test_normalize_supported_input_types_should_keep_only_supported_multimodal_types() -> None:
    assert normalize_supported_input_types(["image"]) == ["image"]
    assert normalize_supported_input_types(["image", "text", "file"]) == ["image", "file"]
    assert normalize_supported_input_types("audio") == ["audio"]
    assert normalize_supported_input_types("invalid") == []


def test_resolve_model_input_policy_should_mark_all_parts_unsupported_when_model_not_multimodal() -> None:
    policy = resolve_model_input_policy(
        llm=_FakeLLM(multimodal=False, supported=["file"]),
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
                "base64": "ZmFrZS1pbWFnZQ==",
                "mime_type": "image/png",
            },
        ],
    )

    assert policy["native_user_content_parts"] == []
    assert policy["unsupported_parts"] == [
        {
            "type": "file",
            "filepath": "/home/ubuntu/upload/notes.md",
            "reason": "model_multimodal_disabled",
        },
        {
            "type": "image",
            "filepath": "/home/ubuntu/upload/image.png",
            "reason": "model_multimodal_disabled",
        },
    ]


def test_resolve_model_input_policy_should_map_native_image_when_supported() -> None:
    policy = resolve_model_input_policy(
        llm=_FakeLLM(multimodal=True, supported=["image"]),
        input_parts=[
            {
                "type": "image",
                "filepath": "/home/ubuntu/upload/image.png",
                "base64": "ZmFrZS1pbWFnZQ==",
                "mime_type": "image/png",
            },
            {
                "type": "audio",
                "filepath": "/home/ubuntu/upload/audio.mp3",
                "base64": "ZmFrZS1hdWRpbw==",
                "mime_type": "audio/mpeg",
            },
        ],
    )

    assert policy["native_user_content_parts"] == [
        {
            "type": "image",
            "base64": "ZmFrZS1pbWFnZQ==",
            "mime_type": "image/png",
        }
    ]
    assert policy["unsupported_parts"] == [
        {
            "type": "audio",
            "filepath": "/home/ubuntu/upload/audio.mp3",
            "reason": "model_supported_not_contains_type",
        }
    ]


def test_resolve_model_input_policy_should_map_file_ref_to_file_block() -> None:
    policy = resolve_model_input_policy(
        llm=_FakeLLM(multimodal=True, supported=["file"]),
        input_parts=[
            {"type": "text", "text": "请读取文本附件"},
            {
                "type": "file_ref",
                "file_id": "file-1",
                "filepath": "/home/ubuntu/upload/notes.md",
                "base64": "ZmFrZS1maWxl",
                "mime_type": "text/markdown",
            },
        ],
    )

    assert len(policy["native_user_content_parts"]) == 1
    native_part = policy["native_user_content_parts"][0]
    assert native_part["type"] == "file"
    assert native_part["base64"] == "ZmFrZS1maWxl"
    assert native_part["mime_type"] == "text/markdown"
    assert policy["unsupported_parts"] == []


def test_resolve_model_input_policy_should_not_fallback_to_sandbox_when_file_not_supported() -> None:
    policy = resolve_model_input_policy(
        llm=_FakeLLM(multimodal=True, supported=["image"]),
        input_parts=[
            {
                "type": "file",
                "file_id": "file-1",
                "filepath": "/home/ubuntu/upload/notes.md",
                "base64": "ZmFrZS1maWxl",
                "mime_type": "text/markdown",
                "extension": "md",
            },
        ],
    )

    assert policy["native_user_content_parts"] == []
    assert policy["unsupported_parts"] == [
        {
            "type": "file",
            "filepath": "/home/ubuntu/upload/notes.md",
            "reason": "model_supported_not_contains_type",
        }
    ]


def test_resolve_model_input_policy_should_map_audio_with_base64_when_supported() -> None:
    policy = resolve_model_input_policy(
        llm=_FakeLLM(multimodal=True, supported=["audio"]),
        input_parts=[
            {
                "type": "audio",
                "filepath": "/home/ubuntu/upload/audio.wav",
                "mime_type": "audio/wav",
                "base64": "AAAAIGZ0eXBtcDQy",
            }
        ],
    )

    assert policy["native_user_content_parts"] == [
        {
            "type": "audio",
            "base64": "AAAAIGZ0eXBtcDQy",
            "mime_type": "audio/wav",
        }
    ]
    assert policy["unsupported_parts"] == []


def test_execute_step_with_prompt_should_send_multimodal_user_content_when_parts_provided() -> None:
    class _CaptureLLM(_FakeLLM):
        def __init__(self) -> None:
            super().__init__(multimodal=True, supported=["image"])
            self.last_messages = None

        async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
            self.last_messages = messages
            return {"role": "assistant", "content": '{"success": true, "result": "ok"}'}

    llm = _CaptureLLM()

    async def _run():
        payload, _ = await execute_step_with_prompt(
            llm=llm,
            step=Step(description="识别图片"),
            runtime_tools=None,
            user_content=[
                {"type": "text", "text": "请识别图片内容"},
                {
                    "type": "image",
                    "base64": "ZmFrZS1pbWFnZQ==",
                    "mime_type": "image/png",
                }
            ],
        )
        return payload

    payload = asyncio.run(_run())
    first_user_message = next(message for message in llm.last_messages if message["role"] == "user")
    assert first_user_message["content"][0] == {"type": "text", "text": "请识别图片内容"}
    assert first_user_message["content"][1] == {
        "type": "image",
        "base64": "ZmFrZS1pbWFnZQ==",
        "mime_type": "image/png",
    }
    assert payload["result"] == "ok"


def test_execute_step_with_prompt_should_block_shell_call_for_research_task_mode() -> None:
    class _FakeFileTool(BaseTool):
        name = "shell"

        def __init__(self) -> None:
            super().__init__()
            self.invoked = False

        @tool(
            name="shell_execute",
            description="Execute shell command",
            parameters={"command": {"type": "string", "description": "command"}},
            required=["command"],
        )
        async def shell_execute(self, command: str):
            self.invoked = True
            return {"success": True, "data": {"stdout": "should-not-run"}}

    class _ToolCallLLM(_FakeLLM):
        def __init__(self) -> None:
            super().__init__(multimodal=True, supported=[])
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
                                "name": "shell_execute",
                                "arguments": '{"command":"ls"}',
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
    shell_tool = _FakeFileTool()
    search_tool = _FakeSearchTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="读取文本"),
            runtime_tools=[shell_tool, search_tool],
            task_mode="research",
            user_content=[{"type": "text", "text": "读取文本内容"}],
        )

    payload, events = asyncio.run(_run())
    assert payload["result"] == "ok"
    assert shell_tool.invoked is False
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "shell_execute"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "任务模式 research 不允许调用工具" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_hide_shell_tool_from_research_schema() -> None:
    class _FakeFileTool(BaseTool):
        name = "shell"

        def __init__(self) -> None:
            super().__init__()
            self.invoked = False

        @tool(
            name="shell_execute",
            description="Execute shell command",
            parameters={"command": {"type": "string", "description": "command"}},
            required=["command"],
        )
        async def shell_execute(self, command: str):
            self.invoked = True
            return {"success": True, "data": {"stdout": "should-not-run"}}

    class _CaptureToolSchemaLLM(_FakeLLM):
        def __init__(self) -> None:
            super().__init__(multimodal=True, supported=[])
            self.last_tools = None

        async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
            self.last_tools = list(tools or [])
            return {"role": "assistant", "content": '{"success": true, "result": "ok"}'}

    llm = _CaptureToolSchemaLLM()
    file_tool = _FakeFileTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="读取文本"),
            runtime_tools=[file_tool],
            task_mode="research",
            user_content=[{"type": "text", "text": "读取文本内容"}],
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
    assert "shell_execute" not in exposed_function_names
