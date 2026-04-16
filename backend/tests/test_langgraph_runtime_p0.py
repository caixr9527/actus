import asyncio
import json

from app.domain.models import (
    ExecutionStatus,
    MessageEvent,
    Plan,
    Step,
    StepArtifactPolicy,
    StepDeliveryContextState,
    StepDeliveryRole,
    StepOutcome,
    StepOutputMode,
    StepTaskModeHint,
    ToolEventStatus,
    ToolResult,
)
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.tools.base import BaseTool, tool
from app.infrastructure.runtime.langgraph.graphs import bind_live_event_sink, unbind_live_event_sink
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes import (
    create_or_reuse_plan_node as _create_or_reuse_plan_node,
    direct_answer_node,
    direct_execute_node,
    direct_wait_node,
    entry_router_node,
    execute_step_node as _execute_step_node,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.routing import route_after_plan
from app.infrastructure.runtime.langgraph.graphs.planner_react.language_checker import (
    build_direct_path_copy,
    infer_working_language_from_message,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.tools import (
    classify_confirmed_user_task_mode,
    classify_step_task_mode,
    execute_step_with_prompt,
    has_available_file_context,
)


_TEST_RUNTIME_CONTEXT_SERVICE = RuntimeContextService()


async def create_or_reuse_plan_node(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return await _create_or_reuse_plan_node(
        *args,
        **kwargs,
    )


async def execute_step_node(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return await _execute_step_node(
        *args,
        **kwargs,
    )


class _DirectAnswerLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "message": "你好，我在。",
                },
                ensure_ascii=False,
            )
        }


class _InlineDeliveryLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "result": "这是完整的内联交付正文。",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


class _IntermediateInlineDeliveryLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "已展示候选课程",
                    "delivery_text": "候选课程 A、候选课程 B、候选课程 C。",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


class _SplitSummaryDeliveryLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "已完成最终整理",
                    "delivery_text": "这是面向用户的完整最终交付正文。",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


class _SplitSummaryDeliveryWithMixedAttachmentsLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "已完成最终整理",
                    "delivery_text": "这是面向用户的完整最终交付正文。",
                    "attachments": [
                        "artifact-id-1",
                        "https://example.com/final.md",
                        "final-output.md",
                        "/home/ubuntu/final-output.md",
                    ],
                },
                ensure_ascii=False,
            )
        }


class _LeakyNonInlineDeliveryLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "已完成搜索步骤",
                    "delivery_text": "这段正文不应在非 inline 步骤中被保留。",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


class _BrowserOnlyTool(BaseTool):
    name = "browser"

    def __init__(self) -> None:
        super().__init__()
        self.invoked = 0

    @tool(
        name="browser_view",
        description="查看当前浏览器页面",
        parameters={},
        required=[],
    )
    async def browser_view(self):
        self.invoked += 1
        return ToolResult(success=True, data={"url": "https://example.com"})


class _ResearchBlockedLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-browser-view",
                        "function": {
                            "name": "browser_view",
                            "arguments": "{}",
                        },
                    }
                ],
            }
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "result": "已切回非浏览器路径完成步骤",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


class _SearchTool(BaseTool):
    name = "search"

    def __init__(self) -> None:
        super().__init__()
        self.invoked = 0

    @tool(
        name="search_web",
        description="搜索网页",
        parameters={"query": {"type": "string"}},
        required=["query"],
    )
    async def search_web(self, query: str):
        self.invoked += 1
        return ToolResult(success=True, data={"query": query, "results": [{"url": "https://example.com"}]})


class _ReadFileTool(BaseTool):
    name = "file"

    def __init__(self) -> None:
        super().__init__()
        self.invocations: list[str] = []

    @tool(
        name="read_file",
        description="读取文件内容",
        parameters={"filepath": {"type": "string"}},
        required=["filepath"],
    )
    async def read_file(self, filepath: str):
        self.invocations.append(filepath)
        return ToolResult(
            success=True,
            data={
                "filepath": filepath,
                "content": "example content",
            },
        )


class _SearchFetchTool(BaseTool):
    name = "search"

    def __init__(self) -> None:
        super().__init__()
        self.invocations: list[str] = []

    @tool(
        name="search_web",
        description="搜索网页",
        parameters={"query": {"type": "string"}},
        required=["query"],
    )
    async def search_web(self, query: str):
        self.invocations.append("search_web")
        return ToolResult(
            success=True,
            data={
                "query": query,
                "results": [
                    {
                        "url": "https://example.com/article",
                        "title": "Example Article",
                        "content": "Example snippet",
                    }
                ],
            },
        )

    @tool(
        name="fetch_page",
        description="读取网页正文",
        parameters={"url": {"type": "string"}},
        required=["url"],
    )
    async def fetch_page(self, url: str):
        self.invocations.append("fetch_page")
        return ToolResult(
            success=True,
            data={
                "url": url,
                "title": "Example Article",
                "content": "Example article content",
                "excerpt": "Example article content",
            },
        )


class _ShellTool(BaseTool):
    name = "shell"

    def __init__(self) -> None:
        super().__init__()
        self.invoked = 0

    @tool(
        name="shell_execute",
        description="执行 shell 命令",
        parameters={"command": {"type": "string"}},
        required=["command"],
    )
    async def shell_execute(self, command: str):
        self.invoked += 1
        return ToolResult(success=True, data={"command": command, "stdout": "ok"})


class _FinalDeliverySearchDriftLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-search",
                        "function": {
                            "name": "search_web",
                            "arguments": json.dumps({"query": "重庆 五一攻略"}, ensure_ascii=False),
                        },
                    }
                ],
            }
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "已直接完成最终交付",
                    "delivery_text": "这是基于已知上下文整理出的完整最终正文。",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


class _FinalDeliveryShellDriftLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-shell",
                        "function": {
                            "name": "shell_execute",
                            "arguments": json.dumps({"command": "echo summarize"}, ensure_ascii=False),
                        },
                    }
                ],
            }
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "已直接完成最终交付",
                    "delivery_text": "这是基于已知上下文整理出的完整最终正文。",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


class _FileProcessingShellDriftLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-file-shell",
                        "function": {
                            "name": "shell_execute",
                            "arguments": json.dumps({"command": "git status"}, ensure_ascii=False),
                        },
                    }
                ],
            }
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "文件处理步骤已完成",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


class _RepeatedSearchLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-search",
                    "function": {
                        "name": "search_web",
                        "arguments": json.dumps({"query": "openai 最新消息"}, ensure_ascii=False),
                    },
                }
            ],
        }


class _RepeatedFetchPageLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": "call-fetch",
                    "function": {
                        "name": "fetch_page",
                        "arguments": json.dumps({"url": "https://example.com/article"}, ensure_ascii=False),
                    },
                }
            ],
        }


class _SearchTransportErrorTool(BaseTool):
    name = "search"

    def __init__(self) -> None:
        super().__init__()
        self.invoked = 0

    @tool(
        name="search_web",
        description="搜索网页",
        parameters={"query": {"type": "string"}},
        required=["query"],
    )
    async def search_web(self, query: str):
        self.invoked += 1
        return ToolResult(
            success=False,
            message="RemoteProtocolError: Server disconnected without sending a response",
            data=None,
        )


class _NoAttachmentPreferenceLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "已创建文件",
                    "attachments": ["/home/ubuntu/workspace/p3-artifact/result.md"],
                },
                ensure_ascii=False,
            )
        }


class _ReadFileThenFinishLLM:
    def __init__(self, filepath: str) -> None:
        self.filepath = filepath
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-read-file",
                        "function": {
                            "name": "read_file",
                            "arguments": json.dumps({"filepath": self.filepath}, ensure_ascii=False),
                        },
                    }
                ],
            }
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "result": "已结束步骤",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


class _FetchBeforeSearchLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-fetch-first",
                        "function": {
                            "name": "fetch_page",
                            "arguments": json.dumps({"url": "https://example.com/article"}, ensure_ascii=False),
                        },
                    }
                ],
            }
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "result": "已按要求结束当前步骤",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


class _SearchThenRepeatSearchLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-search-first",
                        "function": {
                            "name": "search_web",
                            "arguments": json.dumps({"query": "openai docs"}, ensure_ascii=False),
                        },
                    }
                ],
            }
        if self.calls == 2:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-search-again",
                        "function": {
                            "name": "search_web",
                            "arguments": json.dumps({"query": "openai docs"}, ensure_ascii=False),
                        },
                    }
                ],
            }
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "result": "已停止重复搜索并结束步骤",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


class _SearchThenSearchAndFetchLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-search-first",
                        "function": {
                            "name": "search_web",
                            "arguments": json.dumps({"query": "openai docs"}, ensure_ascii=False),
                        },
                    }
                ],
            }
        if self.calls == 2:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-search-again",
                        "function": {
                            "name": "search_web",
                            "arguments": json.dumps({"query": "openai docs"}, ensure_ascii=False),
                        },
                    },
                    {
                        "id": "call-fetch-after-search",
                        "function": {
                            "name": "fetch_page",
                            "arguments": json.dumps({"url": "https://example.com/article"}, ensure_ascii=False),
                        },
                    },
                ],
            }
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "result": "已优先读取正文并结束步骤",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


class _BrowserProgressTool(BaseTool):
    name = "browser"

    def __init__(self) -> None:
        super().__init__()
        self.invocations: list[str] = []

    @tool(
        name="browser_view",
        description="查看当前浏览器页面",
        parameters={},
        required=[],
    )
    async def browser_view(self):
        self.invocations.append("browser_view")
        return ToolResult(success=True, data={"url": "https://example.com", "title": "Example"})

    @tool(
        name="browser_scroll_down",
        description="向下滚动页面",
        parameters={},
        required=[],
    )
    async def browser_scroll_down(self):
        self.invocations.append("browser_scroll_down")
        return ToolResult(success=True, data={"url": "https://example.com", "title": "Example"})


class _BrowserNoProgressLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            function_name = "browser_view"
        elif self.calls == 2:
            function_name = "browser_scroll_down"
        else:
            function_name = "browser_view"
        return {
            "content": "",
            "tool_calls": [
                {
                    "id": f"call-{self.calls}",
                    "function": {
                        "name": function_name,
                        "arguments": "{}",
                    },
                }
            ],
        }


class _PlannerResearchDriftLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "message": "已生成计划",
                    "goal": "先从慕课网上找三门关于AI Agent的课程名称",
                    "title": "课程检索任务",
                    "language": "zh",
                    "steps": [
                        {
                            "id": "1",
                            "description": "访问搜索结果中的多个页面，提取至少三门课程的名称，并将这些名称保存到临时文件中以备后续使用。",
                            "task_mode_hint": "research",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        }


class _PlanOnlyPlannerLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "message": "好的，我将先给出执行步骤，不直接开始执行。",
                    "goal": "规划北京三日游",
                    "title": "北京三日游计划",
                    "language": "zh",
                    "steps": [
                        {
                            "id": "1",
                            "description": "检索北京主要景点、交通与住宿信息",
                            "task_mode_hint": "research",
                            "output_mode": "none",
                            "artifact_policy": "forbid_file_output",
                            "delivery_role": "none",
                            "delivery_context_state": "none",
                        },
                        {
                            "id": "2",
                            "description": "整理成三天行程草案",
                            "task_mode_hint": "general",
                            "output_mode": "inline",
                            "artifact_policy": "default",
                            "delivery_role": "intermediate",
                            "delivery_context_state": "none",
                        },
                    ],
                },
                ensure_ascii=False,
            )
        }


def test_entry_router_node_should_route_direct_answer_for_greeting() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "你好",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_strategy"] == "direct_answer"


def test_entry_router_node_should_route_direct_wait_for_preconfirm_request() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "在你开始搜索之前先让我确认",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_strategy"] == "direct_wait"


def test_entry_router_node_should_not_route_long_mixed_request_to_direct_wait() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "先简单从慕课网(imooc.com)上找三门关于AI Agent的课程名称，使用搜索工具前先让我确认",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_strategy"] == "recall_memory_context"


def test_entry_router_node_should_not_route_waiting_plan_request_to_direct_wait() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "帮我制定周末出行方案；如果你需要我先确认预算和偏好，请先停下来问我。",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_strategy"] == "recall_memory_context"


def test_entry_router_node_should_route_direct_execute_for_simple_tool_task() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "搜索 OpenAI 官网",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_strategy"] == "direct_execute"


def test_entry_router_node_should_route_direct_execute_for_url_request_without_tool_name_hint() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "打开 https://openai.com/docs 看一下当前页面",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_strategy"] == "direct_execute"


def test_entry_router_node_should_route_search_read_and_summarize_request_to_planner() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "帮我搜索并阅读 OpenAI 官网关于 Agents 的文档，整理关键点。",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_strategy"] == "recall_memory_context"


def test_entry_router_node_should_route_explicit_plan_request_to_planner() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "帮我规划一个北京 3 天旅游安排，先不要执行，只给步骤。",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_strategy"] == "recall_memory_context"


def test_entry_router_node_should_mark_plan_only_request_in_control_metadata() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "帮我规划一个北京 3 天旅游安排，先不要执行，只给步骤。",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_strategy"] == "recall_memory_context"
    assert state["graph_metadata"]["control"]["plan_only"] is True


def test_entry_router_node_should_keep_single_file_read_request_on_direct_execute() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "读取 /tmp/backend.log 并整理错误摘要",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_strategy"] == "direct_execute"


def test_classify_step_task_mode_should_use_artifact_and_command_signals() -> None:
    assert classify_step_task_mode(
        Step(description="执行 `pytest backend/tests/test_run_engine_selector.py -q` 并修复失败")
    ) == "coding"
    assert classify_step_task_mode(
        Step(description="读取 /tmp/backend.log 并整理错误摘要")
    ) == "file_processing"


def test_classify_step_task_mode_should_ignore_low_value_success_criteria_noise() -> None:
    assert classify_step_task_mode(
        Step(
            description="读取 /tmp/backend.log 并整理错误摘要",
            success_criteria=["完成任务", "继续处理"],
        )
    ) == "file_processing"


def test_classify_step_task_mode_should_prefer_web_reading_over_browser_interaction_for_page_reading() -> None:
    assert classify_step_task_mode(
        Step(description="打开 https://openai.com/blog 阅读页面内容并提炼要点")
    ) == "web_reading"
    assert classify_step_task_mode(
        Step(description="登录后台后点击提交按钮并填写表单")
    ) == "browser_interaction"


def test_classify_confirmed_user_task_mode_should_ignore_wait_signal() -> None:
    user_message = "先让我确认后再继续搜索课程"

    assert classify_step_task_mode(Step(description=user_message)) == "human_wait"
    assert classify_confirmed_user_task_mode(user_message) == "research"


def test_classify_confirmed_user_task_mode_should_fallback_to_research_for_planning_query() -> None:
    user_message = "帮我制定周末出行方案"

    assert classify_confirmed_user_task_mode(user_message) == "research"


def test_classify_step_task_mode_should_detect_planner_wait_phrases() -> None:
    assert classify_step_task_mode(
        Step(description="向用户请求确认是否开始使用搜索工具查找慕课网上的AI Agent课程。")
    ) == "human_wait"
    assert classify_step_task_mode(
        Step(description="向用户展示找到的三门课程名称，并请求用户选择其中感兴趣的一门。")
    ) == "human_wait"
    assert classify_step_task_mode(
        Step(description="如有疑问，向用户询问并基于反馈决定后续步骤。")
    ) == "human_wait"
    assert classify_step_task_mode(
        Step(description="用户确认后，使用搜索工具搜索课程并整理结果。")
    ) == "research"
    assert classify_step_task_mode(
        Step(description="用户选择课程后，使用浏览器工具访问选定页面查看详情。")
    ) == "web_reading"


def test_classify_step_task_mode_should_prioritize_structured_task_mode_hint() -> None:
    assert classify_step_task_mode(
        Step(
            description="根据用户选择继续处理课程详情。",
            task_mode_hint="human_wait",
        )
    ) == "human_wait"
    assert classify_step_task_mode(
        Step(
            description="请确认是否继续执行当前任务。",
            task_mode_hint="research",
        )
    ) == "research"


def test_language_checker_should_infer_working_language_from_user_message() -> None:
    assert infer_working_language_from_message("请用中文总结这个页面") == "zh"
    assert infer_working_language_from_message("Read this file and summarize in English") == "en"
    assert infer_working_language_from_message("日本語で要約してください") == "ja"
    assert infer_working_language_from_message("") == "zh"


def test_language_checker_should_build_direct_path_copy_by_language() -> None:
    zh_copy = build_direct_path_copy("zh")
    en_copy = build_direct_path_copy("en")

    assert zh_copy["direct_execute_message"] == "已进入直接执行路径。"
    assert zh_copy["direct_wait_confirm_label"] == "继续"
    assert en_copy["direct_execute_message"] == "Entered direct execution path."
    assert en_copy["direct_wait_confirm_label"] == "Continue"


def test_direct_answer_node_should_build_completed_plan() -> None:
    state = asyncio.run(
        direct_answer_node(
            {
                "user_message": "你好",
                "graph_metadata": {},
            },
            _DirectAnswerLLM(),
        )
    )

    assert state["plan"] is not None
    assert state["plan"].status == ExecutionStatus.COMPLETED
    assert state["plan"].language == "zh"
    assert state["final_message"] == "你好，我在。"
    assert state["graph_metadata"]["control"]["skip_replan_when_plan_finished"] is True


def test_direct_wait_node_should_build_synthetic_wait_plan() -> None:
    user_message = "先让我确认后再继续搜索课程"
    state = asyncio.run(
        direct_wait_node(
            {
                "user_message": user_message,
                "graph_metadata": {},
            }
        )
    )

    assert state["plan"] is not None
    assert state["plan"].language == "zh"
    assert [step.id for step in state["plan"].steps] == ["direct-wait-confirm", "direct-wait-execute"]
    assert state["plan"].steps[1].description == "执行原始任务"
    assert state["pending_interrupt"]["kind"] == "confirm"
    assert state["graph_metadata"]["control"]["skip_replan_when_plan_finished"] is True
    assert state["graph_metadata"]["control"]["direct_wait_original_message"] == user_message
    assert state["graph_metadata"]["control"]["direct_wait_execute_task_mode"] == "research"
    assert state["graph_metadata"]["control"]["direct_wait_original_task_executed"] is False
    assert state["plan"].steps[1].task_mode_hint == "research"
    assert state["plan"].steps[1].delivery_role == "final"
    assert state["plan"].steps[1].delivery_context_state == "needs_preparation"


def test_direct_execute_node_should_build_single_step_plan() -> None:
    state = asyncio.run(
        direct_execute_node(
            {
                "user_message": "搜索 OpenAI 官网",
                "graph_metadata": {},
            }
        )
    )

    assert state["plan"] is not None
    assert state["plan"].language == "zh"
    assert len(state["plan"].steps) == 1
    assert state["current_step_id"] == "direct-execute-step"
    assert state["graph_metadata"]["control"]["skip_replan_when_plan_finished"] is True
    assert state["plan"].steps[0].task_mode_hint == "research"
    assert state["plan"].steps[0].delivery_role == "final"
    assert state["plan"].steps[0].delivery_context_state == "needs_preparation"


def test_direct_wait_node_should_preserve_english_working_language() -> None:
    state = asyncio.run(
        direct_wait_node(
            {
                "user_message": "Ask me for confirmation before you search the OpenAI docs",
                "graph_metadata": {},
            }
        )
    )

    assert state["plan"] is not None
    assert state["plan"].language == "en"
    assert state["plan"].message == "This task requires confirmation before execution can continue."
    assert state["pending_interrupt"]["prompt"] == "Please confirm whether execution is allowed before continuing this task."
    assert state["pending_interrupt"]["confirm_label"] == "Continue"
    assert state["pending_interrupt"]["cancel_label"] == "Cancel"


def test_create_or_reuse_plan_node_should_default_research_step_to_forbid_file_output() -> None:
    state = asyncio.run(
        create_or_reuse_plan_node(
            {
                "user_message": "先简单从慕课网上找三门关于AI Agent的课程名称",
                "graph_metadata": {},
                "working_memory": {},
                "recent_run_briefs": [],
                "recent_attempt_briefs": [],
                "session_open_questions": [],
                "session_blockers": [],
                "selected_artifacts": [],
                "historical_artifact_paths": [],
                "input_parts": [],
                "message_window": [],
                "retrieved_memories": [],
                "pending_memory_writes": [],
                "current_step_id": None,
                "execution_count": 0,
                "max_execution_steps": 20,
                "last_executed_step": None,
                "pending_interrupt": {},
                "emitted_events": [],
                "final_message": "",
                "thread_id": "thread-1",
                "conversation_summary": "",
            },
            _PlannerResearchDriftLLM(),
        )
    )

    assert state["plan"] is not None
    assert "临时文件" in state["plan"].steps[0].description
    assert state["plan"].steps[0].output_mode == "none"
    assert state["plan"].steps[0].artifact_policy == "forbid_file_output"


def test_create_or_reuse_plan_node_should_fix_file_processing_write_conflict_step_contract() -> None:
    class _PlannerConflictLLM:
        async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
            return {
                "content": json.dumps(
                    {
                        "message": "已生成计划",
                        "goal": "创建 hello.txt",
                        "title": "文件创建任务",
                        "language": "zh",
                        "steps": [
                            {
                                "id": "1",
                                "description": "在 /tmp 下创建 hello.txt 并写入 HELLO",
                                "task_mode_hint": "file_processing",
                                "output_mode": "none",
                                "artifact_policy": "forbid_file_output",
                                "delivery_role": "none",
                                "delivery_context_state": "none",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            }

    state = asyncio.run(
        create_or_reuse_plan_node(
            {
                "user_message": "在 /tmp 下创建 hello.txt，写入 HELLO",
                "graph_metadata": {},
                "working_memory": {},
                "recent_run_briefs": [],
                "recent_attempt_briefs": [],
                "session_open_questions": [],
                "session_blockers": [],
                "selected_artifacts": [],
                "historical_artifact_paths": [],
                "input_parts": [],
                "message_window": [],
                "retrieved_memories": [],
                "pending_memory_writes": [],
                "current_step_id": None,
                "execution_count": 0,
                "max_execution_steps": 20,
                "last_executed_step": None,
                "pending_interrupt": {},
                "emitted_events": [],
                "final_message": "",
                "thread_id": "thread-1",
                "conversation_summary": "",
            },
            _PlannerConflictLLM(),
        )
    )

    assert state["plan"] is not None
    assert state["plan"].steps[0].artifact_policy == "allow_file_output"
    assert state["plan"].steps[0].output_mode == "file"
    # P3-一次性收口：编译后结构化字段必须保持 Enum 类型，禁止字符串回流持久化层。
    assert isinstance(state["plan"].steps[0].task_mode_hint, StepTaskModeHint)
    assert isinstance(state["plan"].steps[0].output_mode, StepOutputMode)
    assert isinstance(state["plan"].steps[0].artifact_policy, StepArtifactPolicy)
    assert isinstance(state["plan"].steps[0].delivery_role, StepDeliveryRole)
    assert isinstance(state["plan"].steps[0].delivery_context_state, StepDeliveryContextState)


def test_create_or_reuse_plan_node_should_keep_human_wait_contract_fields_as_enum() -> None:
    class _PlannerHumanWaitLLM:
        async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
            return {
                "content": json.dumps(
                    {
                        "message": "已生成计划",
                        "goal": "等待用户确认",
                        "title": "等待确认任务",
                        "language": "zh",
                        "steps": [
                            {
                                "id": "1",
                                "description": "先等待用户确认后再继续",
                                "task_mode_hint": "human_wait",
                                "output_mode": "inline",
                                "artifact_policy": "allow_file_output",
                                "delivery_role": "final",
                                "delivery_context_state": "ready",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            }

    state = asyncio.run(
        create_or_reuse_plan_node(
            {
                "user_message": "先让我确认",
                "graph_metadata": {},
                "working_memory": {},
                "recent_run_briefs": [],
                "recent_attempt_briefs": [],
                "session_open_questions": [],
                "session_blockers": [],
                "selected_artifacts": [],
                "historical_artifact_paths": [],
                "input_parts": [],
                "message_window": [],
                "retrieved_memories": [],
                "pending_memory_writes": [],
                "current_step_id": None,
                "execution_count": 0,
                "max_execution_steps": 20,
                "last_executed_step": None,
                "pending_interrupt": {},
                "emitted_events": [],
                "final_message": "",
                "thread_id": "thread-1",
                "conversation_summary": "",
            },
            _PlannerHumanWaitLLM(),
        )
    )

    assert state["plan"] is not None
    step = state["plan"].steps[0]
    assert isinstance(step.output_mode, StepOutputMode)
    assert isinstance(step.artifact_policy, StepArtifactPolicy)
    assert isinstance(step.delivery_role, StepDeliveryRole)
    assert isinstance(step.delivery_context_state, StepDeliveryContextState)
    assert step.output_mode == StepOutputMode.NONE
    assert step.artifact_policy == StepArtifactPolicy.FORBID_FILE_OUTPUT
    assert step.delivery_role == StepDeliveryRole.NONE
    assert step.delivery_context_state == StepDeliveryContextState.NONE


def test_direct_execute_node_should_preserve_english_working_language() -> None:
    state = asyncio.run(
        direct_execute_node(
            {
                "user_message": "Read this file and summarize in English",
                "graph_metadata": {},
            }
        )
    )

    assert state["plan"] is not None
    assert state["plan"].language == "en"
    assert state["plan"].message == "Entered direct execution path."
    assert state["plan"].steps[0].title == "Execute the user request directly"


def test_create_or_reuse_plan_node_should_stop_after_planning_for_plan_only_request() -> None:
    state = asyncio.run(
        create_or_reuse_plan_node(
            {
                "user_message": "帮我规划一个北京 3 天旅游安排，先不要执行，只给步骤。",
                "graph_metadata": {"control": {"plan_only": True}},
                "working_memory": {},
                "recent_run_briefs": [],
                "recent_attempt_briefs": [],
                "session_open_questions": [],
                "session_blockers": [],
                "selected_artifacts": [],
                "historical_artifact_paths": [],
                "input_parts": [],
                "message_window": [],
                "retrieved_memories": [],
                "pending_memory_writes": [],
                "current_step_id": None,
                "execution_count": 0,
                "max_execution_steps": 20,
                "last_executed_step": None,
                "pending_interrupt": {},
                "emitted_events": [],
                "final_message": "",
                "thread_id": "thread-1",
                "conversation_summary": "",
            },
            _PlanOnlyPlannerLLM(),
        )
    )

    assert state["plan"] is not None
    assert state["current_step_id"] is None
    assert state["final_message"] == "好的，我将先给出执行步骤，不直接开始执行。"
    assert route_after_plan(state) == "consolidate_memory"


def test_execute_step_with_prompt_should_block_browser_for_research_task() -> None:
    llm = _ResearchBlockedLLM()
    browser_tool = _BrowserOnlyTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="调研 OpenAI 官网信息"),
            runtime_tools=[browser_tool],
            task_mode="research",
            user_content=[{"type": "text", "text": "请先调研 OpenAI 官网信息"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["result"] == "已切回非浏览器路径完成步骤"
    assert browser_tool.invoked == 0
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "browser_view"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "任务模式 research 不允许调用工具" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_block_fetch_page_before_search_for_research_task() -> None:
    llm = _FetchBeforeSearchLLM()
    search_fetch_tool = _SearchFetchTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="调研 OpenAI 文档并提炼要点"),
            runtime_tools=[search_fetch_tool],
            task_mode="research",
            user_content=[{"type": "text", "text": "请调研 OpenAI 文档并提炼要点"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is True
    assert payload["result"] == "已按要求结束当前步骤"
    assert search_fetch_tool.invocations == []
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "fetch_page"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "请先调用 search_web 获取候选链接" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_rewrite_repeated_search_to_fetch_page() -> None:
    llm = _SearchThenRepeatSearchLLM()
    search_fetch_tool = _SearchFetchTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="调研 OpenAI 文档并提炼要点"),
            runtime_tools=[search_fetch_tool],
            task_mode="research",
            user_content=[{"type": "text", "text": "请调研 OpenAI 文档并提炼要点"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is True
    assert payload["result"] == "已停止重复搜索并结束步骤"
    assert search_fetch_tool.invocations == ["search_web", "fetch_page"]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 2
    assert called_events[1].function_name == "fetch_page"
    assert called_events[1].function_result is not None
    assert called_events[1].function_result.success is True


def test_execute_step_with_prompt_should_prefer_fetch_page_after_search_results_are_ready() -> None:
    llm = _SearchThenSearchAndFetchLLM()
    search_fetch_tool = _SearchFetchTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="调研 OpenAI 文档并提炼要点"),
            runtime_tools=[search_fetch_tool],
            task_mode="research",
            user_content=[{"type": "text", "text": "请调研 OpenAI 文档并提炼要点"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is True
    assert payload["result"] == "已优先读取正文并结束步骤"
    assert search_fetch_tool.invocations == ["search_web", "fetch_page"]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 2
    assert [event.function_name for event in called_events] == ["search_web", "fetch_page"]


def test_execute_step_with_prompt_should_break_on_repeated_search_query() -> None:
    llm = _RepeatedSearchLLM()
    search_tool = _SearchTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="检索 OpenAI 最新消息"),
            runtime_tools=[search_tool],
            task_mode="research",
            user_content=[{"type": "text", "text": "请检索 OpenAI 最新消息"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is False
    assert payload["result"] == "当前步骤暂时未能完成：检索 OpenAI 最新消息"
    assert search_tool.invoked == 2
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 3


def test_execute_step_with_prompt_should_break_on_repeated_fetch_page_url() -> None:
    llm = _RepeatedFetchPageLLM()
    search_fetch_tool = _SearchFetchTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="读取同一页面并整理内容"),
            runtime_tools=[search_fetch_tool],
            task_mode="research",
            user_content=[{"type": "text", "text": "请读取 https://example.com/article 这个页面并整理内容"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is False
    assert payload["blockers"] == ["同一页面抓取请求已重复触发多次，当前检索路径没有新增信息。"]
    assert str(payload["next_hint"]).startswith("请切换其他候选 URL、改用其他工具，或结束当前步骤。")
    assert search_fetch_tool.invocations == ["fetch_page", "fetch_page"]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 3
    assert called_events[-1].function_result is not None
    assert called_events[-1].function_result.success is False


def test_execute_step_with_prompt_should_fast_fail_on_search_transport_error() -> None:
    llm = _RepeatedSearchLLM()
    search_tool = _SearchTransportErrorTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="检索 OpenAI 最新消息"),
            runtime_tools=[search_tool],
            task_mode="research",
            user_content=[{"type": "text", "text": "请检索 OpenAI 最新消息"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is False
    assert payload["blockers"] == ["检索/抓取链路出现瞬时网络错误，当前步骤已停止重试。"]
    assert payload["next_hint"] == "请稍后重试，或先基于已有信息继续后续步骤。"
    assert search_tool.invoked == 1
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False


def test_execute_step_with_prompt_should_block_read_file_without_explicit_file_context_in_research() -> None:
    llm = _ReadFileThenFinishLLM("/home/ubuntu/report.md")
    file_tool = _ReadFileTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="调研慕课网 AI Agent 课程信息"),
            runtime_tools=[file_tool],
            task_mode="research",
            user_content=[{"type": "text", "text": "请调研慕课网 AI Agent 课程信息"}],
            has_available_file_context=False,
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is True
    assert payload["result"] == "已结束步骤"
    assert file_tool.invocations == []
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "read_file"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "明确文件路径/文件名" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_allow_read_file_with_explicit_file_context_in_research() -> None:
    llm = _ReadFileThenFinishLLM("/home/ubuntu/upload/course.txt")
    file_tool = _ReadFileTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="读取课程附件并结合检索结果整理摘要"),
            runtime_tools=[file_tool],
            task_mode="research",
            user_content=[{"type": "text", "text": "请读取 /home/ubuntu/upload/course.txt 并整理课程摘要"}],
            has_available_file_context=True,
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is True
    assert payload["result"] == "已结束步骤"
    assert file_tool.invocations == ["/home/ubuntu/upload/course.txt"]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "read_file"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is True


def test_has_available_file_context_should_not_treat_file_signal_as_real_file_context() -> None:
    assert has_available_file_context(
        user_message="请读取课程文件并整理摘要",
        attachment_paths=[],
        artifact_paths=["artifact-id-1", "https://example.com/file.md", "course.md"],
    ) is False
    assert has_available_file_context(
        user_message="请读取 /home/ubuntu/course.md 并整理摘要",
        attachment_paths=[],
        artifact_paths=[],
    ) is True


def test_execute_step_with_prompt_should_block_read_file_in_web_reading() -> None:
    llm = _ReadFileThenFinishLLM("/home/ubuntu/course-detail.md")
    file_tool = _ReadFileTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="读取当前课程页面详情"),
            runtime_tools=[file_tool],
            task_mode="web_reading",
            user_content=[{"type": "text", "text": "请读取当前课程页面详情"}],
            has_available_file_context=True,
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is True
    assert payload["result"] == "已结束步骤"
    assert file_tool.invocations == []
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "read_file"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "网页读取任务" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_block_file_tools_for_inline_general_without_file_context() -> None:
    llm = _ReadFileThenFinishLLM("/home/ubuntu/course-detail.md")
    file_tool = _ReadFileTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="将课程详情直接展示给用户",
                output_mode="inline",
                artifact_policy="default",
            ),
            runtime_tools=[file_tool],
            task_mode="general",
            user_content=[{"type": "text", "text": "请直接展示课程详情"}],
            has_available_file_context=False,
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is True
    assert payload["result"] == "已结束步骤"
    assert file_tool.invocations == []
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "read_file"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "内联展示结果" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_allow_file_tools_for_inline_general_with_file_context() -> None:
    llm = _ReadFileThenFinishLLM("/home/ubuntu/course-detail.md")
    file_tool = _ReadFileTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="基于已有课程详情文件整理内联摘要",
                output_mode="inline",
                artifact_policy="default",
            ),
            runtime_tools=[file_tool],
            task_mode="general",
            user_content=[{"type": "text", "text": "请基于已有课程详情文件整理摘要"}],
            has_available_file_context=True,
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is True
    assert payload["result"] == "已结束步骤"
    assert file_tool.invocations == ["/home/ubuntu/course-detail.md"]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "read_file"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is True


def test_execute_step_with_prompt_should_block_search_for_final_delivery_step() -> None:
    llm = _FinalDeliverySearchDriftLLM()
    search_tool = _SearchFetchTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="基于已有信息整理最终攻略",
                output_mode="inline",
                artifact_policy="default",
                delivery_role="final",
            ),
            runtime_tools=[search_tool],
            task_mode="general",
            user_content=[{"type": "text", "text": "请基于已有信息整理最终攻略"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["summary"] == "已直接完成最终交付"
    assert payload["delivery_text"] == "这是基于已知上下文整理出的完整最终正文。"
    assert search_tool.invocations == []
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "search_web"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "负责最终交付正文" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_block_search_for_final_delivery_step_even_if_task_mode_is_web_reading() -> None:
    llm = _FinalDeliverySearchDriftLLM()
    search_tool = _SearchFetchTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="基于已有页面信息整理最终课程详情",
                output_mode="inline",
                artifact_policy="default",
                delivery_role="final",
                delivery_context_state="ready",
            ),
            runtime_tools=[search_tool],
            task_mode="web_reading",
            user_content=[{"type": "text", "text": "请整理最终课程详情"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["summary"] == "已直接完成最终交付"
    assert payload["delivery_text"] == "这是基于已知上下文整理出的完整最终正文。"
    assert search_tool.invocations == []
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "search_web"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "负责最终交付正文" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_block_shell_for_final_delivery_step() -> None:
    llm = _FinalDeliveryShellDriftLLM()
    shell_tool = _ShellTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="基于已有上下文输出最终结论",
                output_mode="inline",
                artifact_policy="default",
                delivery_role="final",
                delivery_context_state="ready",
            ),
            runtime_tools=[shell_tool],
            task_mode="general",
            user_content=[{"type": "text", "text": "请输出最终结论"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["summary"] == "已直接完成最终交付"
    assert payload["delivery_text"] == "这是基于已知上下文整理出的完整最终正文。"
    assert shell_tool.invoked == 0
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "shell_execute"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "不要调用 shell_execute" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_block_shell_for_file_processing_without_explicit_command() -> None:
    llm = _FileProcessingShellDriftLLM()
    shell_tool = _ShellTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="创建结果文件并返回摘要"),
            runtime_tools=[shell_tool],
            task_mode="file_processing",
            user_content=[{"type": "text", "text": "创建 /home/ubuntu/output.md 并写入总结内容"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["summary"] == "文件处理步骤已完成"
    assert shell_tool.invoked == 0
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "shell_execute"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "默认禁止调用 shell_execute" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_allow_shell_for_file_processing_with_explicit_command() -> None:
    llm = _FileProcessingShellDriftLLM()
    shell_tool = _ShellTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="执行命令并返回输出"),
            runtime_tools=[shell_tool],
            task_mode="file_processing",
            user_content=[{"type": "text", "text": "请执行命令 git status 并返回输出"}],
        )

    payload, _events = asyncio.run(_run())

    assert payload["summary"] == "文件处理步骤已完成"
    assert shell_tool.invoked == 1


def test_direct_execute_step_should_allow_search_before_final_delivery_when_context_not_ready() -> None:
    llm = _FinalDeliverySearchDriftLLM()
    search_tool = _SearchTool()
    state = asyncio.run(
        direct_execute_node(
            {
                "user_message": "搜索 OpenAI 官网",
                "graph_metadata": {},
            }
        )
    )

    next_state = asyncio.run(execute_step_node(state, llm, runtime_tools=[search_tool]))

    assert search_tool.invoked == 1
    assert next_state["last_executed_step"].status == ExecutionStatus.COMPLETED
    assert next_state["final_message"] == "已直接完成最终交付"
    assert next_state["working_memory"]["final_delivery_payload"] == {
        "text": "这是基于已知上下文整理出的完整最终正文。",
        "sections": [],
        "source_refs": [],
    }


def test_direct_wait_execute_step_should_allow_search_before_final_delivery_when_context_not_ready() -> None:
    llm = _FinalDeliverySearchDriftLLM()
    search_tool = _SearchTool()
    state = asyncio.run(
        direct_wait_node(
            {
                "user_message": "先让我确认后再继续搜索课程",
                "graph_metadata": {},
            }
        )
    )
    state["plan"].steps[0].status = ExecutionStatus.COMPLETED
    state["plan"].steps[0].outcome = StepOutcome(done=True, summary="用户已确认继续")
    state["current_step_id"] = "direct-wait-execute"
    state["pending_interrupt"] = {}

    next_state = asyncio.run(execute_step_node(state, llm, runtime_tools=[search_tool]))

    assert search_tool.invoked == 1
    assert next_state["last_executed_step"].status == ExecutionStatus.COMPLETED
    assert next_state["graph_metadata"]["control"]["direct_wait_original_task_executed"] is True
    assert next_state["final_message"] == "已直接完成最终交付"
    assert next_state["working_memory"]["final_delivery_payload"] == {
        "text": "这是基于已知上下文整理出的完整最终正文。",
        "sections": [],
        "source_refs": [],
    }


def test_execute_step_node_should_treat_selected_artifacts_as_available_file_context() -> None:
    llm = _ReadFileThenFinishLLM("/home/ubuntu/course-detail.md")
    file_tool = _ReadFileTool()
    plan = Plan(
        title="课程详情展示",
        goal="展示课程详情",
        language="zh",
        steps=[
            Step(
                id="step-1",
                title="展示课程详情",
                description="基于已有课程详情文件整理内联摘要",
                output_mode="inline",
                artifact_policy="default",
                status=ExecutionStatus.PENDING,
            )
        ],
    )

    state = {
        "plan": plan,
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {},
        "execution_count": 0,
        "input_parts": [],
        "selected_artifacts": ["/home/ubuntu/course-detail.md"],
        "step_states": [],
        "pending_interrupt": {},
        "retrieved_memories": [],
        "recent_run_briefs": [],
        "recent_attempt_briefs": [],
        "session_open_questions": [],
        "session_blockers": [],
        "historical_artifact_paths": [],
        "emitted_events": [],
        "user_message": "请直接展示课程详情",
        "current_step_id": None,
        "final_message": "",
        "thread_id": "thread-1",
        "conversation_summary": "",
    }

    next_state = asyncio.run(
        execute_step_node(
            state,
            llm,
            runtime_tools=[file_tool],
        )
    )

    assert file_tool.invocations == ["/home/ubuntu/course-detail.md"]
    assert next_state["last_executed_step"].status == ExecutionStatus.COMPLETED
    assert next_state["final_message"] == "已结束步骤"


def test_execute_step_node_should_store_final_delivery_payload_for_inline_step() -> None:
    llm = _InlineDeliveryLLM()
    search_tool = _SearchTool()
    plan = Plan(
        title="生成内联结果",
        goal="生成内联结果",
        language="zh",
        steps=[
            Step(
                id="step-1",
                title="整理最终答案",
                description="整理最终答案并内联展示给用户",
                output_mode="inline",
                artifact_policy="default",
                delivery_role="final",
                status=ExecutionStatus.PENDING,
            )
        ],
    )
    state = {
        "plan": plan,
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {},
        "execution_count": 0,
        "input_parts": [],
        "selected_artifacts": [],
        "step_states": [],
        "pending_interrupt": {},
        "retrieved_memories": [],
        "recent_run_briefs": [],
        "recent_attempt_briefs": [],
        "session_open_questions": [],
        "session_blockers": [],
        "historical_artifact_paths": [],
        "emitted_events": [],
        "user_message": "请直接给我最终答案",
        "current_step_id": None,
        "final_message": "",
        "thread_id": "thread-1",
        "conversation_summary": "",
    }

    next_state = asyncio.run(execute_step_node(state, llm, runtime_tools=[search_tool]))

    assert next_state["final_message"] == "这是完整的内联交付正文。"
    assert next_state["working_memory"]["final_delivery_payload"] == {
        "text": "这是完整的内联交付正文。",
        "sections": [],
        "source_refs": [],
    }


def test_execute_step_node_should_split_light_summary_and_final_delivery_text() -> None:
    llm = _SplitSummaryDeliveryLLM()
    search_tool = _SearchTool()
    plan = Plan(
        title="生成最终交付",
        goal="生成最终交付",
        language="zh",
        steps=[
            Step(
                id="step-1",
                title="整理最终答案",
                description="整理最终答案并内联展示给用户",
                output_mode="inline",
                artifact_policy="default",
                delivery_role="final",
                status=ExecutionStatus.PENDING,
            )
        ],
    )
    state = {
        "plan": plan,
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {},
        "execution_count": 0,
        "input_parts": [],
        "selected_artifacts": [],
        "step_states": [],
        "pending_interrupt": {},
        "retrieved_memories": [],
        "recent_run_briefs": [],
        "recent_attempt_briefs": [],
        "session_open_questions": [],
        "session_blockers": [],
        "historical_artifact_paths": [],
        "emitted_events": [],
        "user_message": "请给我最终答案",
        "current_step_id": None,
        "final_message": "",
        "thread_id": "thread-1",
        "conversation_summary": "",
    }

    next_state = asyncio.run(execute_step_node(state, llm, runtime_tools=[search_tool]))

    # final_message 只保留轻量摘要，最终重正文单独进入 final_delivery_payload。
    assert next_state["final_message"] == "已完成最终整理"
    assert next_state["working_memory"]["final_delivery_payload"] == {
        "text": "这是面向用户的完整最终交付正文。",
        "sections": [],
        "source_refs": [],
    }


def test_execute_step_node_should_filter_non_file_attachment_refs_from_final_delivery_payload() -> None:
    llm = _SplitSummaryDeliveryWithMixedAttachmentsLLM()
    plan = Plan(
        title="生成最终交付",
        goal="生成最终交付",
        language="zh",
        steps=[
            Step(
                id="step-1",
                title="整理最终答案",
                description="整理最终答案并内联展示给用户",
                output_mode="inline",
                artifact_policy="default",
                delivery_role="final",
                status=ExecutionStatus.PENDING,
            )
        ],
    )
    state = {
        "plan": plan,
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {},
        "execution_count": 0,
        "input_parts": [],
        "selected_artifacts": [],
        "step_states": [],
        "pending_interrupt": {},
        "retrieved_memories": [],
        "recent_run_briefs": [],
        "recent_attempt_briefs": [],
        "session_open_questions": [],
        "session_blockers": [],
        "historical_artifact_paths": [],
        "emitted_events": [],
        "user_message": "请给我最终答案",
        "current_step_id": None,
        "final_message": "",
        "thread_id": "thread-1",
        "conversation_summary": "",
    }

    next_state = asyncio.run(execute_step_node(state, llm, runtime_tools=[]))

    assert next_state["last_executed_step"].outcome.produced_artifacts == ["/home/ubuntu/final-output.md"]
    assert next_state["working_memory"]["final_delivery_payload"] == {
        "text": "这是面向用户的完整最终交付正文。",
        "sections": [],
        "source_refs": ["/home/ubuntu/final-output.md"],
    }


def test_execute_step_node_should_support_no_tool_execution_path() -> None:
    llm = _SplitSummaryDeliveryLLM()
    plan = Plan(
        title="无工具最终交付",
        goal="无工具最终交付",
        language="zh",
        steps=[
            Step(
                id="step-1",
                title="直接整理最终答案",
                description="直接整理最终答案并返回",
                output_mode="inline",
                artifact_policy="default",
                delivery_role="final",
                status=ExecutionStatus.PENDING,
            )
        ],
    )
    state = {
        "plan": plan,
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {},
        "execution_count": 0,
        "input_parts": [],
        "selected_artifacts": [],
        "step_states": [],
        "pending_interrupt": {},
        "retrieved_memories": [],
        "recent_run_briefs": [],
        "recent_attempt_briefs": [],
        "session_open_questions": [],
        "session_blockers": [],
        "historical_artifact_paths": [],
        "emitted_events": [],
        "user_message": "请直接给我最终答案",
        "current_step_id": None,
        "final_message": "",
        "thread_id": "thread-1",
        "conversation_summary": "",
    }

    next_state = asyncio.run(execute_step_node(state, llm, runtime_tools=[]))

    assert next_state["last_executed_step"].status == ExecutionStatus.COMPLETED
    assert next_state["final_message"] == "已完成最终整理"
    assert next_state["working_memory"]["final_delivery_payload"] == {
        "text": "这是面向用户的完整最终交付正文。",
        "sections": [],
        "source_refs": [],
    }


def test_execute_step_node_should_store_attachment_delivery_preference_in_outcome_and_working_memory() -> None:
    llm = _NoAttachmentPreferenceLLM()
    state = asyncio.run(
        direct_execute_node(
            {
                "user_message": "创建文件 /home/ubuntu/workspace/p3-artifact/result.md，内容为 VERSION_1。这一步不要作为最终附件返回。",
                "graph_metadata": {},
            }
        )
    )

    next_state = asyncio.run(execute_step_node(state, llm, runtime_tools=[]))

    last_step = next_state["last_executed_step"]
    assert last_step is not None
    assert last_step.outcome is not None
    assert last_step.outcome.deliver_result_as_attachment is False
    assert next_state["working_memory"]["delivery_controls"] == {
        "source_step_id": str(last_step.id),
        "deliver_result_as_attachment": False,
    }


def test_execute_step_node_should_not_overwrite_final_delivery_payload_for_intermediate_inline_step() -> None:
    llm = _InlineDeliveryLLM()
    search_tool = _SearchTool()
    plan = Plan(
        title="生成中间展示",
        goal="生成中间展示",
        language="zh",
        steps=[
            Step(
                id="step-1",
                title="整理候选项",
                description="整理候选项并内联展示给用户",
                output_mode="inline",
                artifact_policy="default",
                delivery_role="intermediate",
                status=ExecutionStatus.PENDING,
            )
        ],
    )
    existing_payload = {
        "text": "已有最终正文。",
        "sections": [],
        "source_refs": [],
    }
    state = {
        "plan": plan,
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {"final_delivery_payload": existing_payload},
        "execution_count": 0,
        "input_parts": [],
        "selected_artifacts": [],
        "step_states": [],
        "pending_interrupt": {},
        "retrieved_memories": [],
        "recent_run_briefs": [],
        "recent_attempt_briefs": [],
        "session_open_questions": [],
        "session_blockers": [],
        "historical_artifact_paths": [],
        "emitted_events": [],
        "user_message": "请展示候选项",
        "current_step_id": None,
        "final_message": "",
        "thread_id": "thread-1",
        "conversation_summary": "",
    }

    next_state = asyncio.run(execute_step_node(state, llm, runtime_tools=[search_tool]))

    assert next_state["final_message"] == "这是完整的内联交付正文。"
    assert next_state["working_memory"]["final_delivery_payload"] == existing_payload


def test_execute_step_node_should_not_emit_intermediate_message_for_inline_candidate_step() -> None:
    llm = _IntermediateInlineDeliveryLLM()
    search_tool = _SearchTool()
    plan = Plan(
        title="生成候选课程",
        goal="生成候选课程",
        language="zh",
        steps=[
            Step(
                id="step-1",
                title="展示候选课程",
                description="展示候选课程并等待用户选择",
                task_mode_hint="general",
                output_mode="inline",
                artifact_policy="default",
                delivery_role="intermediate",
                status=ExecutionStatus.PENDING,
            )
        ],
    )
    state = {
        "plan": plan,
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {},
        "execution_count": 0,
        "input_parts": [],
        "selected_artifacts": [],
        "step_states": [],
        "pending_interrupt": {},
        "retrieved_memories": [],
        "recent_run_briefs": [],
        "recent_attempt_briefs": [],
        "session_open_questions": [],
        "session_blockers": [],
        "historical_artifact_paths": [],
        "emitted_events": [],
        "user_message": "请展示候选课程",
        "current_step_id": None,
        "final_message": "",
        "thread_id": "thread-1",
        "conversation_summary": "",
    }
    emitted_events = []

    async def _sink(event):
        emitted_events.append(event)

    token = bind_live_event_sink(_sink)
    try:
        next_state = asyncio.run(execute_step_node(state, llm, runtime_tools=[search_tool]))
    finally:
        unbind_live_event_sink(token)

    intermediate_events = [event for event in emitted_events if isinstance(event, MessageEvent)]
    assert len(intermediate_events) == 0
    assert next_state["plan"].steps[0].outcome is not None
    assert next_state["plan"].steps[0].outcome.delivery_text == "候选课程 A、候选课程 B、候选课程 C。"
    assert next_state["final_message"] == "已展示候选课程"


def test_execute_step_node_should_drop_delivery_text_for_non_inline_step() -> None:
    llm = _LeakyNonInlineDeliveryLLM()
    plan = Plan(
        title="搜索课程",
        goal="搜索课程",
        language="zh",
        steps=[
            Step(
                id="step-1",
                title="搜索课程",
                description="搜索课程信息",
                output_mode="none",
                artifact_policy="forbid_file_output",
                delivery_role="none",
                status=ExecutionStatus.PENDING,
            )
        ],
    )
    state = {
        "plan": plan,
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {},
        "execution_count": 0,
        "input_parts": [],
        "selected_artifacts": [],
        "step_states": [],
        "pending_interrupt": {},
        "retrieved_memories": [],
        "recent_run_briefs": [],
        "recent_attempt_briefs": [],
        "session_open_questions": [],
        "session_blockers": [],
        "historical_artifact_paths": [],
        "emitted_events": [],
        "user_message": "请先搜索课程",
        "current_step_id": None,
        "final_message": "",
        "thread_id": "thread-1",
        "conversation_summary": "",
    }

    next_state = asyncio.run(execute_step_node(state, llm, runtime_tools=[]))

    assert next_state["last_executed_step"].outcome is not None
    assert next_state["last_executed_step"].outcome.delivery_text == ""
    assert next_state["final_message"] == "已完成搜索步骤"


def test_execute_step_with_prompt_should_break_on_browser_no_progress() -> None:
    llm = _BrowserNoProgressLLM()
    browser_tool = _BrowserProgressTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="在浏览器中查看课程详情"),
            runtime_tools=[browser_tool],
            task_mode="browser_interaction",
            user_content=[{"type": "text", "text": "请在浏览器中查看课程详情"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is False
    assert payload["result"] == "当前步骤暂时未能完成：在浏览器中查看课程详情"
    assert payload["blockers"] == ["浏览器连续观察未发现新的有效信息，当前页面路径已无进展。"]
    assert payload["next_hint"] == "请更换页面、改用搜索/正文读取，或重新规划当前步骤。"
    assert browser_tool.invocations == ["browser_view", "browser_scroll_down", "browser_view"]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 3


def test_execute_step_with_prompt_should_preserve_loop_break_blockers_and_next_hint() -> None:
    llm = _RepeatedSearchLLM()
    search_tool = _SearchTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="检索 OpenAI 最新消息"),
            runtime_tools=[search_tool],
            task_mode="research",
            user_content=[{"type": "text", "text": "请检索 OpenAI 最新消息"}],
        )

    payload, _events = asyncio.run(_run())

    assert payload["success"] is False
    assert payload["blockers"] == ["同一搜索查询已重复触发多次，当前检索路径没有继续收获。"]
    assert payload["next_hint"] == "请改写搜索关键词、缩小范围，或改用 fetch_page / 文件读取继续。"
