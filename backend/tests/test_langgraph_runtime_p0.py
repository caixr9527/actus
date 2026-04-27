import asyncio
import json

from app.domain.models import (
    ExecutionStatus,
    MessageEvent,
    Plan,
    PlanEvent,
    SearchResultItem,
    SearchResults,
    Step,
    StepArtifactPolicy,
    StepOutcome,
    StepOutputMode,
    StepTaskModeHint,
    TextStreamDeltaEvent,
    TextStreamEndEvent,
    TextStreamStartEvent,
    ToolEventStatus,
    ToolResult,
)
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.workspace_runtime.entry import EntryCompiler
from app.domain.services.workspace_runtime.entry import reason_codes as entry_reason_codes
from app.domain.services.workspace_runtime.policies.task_mode_policy import (
    classify_confirmed_user_task_mode,
    classify_step_task_mode,
)
from app.domain.services.workspace_runtime.policies.step_contract_policy import (
    compile_step_contracts,
    filter_final_delivery_steps,
)
from app.domain.services.tools.base import BaseTool, tool
from app.infrastructure.runtime.langgraph.graphs import bind_live_event_sink, unbind_live_event_sink
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes import (
    atomic_action_node,
    create_or_reuse_plan_node as _create_or_reuse_plan_node,
    direct_answer_node,
    direct_wait_node,
    entry_router_node,
    execute_step_node as _execute_step_node,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.routing import route_after_plan
from app.infrastructure.runtime.langgraph.graphs.planner_react.language_checker import (
    build_direct_path_copy,
    infer_working_language_from_message,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.tools import (
    execute_step_with_prompt,
    has_available_file_context,
)


_TEST_RUNTIME_CONTEXT_SERVICE = RuntimeContextService()


def _entry_contract_control(user_message: str, **state_overrides):
    state = {
        "user_message": user_message,
        "input_parts": [],
        "message_window": [],
        "recent_run_briefs": [],
        "conversation_summary": "",
        **state_overrides,
    }
    contract = EntryCompiler().compile_state(state)
    return {"entry_contract": contract.model_dump(mode="json")}


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


class _CaptureDirectAnswerPromptLLM:
    def __init__(self) -> None:
        self.messages = None

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.messages = messages
        return {
            "content": json.dumps(
                {
                    "message": "已结合历史上下文作答。",
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
                    "facts_learned": ["候选课程 A", "候选课程 B", "候选课程 C"],
                    "attachments": [],
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
                    "facts_learned": ["这段正文不应在非 inline 步骤中被保留。"],
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
            data=SearchResults(
                query=query,
                results=[
                    SearchResultItem(
                        url="https://example.com/article",
                        title="Example Article",
                        snippet="Short snippet",
                    )
                ],
            ),
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
                "content": "Example article content " * 20,
                "excerpt": "Example article content " * 5,
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


class _SearchThenFinishLLM:
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
                    "summary": "已完成当前步骤",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


class _ShellThenFinishLLM:
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
                    "summary": "已完成当前步骤",
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
        self.tool_feedback_payloads: list[dict] = []

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
            tool_messages = [
                message for message in list(messages or [])
                if isinstance(message, dict) and message.get("role") == "tool"
            ]
            if tool_messages:
                self.tool_feedback_payloads.append(json.loads(str(tool_messages[-1].get("content") or "{}")))
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
                        },
                        {
                            "id": "2",
                            "description": "整理成三天行程草案",
                            "task_mode_hint": "general",
                            "output_mode": "none",
                            "artifact_policy": "forbid_file_output",
                        },
                    ],
                },
                ensure_ascii=False,
            )
        }


class _NoStepPlannerWithExtraTextLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": (
                "```json\n"
                "{\n"
                '  "message": "我会详细说明。",\n'
                '  "language": "zh",\n'
                '  "goal": "",\n'
                '  "title": "",\n'
                '  "steps": []\n'
                "}\n"
                "```\n\n"
                "## 三个典型使用场景\n\n"
                "1. 多阶段智能体工作流。\n"
                "2. 多智能体协作。\n"
                "3. 人机协作审批。"
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

    assert state["graph_metadata"]["control"]["entry_contract"]["route"] == "answer"


def test_entry_router_node_should_keep_followup_on_original_direct_answer_route() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "不够详细，需要详细点4天3夜的攻略",
                "graph_metadata": {},
                "conversation_summary": "上一轮已讨论重庆旅游攻略",
                "message_window": [
                    {"role": "user", "message": "给我一份重庆的旅游攻略"},
                    {"role": "assistant", "message": "这里是一份重庆旅游攻略"},
                ],
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_contract"]["route"] == "answer"


def test_entry_router_node_should_route_followup_with_run_brief_to_direct_answer() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "再详细一点，给我 3 个典型使用场景。",
                "graph_metadata": {},
                "conversation_summary": "上一轮解释了 LangGraph 的基本概念。",
                "message_window": [
                    {"role": "user", "message": "再详细一点，给我 3 个典型使用场景。"},
                ],
                "recent_run_briefs": [
                    {
                        "run_id": "run-1",
                        "title": "LangGraph 简介",
                        "goal": "解释 LangGraph 是什么",
                        "status": "completed",
                    }
                ],
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_contract"]["route"] == "answer"


def test_entry_router_node_should_route_direct_wait_for_preconfirm_request() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "在你开始搜索之前先让我确认",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_contract"]["route"] == "wait"


def test_entry_router_node_should_not_route_long_mixed_request_to_direct_wait() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "先简单从慕课网(imooc.com)上找三门关于AI Agent的课程名称，使用搜索工具前先让我确认",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_contract"]["route"] == "planned_task"


def test_entry_router_node_should_not_route_waiting_plan_request_to_direct_wait() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "帮我制定周末出行方案；如果你需要我先确认预算和偏好，请先停下来问我。",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_contract"]["route"] == "planned_task"


def test_entry_router_node_should_route_atomic_action_for_simple_tool_task() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "搜索 OpenAI 官网",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_contract"]["route"] == "atomic_action"


def test_entry_router_node_should_route_atomic_action_for_url_request_without_tool_name_hint() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "打开 https://openai.com/docs 看一下当前页面",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_contract"]["route"] == "atomic_action"


def test_entry_router_node_should_route_search_read_and_summarize_request_to_planner() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "帮我搜索并阅读 OpenAI 官网关于 Agents 的文档，整理关键点。",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_contract"]["route"] == "planned_task"


def test_entry_router_node_should_route_explicit_plan_request_to_planner() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "帮我规划一个北京 3 天旅游安排，先不要执行，只给步骤。",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_contract"]["route"] == "planned_task"


def test_entry_router_node_should_mark_plan_only_request_in_control_metadata() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "帮我规划一个北京 3 天旅游安排，先不要执行，只给步骤。",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_contract"]["route"] == "planned_task"
    assert state["graph_metadata"]["control"]["entry_contract"]["plan_only"] is True


def test_entry_router_node_should_keep_single_file_read_request_on_atomic_action() -> None:
    state = asyncio.run(
        entry_router_node(
            {
                "user_message": "读取 /tmp/backend.log 并整理错误摘要",
                "graph_metadata": {},
            }
        )
    )

    assert state["graph_metadata"]["control"]["entry_contract"]["route"] == "atomic_action"


def test_classify_step_task_mode_should_use_artifact_and_command_signals() -> None:
    assert classify_step_task_mode(
        Step(description="执行 `pytest backend/tests/test_run_engine_selector.py -q` 并修复失败")
    ) == "coding"
    assert classify_step_task_mode(
        Step(description="在 /tmp 下创建 hello.txt 并写入 HELLO")
    ) == "coding"
    assert classify_step_task_mode(
        Step(description="读取 /tmp/backend.log 并整理错误摘要")
    ) == "file_processing"
    assert classify_step_task_mode(
        Step(description="改写这段文案，让表达更自然")
    ) == "general"


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

    assert zh_copy["atomic_action_message"] == "已进入原子动作路径。"
    assert zh_copy["direct_wait_confirm_label"] == "继续"
    assert en_copy["atomic_action_message"] == "Entered atomic action path."
    assert en_copy["direct_wait_confirm_label"] == "Continue"


def test_direct_answer_node_should_build_completed_plan() -> None:
    state = asyncio.run(
        direct_answer_node(
            {
                "user_message": "你好",
                "graph_metadata": {"control": _entry_contract_control("你好")},
            },
            _DirectAnswerLLM(),
            runtime_context_service=_TEST_RUNTIME_CONTEXT_SERVICE,
        )
    )

    assert state["plan"] is not None
    assert state["plan"].status == ExecutionStatus.COMPLETED
    assert state["plan"].language == "zh"
    assert state["final_message"] == "你好，我在。"
    assert state["graph_metadata"]["control"]["entry_contract"]["route"] == "answer"


def test_direct_answer_node_should_emit_final_message_stream_before_final_message_event() -> None:
    captured_events = []

    async def _sink(event):
        captured_events.append(event)

    token = bind_live_event_sink(_sink)
    try:
        state = asyncio.run(
            direct_answer_node(
                {
                    "session_id": "session-1",
                    "run_id": "run-1",
                    "thread_id": "thread-1",
                    "user_message": "你好",
                    "graph_metadata": {"control": _entry_contract_control("你好")},
                    "message_window": [],
                },
                _DirectAnswerLLM(),
                runtime_context_service=_TEST_RUNTIME_CONTEXT_SERVICE,
            )
        )
    finally:
        unbind_live_event_sink(token)

    assert isinstance(captured_events[0], TextStreamStartEvent)
    assert isinstance(captured_events[1], TextStreamDeltaEvent)
    assert isinstance(captured_events[2], TextStreamEndEvent)
    assert isinstance(captured_events[-1], MessageEvent)
    assert captured_events[-1].stage == "final"
    assert captured_events[-1].message == "你好，我在。"
    assert [event.type for event in state["emitted_events"]] == ["title", "message"]


def test_direct_answer_node_should_append_history_context_to_prompt() -> None:
    llm = _CaptureDirectAnswerPromptLLM()

    state = asyncio.run(
        direct_answer_node(
            {
                "user_message": "不够详细，需要详细点4天3夜的攻略",
                "graph_metadata": {
                    "control": _entry_contract_control(
                        "不够详细，需要详细点4天3夜的攻略",
                        conversation_summary="上一轮已讨论重庆旅游攻略",
                        message_window=[
                            {"role": "user", "message": "给我一份重庆的旅游攻略"},
                            {"role": "assistant", "message": "这里是一份重庆旅游攻略"},
                        ],
                    )
                },
                "conversation_summary": "上一轮已讨论重庆旅游攻略",
                "final_message": "重庆旅游攻略简版",
                "message_window": [
                    {"role": "user", "message": "给我一份重庆的旅游攻略"},
                    {"role": "assistant", "message": "这里是一份重庆旅游攻略"},
                ],
                "recent_run_briefs": [
                    {
                        "run_id": "run-1",
                        "title": "重庆旅游攻略",
                        "goal": "给出重庆旅游攻略",
                        "status": "completed",
                    }
                ],
            },
            llm,
            runtime_context_service=_TEST_RUNTIME_CONTEXT_SERVICE,
        )
    )

    assert state["final_message"] == "已结合历史上下文作答。"
    assert llm.messages is not None
    user_prompt = llm.messages[1]["content"]
    assert "已知上下文" in user_prompt
    assert "重庆旅游攻略" in user_prompt
    assert "recent_messages" in user_prompt
    assert "topic_anchor" in user_prompt
    assert "conversation_summary" in user_prompt
    assert "回答详略必须与用户当前意图匹配" in user_prompt
    assert "必须给出足够完整、结构化、可直接使用的回答" in user_prompt


def test_runtime_context_service_should_build_direct_answer_topic_anchor_from_recent_run_brief() -> None:
    context_packet = _TEST_RUNTIME_CONTEXT_SERVICE.build_packet(
        stage="direct_answer",
        state={
            "user_message": "不够详细，需要详细点4天3夜的攻略",
            "message_window": [
                {"role": "user", "message": "不够详细，需要详细点4天3夜的攻略"},
            ],
            "conversation_summary": "",
            "final_message": "",
            "recent_run_briefs": [
                {
                    "run_id": "run-1",
                    "title": "重庆旅游攻略",
                    "goal": "给出重庆旅游攻略",
                    "status": "completed",
                    "final_answer_summary": "重庆攻略涵盖解放碑、洪崖洞、长江索道和火锅安排。",
                    "final_answer_text_excerpt": "重庆旅游可以安排 3 到 4 天，核心体验山城交通和夜景。",
                }
            ],
        },
        task_mode="general",
    )

    stable_background = dict(context_packet.get("stable_background") or {})
    topic_anchor = dict(stable_background.get("topic_anchor") or {})
    assert topic_anchor["source"] == "recent_run_brief"
    assert "重庆旅游攻略" in topic_anchor["text"]
    assert "洪崖洞" in topic_anchor["text"]


def test_runtime_context_service_should_fallback_to_previous_final_message_for_topic_anchor() -> None:
    context_packet = _TEST_RUNTIME_CONTEXT_SERVICE.build_packet(
        stage="direct_answer",
        state={
            "user_message": "不够详细，需要详细点4天3夜的攻略",
            "message_window": [
                {"role": "user", "message": "不够详细，需要详细点4天3夜的攻略"},
            ],
            "conversation_summary": "",
            "previous_final_message": "上一轮给出的是重庆旅游攻略简版，包含解放碑、洪崖洞和长江索道。",
            "final_message": "",
            "recent_run_briefs": [],
        },
        task_mode="general",
    )

    stable_background = dict(context_packet.get("stable_background") or {})
    topic_anchor = dict(stable_background.get("topic_anchor") or {})
    assert topic_anchor["source"] == "previous_final_message"
    assert "重庆旅游攻略简版" in topic_anchor["text"]


def test_direct_wait_node_should_build_synthetic_wait_plan() -> None:
    user_message = "先让我确认后再继续搜索课程"
    state = asyncio.run(
        direct_wait_node(
            {
                "user_message": user_message,
                "graph_metadata": {"control": _entry_contract_control(user_message)},
            }
        )
    )

    assert state["plan"] is not None
    assert state["plan"].language == "zh"
    assert [step.id for step in state["plan"].steps] == ["direct-wait-confirm", "direct-wait-execute"]
    assert state["plan"].steps[1].description == "执行原始任务"
    assert state["pending_interrupt"]["kind"] == "confirm"
    assert state["graph_metadata"]["control"]["entry_contract"]["source"]["user_message"] == user_message
    assert state["graph_metadata"]["control"]["entry_contract"]["task_mode"] == "research"
    assert state["plan"].steps[1].task_mode_hint == "research"
    assert state["plan"].steps[1].output_mode == "none"
    assert state["plan"].steps[1].artifact_policy == "default"


def test_atomic_action_node_should_build_single_step_plan() -> None:
    state = asyncio.run(
        atomic_action_node(
            {
                "user_message": "搜索 OpenAI 官网",
                "graph_metadata": {"control": _entry_contract_control("搜索 OpenAI 官网")},
            }
        )
    )

    assert state["plan"] is not None
    assert state["plan"].language == "zh"
    assert len(state["plan"].steps) == 1
    assert state["current_step_id"] == "atomic-action-step"
    assert state["graph_metadata"]["control"]["entry_contract"]["route"] == "atomic_action"
    assert state["plan"].steps[0].task_mode_hint == "research"
    assert state["plan"].steps[0].output_mode == "none"
    assert state["plan"].steps[0].artifact_policy == "default"


def test_direct_wait_node_should_preserve_english_working_language() -> None:
    state = asyncio.run(
        direct_wait_node(
            {
                "user_message": "Ask me for confirmation before you search the OpenAI docs",
                "graph_metadata": {
                    "control": _entry_contract_control("Ask me for confirmation before you search the OpenAI docs")
                },
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
    assert state["plan"].steps[0].task_mode_hint == StepTaskModeHint.CODING
    assert state["plan"].steps[0].artifact_policy == "allow_file_output"
    assert state["plan"].steps[0].output_mode == "file"
    # P3-一次性收口：编译后结构化字段必须保持 Enum 类型，禁止字符串回流持久化层。
    assert isinstance(state["plan"].steps[0].task_mode_hint, StepTaskModeHint)
    assert isinstance(state["plan"].steps[0].output_mode, StepOutputMode)
    assert isinstance(state["plan"].steps[0].artifact_policy, StepArtifactPolicy)


def test_compile_step_contracts_should_not_promote_plain_text_edit_to_coding() -> None:
    steps, issues, corrected_count = compile_step_contracts(
        steps=[
            Step(
                id="1",
                description="改写这段文案，让表达更自然",
                task_mode_hint="general",
                output_mode="none",
                artifact_policy="forbid_file_output",
            )
        ],
        user_message="帮我改写这段文案",
    )

    assert issues == []
    assert corrected_count == 0
    assert steps[0].task_mode_hint == StepTaskModeHint.GENERAL
    assert steps[0].artifact_policy == StepArtifactPolicy.FORBID_FILE_OUTPUT
    assert steps[0].output_mode == StepOutputMode.NONE


def test_filter_final_delivery_steps_should_drop_summary_like_general_step() -> None:
    steps, issues, corrected_count = compile_step_contracts(
        steps=[
            Step(
                id="1",
                description="整理 LangGraph human-in-the-loop 的常见实现模式，归纳为 5 条要点，并标注对应来源链接",
                task_mode_hint="general",
                output_mode="none",
                artifact_policy="forbid_file_output",
            )
        ],
        user_message="调研 LangGraph human-in-the-loop 常见实现模式，给我 5 条要点和来源链接",
    )
    filtered_steps, dropped_count = filter_final_delivery_steps(
        steps=steps,
        user_message="调研 LangGraph human-in-the-loop 常见实现模式，给我 5 条要点和来源链接",
    )

    assert corrected_count == 0
    assert issues == []
    assert filtered_steps == []
    assert dropped_count == 1


def test_compile_step_contracts_should_allow_execution_side_organization_step() -> None:
    steps, issues, corrected_count = compile_step_contracts(
        steps=[
            Step(
                id="1",
                description="整理已收集的事实并生成 Markdown 文件，供后续 summary 使用",
                task_mode_hint="coding",
                output_mode="file",
                artifact_policy="allow_file_output",
            )
        ],
        user_message="调研后把中间整理结果导出成 markdown 文件",
    )

    assert corrected_count == 0
    assert len(steps) == 1
    assert issues == []


def test_entry_compiler_should_not_treat_denied_file_output_as_file_task() -> None:
    contract = EntryCompiler().compile(
        user_message="先给我一版团队 AI 编程规范草稿预览，不需要写文件。",
        has_input_parts=False,
        has_active_plan=False,
        contextual_followup_anchor=False,
    )

    assert contract.route == "answer"
    assert contract.task_mode == StepTaskModeHint.GENERAL
    assert entry_reason_codes.SINGLE_FILE_READ_ATOMIC_ACTION not in contract.reason_codes


def test_create_or_reuse_plan_node_should_build_fallback_step_when_planned_task_returns_empty_steps() -> None:
    class _EmptyPlannerLLM:
        async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
            return {
                "content": json.dumps(
                    {
                        "message": "该任务为单步文件创建操作，无需多步骤规划。",
                        "goal": "",
                        "title": "",
                        "language": "zh",
                        "steps": [],
                    },
                    ensure_ascii=False,
                )
            }

    state = asyncio.run(
        create_or_reuse_plan_node(
            {
                "user_message": "创建文件 /home/ubuntu/workspace/p3-case-file/result.md，内容为 VERSION_1。",
                "graph_metadata": {"control": _entry_contract_control(
                    "创建文件 /home/ubuntu/workspace/p3-case-file/result.md，内容为 VERSION_1。"
                )},
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
            _EmptyPlannerLLM(),
        )
    )

    assert state["plan"] is not None
    assert state["plan"].status == ExecutionStatus.PENDING
    assert len(state["plan"].steps) == 1
    assert state["current_step_id"] == "fallback-execution-step"
    assert state["plan"].steps[0].task_mode_hint == StepTaskModeHint.CODING
    assert state["final_message"] == ""


def test_create_or_reuse_plan_node_should_build_non_file_fallback_step_for_research_task() -> None:
    class _EmptyPlannerLLM:
        async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
            return {
                "content": json.dumps(
                    {
                        "message": "该任务可以直接检索处理。",
                        "goal": "",
                        "title": "",
                        "language": "zh",
                        "steps": [],
                    },
                    ensure_ascii=False,
                )
            }

    state = asyncio.run(
        create_or_reuse_plan_node(
            {
                "user_message": "调研 LangGraph human-in-the-loop 常见实现模式，给我 5 条要点和来源链接",
                "graph_metadata": {"control": _entry_contract_control(
                    "调研 LangGraph human-in-the-loop 常见实现模式，给我 5 条要点和来源链接"
                )},
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
            _EmptyPlannerLLM(),
        )
    )

    assert state["plan"] is not None
    assert state["plan"].status == ExecutionStatus.PENDING
    assert len(state["plan"].steps) == 1
    assert state["current_step_id"] == "fallback-execution-step"
    assert state["plan"].steps[0].task_mode_hint == StepTaskModeHint.RESEARCH
    assert state["plan"].steps[0].output_mode == StepOutputMode.NONE
    assert state["plan"].steps[0].artifact_policy == StepArtifactPolicy.FORBID_FILE_OUTPUT


def test_create_or_reuse_plan_node_should_route_to_summary_when_final_delivery_steps_are_filtered() -> None:
    class _FinalDeliveryPlannerLLM:
        async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
            return {
                "content": json.dumps(
                    {
                        "message": "已生成调研计划。",
                        "goal": "调研 LangGraph human-in-the-loop 常见实现模式",
                        "title": "LangGraph HITL 调研",
                        "language": "zh",
                        "steps": [
                            {
                                "id": "1",
                                "description": "整理 LangGraph human-in-the-loop 的常见实现模式，归纳为 5 条要点，并标注对应来源链接",
                                "task_mode_hint": "general",
                                "output_mode": "none",
                                "artifact_policy": "forbid_file_output",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
            }

    state = asyncio.run(
        create_or_reuse_plan_node(
            {
                "user_message": "调研 LangGraph human-in-the-loop 常见实现模式，给我 5 条要点和来源链接",
                "graph_metadata": {"control": _entry_contract_control(
                    "调研 LangGraph human-in-the-loop 常见实现模式，给我 5 条要点和来源链接"
                )},
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
            _FinalDeliveryPlannerLLM(),
        )
    )

    assert state["plan"] is not None
    assert state["plan"].status == ExecutionStatus.PENDING
    assert state["plan"].steps == []
    assert state["current_step_id"] is None
    assert route_after_plan(state) == "summarize"


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
                                "output_mode": "none",
                                "artifact_policy": "forbid_file_output",
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
    assert step.output_mode == StepOutputMode.NONE
    assert step.artifact_policy == StepArtifactPolicy.FORBID_FILE_OUTPUT


def test_atomic_action_node_should_preserve_english_working_language() -> None:
    state = asyncio.run(
        atomic_action_node(
            {
                "user_message": "Read this file and summarize in English",
                "graph_metadata": {"control": _entry_contract_control("Read this file and summarize in English")},
            }
        )
    )

    assert state["plan"] is not None
    assert state["plan"].language == "en"
    assert state["plan"].message == "Entered atomic action path."
    assert state["plan"].steps[0].title == "Execute the user request"


def test_create_or_reuse_plan_node_should_stop_after_planning_for_plan_only_request() -> None:
    state = asyncio.run(
        create_or_reuse_plan_node(
            {
                "user_message": "帮我规划一个北京 3 天旅游安排，先不要执行，只给步骤。",
                "graph_metadata": {"control": _entry_contract_control(
                    "帮我规划一个北京 3 天旅游安排，先不要执行，只给步骤。"
                )},
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


def test_create_or_reuse_plan_node_should_emit_planner_message_stream_before_plan_events() -> None:
    captured_events = []

    async def _sink(event):
        captured_events.append(event)

    token = bind_live_event_sink(_sink)
    try:
        state = asyncio.run(
            create_or_reuse_plan_node(
                {
                    "session_id": "session-1",
                    "run_id": "run-1",
                    "user_message": "帮我规划一个北京 3 天旅游安排，先不要执行，只给步骤。",
                    "graph_metadata": {"control": _entry_contract_control(
                        "帮我规划一个北京 3 天旅游安排，先不要执行，只给步骤。"
                    )},
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
    finally:
        unbind_live_event_sink(token)

    assert isinstance(captured_events[0], TextStreamStartEvent)
    assert isinstance(captured_events[1], TextStreamDeltaEvent)
    assert isinstance(captured_events[2], TextStreamEndEvent)
    assert captured_events[0].stream_id == "run-1:final_message"
    assert captured_events[0].stage == "final"
    assert isinstance(captured_events[-2], PlanEvent)
    assert isinstance(captured_events[-1], MessageEvent)
    assert captured_events[-1].stage == "final"
    assert [event.type for event in state["emitted_events"]] == ["title", "plan", "message"]


def test_create_or_reuse_plan_node_should_preserve_extra_text_for_no_step_direct_answer() -> None:
    state = asyncio.run(
        create_or_reuse_plan_node(
            {
                "session_id": "session-1",
                "run_id": "run-1",
                "user_message": "再详细一点，给我 3 个典型使用场景。",
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
            _NoStepPlannerWithExtraTextLLM(),
        )
    )

    assert state["plan"] is not None
    assert state["plan"].status == ExecutionStatus.COMPLETED
    assert "三个典型使用场景" in state["final_message"]
    assert "多阶段智能体工作流" in state["final_message"]
    assert "我会详细说明" not in state["final_message"]
    assert state["emitted_events"][-1].message == state["final_message"]


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


def test_execute_step_with_prompt_should_return_search_snippet_feedback_before_optional_fetch() -> None:
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
    assert search_fetch_tool.invocations == ["search_web", "search_web"]
    assert llm.tool_feedback_payloads
    assert llm.tool_feedback_payloads[0]["search_evidence_quality"]["need_fetch"] is True
    assert llm.tool_feedback_payloads[0]["recommended_fetch_urls"] == ["https://example.com/article"]
    assert llm.tool_feedback_payloads[0]["search_evidence_summaries"][0]["snippet"] == "Short snippet"
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 2
    assert called_events[1].function_name == "search_web"
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


def test_execute_step_with_prompt_should_fail_web_reading_when_file_tool_is_requested() -> None:
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

    assert payload["success"] is False
    assert payload["summary"] == "当前步骤暂时未能完成：读取当前课程页面详情"
    assert payload["blockers"] == ["达到最大工具调用轮次，当前步骤仍未形成可交付结果。"]
    assert file_tool.invocations == []
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "read_file"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "网页读取任务" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_allow_general_file_tool_call_without_file_context_flag() -> None:
    llm = _ReadFileThenFinishLLM("/home/ubuntu/course-detail.md")
    file_tool = _ReadFileTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="将课程详情直接展示给用户",
                output_mode="none",
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
    assert file_tool.invocations == ["/home/ubuntu/course-detail.md"]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "read_file"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is True


def test_execute_step_with_prompt_should_allow_file_tools_for_inline_general_with_file_context() -> None:
    llm = _ReadFileThenFinishLLM("/home/ubuntu/course-detail.md")
    file_tool = _ReadFileTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="基于已有课程详情文件整理内联摘要",
                output_mode="none",
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


def test_execute_step_with_prompt_should_block_search_in_general_task_mode() -> None:
    llm = _SearchThenFinishLLM()
    search_tool = _SearchFetchTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="基于已有信息整理最终攻略",
                output_mode="none",
                artifact_policy="default",
            ),
            runtime_tools=[search_tool],
            task_mode="general",
            user_content=[{"type": "text", "text": "请基于已有信息整理最终攻略"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["summary"] == "已完成当前步骤"
    assert search_tool.invocations == []
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "search_web"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "任务模式 general 不允许调用工具" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_fail_web_reading_when_search_evidence_is_insufficient() -> None:
    llm = _SearchThenFinishLLM()
    search_tool = _SearchFetchTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="基于已有页面信息整理最终课程详情",
                output_mode="none",
                artifact_policy="default",
            ),
            runtime_tools=[search_tool],
            task_mode="web_reading",
            user_content=[{"type": "text", "text": "请整理最终课程详情"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is False
    assert payload["summary"] == "当前步骤暂时未能完成：基于已有页面信息整理最终课程详情"
    assert payload["blockers"] == ["达到最大工具调用轮次，当前步骤仍未形成可交付结果。"]
    assert search_tool.invocations == ["search_web"]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "search_web"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is True


def test_execute_step_with_prompt_should_not_block_shell_without_legacy_delivery_semantics() -> None:
    llm = _ShellThenFinishLLM()
    shell_tool = _ShellTool()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="基于已有上下文输出最终结论",
                output_mode="none",
                artifact_policy="default",
            ),
            runtime_tools=[shell_tool],
            task_mode="general",
            user_content=[{"type": "text", "text": "请输出最终结论"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["summary"] == "已完成当前步骤"
    assert shell_tool.invoked == 1
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "shell_execute"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is True


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


def test_atomic_action_step_should_allow_search_without_legacy_delivery_contract() -> None:
    llm = _SearchThenFinishLLM()
    search_tool = _SearchTool()
    state = asyncio.run(
        atomic_action_node(
            {
                "user_message": "搜索 OpenAI 官网",
                "graph_metadata": {"control": _entry_contract_control("搜索 OpenAI 官网")},
            }
        )
    )

    next_state = asyncio.run(execute_step_node(state, llm, runtime_tools=[search_tool]))

    assert search_tool.invoked == 1
    assert next_state["last_executed_step"].status == ExecutionStatus.COMPLETED
    assert next_state["final_message"] == ""
    assert next_state["last_executed_step"].outcome is not None
    assert next_state["last_executed_step"].outcome.summary == "已完成当前步骤"


def test_direct_wait_execute_step_should_allow_search_without_legacy_delivery_contract() -> None:
    llm = _SearchThenFinishLLM()
    search_tool = _SearchTool()
    state = asyncio.run(
        direct_wait_node(
            {
                "user_message": "先让我确认后再继续搜索课程",
                "graph_metadata": {"control": _entry_contract_control("先让我确认后再继续搜索课程")},
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
    assert "direct_wait_original_task_executed" not in next_state["graph_metadata"]["control"]
    assert next_state["final_message"] == ""
    assert next_state["last_executed_step"].outcome is not None
    assert next_state["last_executed_step"].outcome.summary == "已完成当前步骤"


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
                output_mode="none",
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
    assert next_state["final_message"] == ""


def test_execute_step_node_should_store_attachment_delivery_preference_in_outcome_and_working_memory() -> None:
    llm = _NoAttachmentPreferenceLLM()
    state = asyncio.run(
        atomic_action_node(
            {
                "user_message": "创建文件 /home/ubuntu/workspace/p3-artifact/result.md，内容为 VERSION_1。这一步不要作为最终附件返回。",
                "graph_metadata": {
                    "control": _entry_contract_control(
                        "创建文件 /home/ubuntu/workspace/p3-artifact/result.md，内容为 VERSION_1。这一步不要作为最终附件返回。"
                    )
                },
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
                output_mode="none",
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
    assert next_state["plan"].steps[0].outcome.facts_learned == ["候选课程 A", "候选课程 B", "候选课程 C"]
    assert next_state["final_message"] == ""


def test_execute_step_node_should_not_store_step_level_body_for_non_summary_step() -> None:
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
    assert next_state["last_executed_step"].outcome.summary == "已完成搜索步骤"
    assert next_state["final_message"] == ""


def test_execute_step_node_should_not_overwrite_existing_final_message_with_step_summary() -> None:
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
        "final_message": "已有最终正文",
        "thread_id": "thread-1",
        "conversation_summary": "",
    }

    next_state = asyncio.run(execute_step_node(state, llm, runtime_tools=[]))

    assert next_state["last_executed_step"].outcome is not None
    assert next_state["last_executed_step"].outcome.summary == "已完成搜索步骤"
    assert next_state["final_message"] == "已有最终正文"


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
    assert payload["next_hint"] == "请改写搜索主题描述、缩小范围，或改用 fetch_page / 文件读取继续。"
