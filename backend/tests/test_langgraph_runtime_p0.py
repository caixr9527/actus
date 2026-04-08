import asyncio
import json

from app.domain.models import ExecutionStatus, Step, ToolEventStatus, ToolResult
from app.domain.services.tools.base import BaseTool, tool
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.nodes import (
    direct_answer_node,
    direct_execute_node,
    direct_wait_node,
    entry_router_node,
)
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.language_checker import (
    build_direct_path_copy,
    infer_working_language_from_message,
)
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.tools import (
    classify_step_task_mode,
    execute_step_with_prompt,
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


def test_classify_step_task_mode_should_use_artifact_and_command_signals() -> None:
    assert classify_step_task_mode(
        Step(description="执行 `pytest backend/tests/test_run_engine_selector.py -q` 并修复失败")
    ) == "coding"
    assert classify_step_task_mode(
        Step(description="读取 /tmp/backend.log 并整理错误摘要")
    ) == "file_processing"


def test_classify_step_task_mode_should_prefer_web_reading_over_browser_interaction_for_page_reading() -> None:
    assert classify_step_task_mode(
        Step(description="打开 https://openai.com/blog 阅读页面内容并提炼要点")
    ) == "web_reading"
    assert classify_step_task_mode(
        Step(description="登录后台后点击提交按钮并填写表单")
    ) == "browser_interaction"


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
    state = asyncio.run(
        direct_wait_node(
            {
                "user_message": "先让我确认后再继续搜索课程",
                "graph_metadata": {},
            }
        )
    )

    assert state["plan"] is not None
    assert state["plan"].language == "zh"
    assert [step.id for step in state["plan"].steps] == ["direct-wait-confirm", "direct-wait-execute"]
    assert state["pending_interrupt"]["kind"] == "confirm"
    assert state["graph_metadata"]["control"]["skip_replan_when_plan_finished"] is True


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


def test_execute_step_with_prompt_should_block_repeated_search_before_fetch_page() -> None:
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
    assert search_fetch_tool.invocations == ["search_web"]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 2
    assert called_events[1].function_name == "search_web"
    assert called_events[1].function_result is not None
    assert called_events[1].function_result.success is False
    assert "优先对搜索结果中的 URL 调用 fetch_page" in str(called_events[1].function_result.message or "")


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
    assert browser_tool.invocations == ["browser_view", "browser_scroll_down", "browser_view"]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 3
