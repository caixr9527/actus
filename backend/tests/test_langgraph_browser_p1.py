import asyncio
import json

from app.domain.models import Step, ToolEventStatus, ToolResult
from app.domain.services.tools.base import BaseTool, tool
from app.infrastructure.runtime.langgraph.graphs.planner_react.tools import execute_step_with_prompt


class _BrowserPriorityTool(BaseTool):
    name = "browser"

    def __init__(self, *, main_content_success: bool = True) -> None:
        super().__init__()
        self.invocations: list[str] = []
        self._main_content_success = main_content_success

    @tool(
        name="browser_extract_main_content",
        description="extract main content",
        parameters={},
        required=[],
    )
    async def browser_extract_main_content(self):
        self.invocations.append("browser_extract_main_content")
        if not self._main_content_success:
            return ToolResult(success=False, message="no main content")
        return ToolResult(success=True, data={"url": "https://example.com/article", "title": "Article"})

    @tool(
        name="browser_find_actionable_elements",
        description="find actionable elements",
        parameters={},
        required=[],
    )
    async def browser_find_actionable_elements(self):
        self.invocations.append("browser_find_actionable_elements")
        return ToolResult(success=True, data={"url": "https://example.com/form", "title": "Form"})

    @tool(
        name="browser_view",
        description="view page",
        parameters={},
        required=[],
    )
    async def browser_view(self):
        self.invocations.append("browser_view")
        return ToolResult(success=True, data={"url": "https://example.com/fallback", "title": "Fallback"})


class _BrowserRoutingTool(BaseTool):
    name = "browser"

    def __init__(
            self,
            *,
            page_type: str = "document",
            main_content_success: bool = True,
            link_match_fail_queries: set[str] | None = None,
    ) -> None:
        super().__init__()
        self.invocations: list[str] = []
        self.link_queries: list[str] = []
        self._page_type = page_type
        self._main_content_success = main_content_success
        self._link_match_fail_queries = {item.strip().lower() for item in (link_match_fail_queries or set()) if item.strip()}

    @tool(
        name="browser_read_current_page_structured",
        description="structured page",
        parameters={},
        required=[],
    )
    async def browser_read_current_page_structured(self):
        self.invocations.append("browser_read_current_page_structured")
        return ToolResult(
            success=True,
            data={
                "url": "https://example.com/current",
                "title": "Current Page",
                "page_type": self._page_type,
                "cards": [
                    {
                        "title": "Execution Model",
                        "summary": "Execution details",
                        "url": "https://example.com/docs/execution",
                    }
                ] if self._page_type in {"listing", "search_results"} else [],
            },
        )

    @tool(
        name="browser_extract_main_content",
        description="main content",
        parameters={},
        required=[],
    )
    async def browser_extract_main_content(self):
        self.invocations.append("browser_extract_main_content")
        if not self._main_content_success:
            return ToolResult(success=False, message="no main content")
        return ToolResult(
            success=True,
            data={
                "url": "https://example.com/current",
                "title": "Current Page",
                "page_type": self._page_type,
                "content": "main content",
            },
        )

    @tool(
        name="browser_extract_cards",
        description="extract cards",
        parameters={},
        required=[],
    )
    async def browser_extract_cards(self):
        self.invocations.append("browser_extract_cards")
        return ToolResult(
            success=True,
            data={
                "url": "https://example.com/current",
                "title": "Current Page",
                "page_type": self._page_type,
                "cards": [
                    {
                        "title": "Execution Model",
                        "summary": "Execution details",
                        "url": "https://example.com/docs/execution",
                    }
                ],
                "total_cards": 1,
            },
        )

    @tool(
        name="browser_find_link_by_text",
        description="find link by text",
        parameters={"text": {"type": "string"}},
        required=["text"],
    )
    async def browser_find_link_by_text(self, text: str):
        self.invocations.append("browser_find_link_by_text")
        self.link_queries.append(text)
        if text.strip().lower() in self._link_match_fail_queries:
            return ToolResult(success=False, message=f"未找到 {text}")
        return ToolResult(
            success=True,
            data={
                "query": text,
                "matched_text": "Execution Model",
                "url": "https://example.com/docs/execution",
                "index": 0,
                "selector": "[data-manus-id='manus-element-0']",
            },
        )

    @tool(
        name="browser_find_actionable_elements",
        description="find actionable elements",
        parameters={},
        required=[],
    )
    async def browser_find_actionable_elements(self):
        self.invocations.append("browser_find_actionable_elements")
        return ToolResult(
            success=True,
            data={
                "url": "https://example.com/form",
                "title": "Form",
                "page_type": self._page_type,
                "elements": [
                    {
                        "index": 0,
                        "tag": "button",
                        "text": "提交",
                        "role": "button",
                        "selector": "[data-manus-id='manus-element-0']",
                    }
                ],
            },
        )

    @tool(
        name="browser_view",
        description="view page",
        parameters={},
        required=[],
    )
    async def browser_view(self):
        self.invocations.append("browser_view")
        return ToolResult(success=True, data={"url": "https://example.com/fallback", "title": "Fallback"})

    @tool(
        name="browser_click",
        description="click",
        parameters={"index": {"type": "integer"}},
        required=["index"],
    )
    async def browser_click(self, index: int):
        self.invocations.append("browser_click")
        return ToolResult(success=True, data={"url": f"https://example.com/click/{index}", "title": "Clicked"})

    @tool(
        name="browser_navigate",
        description="navigate",
        parameters={"url": {"type": "string"}},
        required=["url"],
    )
    async def browser_navigate(self, url: str):
        self.invocations.append("browser_navigate")
        return ToolResult(success=True, data={"url": url, "title": "Navigated"})


class _WebReadingPriorityLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-view",
                        "function": {"name": "browser_view", "arguments": "{}"},
                    },
                    {
                        "id": "call-main",
                        "function": {"name": "browser_extract_main_content", "arguments": "{}"},
                    },
                ],
            }
        return {
            "content": json.dumps({"success": True, "result": "已完成阅读", "attachments": []}, ensure_ascii=False),
        }


class _BrowserInteractionPriorityLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-click",
                        "function": {"name": "browser_view", "arguments": "{}"},
                    },
                    {
                        "id": "call-actionables",
                        "function": {"name": "browser_find_actionable_elements", "arguments": "{}"},
                    },
                ],
            }
        return {
            "content": json.dumps({"success": True, "result": "已完成交互准备", "attachments": []}, ensure_ascii=False),
        }


class _BrowserDegradeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-main",
                        "function": {"name": "browser_extract_main_content", "arguments": "{}"},
                    }
                ],
            }
        if self.calls == 2:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-main-again",
                        "function": {"name": "browser_extract_main_content", "arguments": "{}"},
                    },
                    {
                        "id": "call-view",
                        "function": {"name": "browser_view", "arguments": "{}"},
                    },
                ],
            }
        return {
            "content": json.dumps({"success": True, "result": "已降级到兜底浏览器观察", "attachments": []}, ensure_ascii=False),
        }


class _AtomicOnlyWebReadingLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-view-first",
                        "function": {"name": "browser_view", "arguments": "{}"},
                    }
                ],
            }
        if self.calls == 2:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-structured",
                        "function": {"name": "browser_read_current_page_structured", "arguments": "{}"},
                    }
                ],
            }
        if self.calls == 3:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-main-content",
                        "function": {"name": "browser_extract_main_content", "arguments": "{}"},
                    }
                ],
            }
        return {
            "content": json.dumps({"success": True, "result": "已按固定路径完成阅读", "attachments": []}, ensure_ascii=False),
        }


class _BrowserListingRouteLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-structured",
                        "function": {"name": "browser_read_current_page_structured", "arguments": "{}"},
                    }
                ],
            }
        if self.calls == 2:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-view",
                        "function": {"name": "browser_view", "arguments": "{}"},
                    },
                    {
                        "id": "call-cards",
                        "function": {"name": "browser_extract_cards", "arguments": "{}"},
                    },
                ],
            }
        if self.calls == 3:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-click",
                        "function": {"name": "browser_click", "arguments": json.dumps({"index": 0}, ensure_ascii=False)},
                    },
                    {
                        "id": "call-find-link",
                        "function": {
                            "name": "browser_find_link_by_text",
                            "arguments": json.dumps({"text": "Execution Model"}, ensure_ascii=False),
                        },
                    },
                ],
            }
        return {
            "content": json.dumps({"success": True, "result": "已按列表页固定路径完成定位", "attachments": []}, ensure_ascii=False),
        }


class _BrowserListingRetryLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-structured",
                        "function": {"name": "browser_read_current_page_structured", "arguments": "{}"},
                    }
                ],
            }
        if self.calls == 2:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-cards",
                        "function": {"name": "browser_extract_cards", "arguments": "{}"},
                    }
                ],
            }
        if self.calls == 3:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-find-link-first",
                        "function": {
                            "name": "browser_find_link_by_text",
                            "arguments": json.dumps({"text": "No Match"}, ensure_ascii=False),
                        },
                    }
                ],
            }
        if self.calls == 4:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-find-link-second",
                        "function": {
                            "name": "browser_find_link_by_text",
                            "arguments": json.dumps({"text": "Execution Model"}, ensure_ascii=False),
                        },
                    }
                ],
            }
        return {
            "content": json.dumps({"success": True, "result": "已更换关键词完成定位", "attachments": []}, ensure_ascii=False),
        }


class _BrowserListingClickPriorityLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-structured",
                        "function": {"name": "browser_read_current_page_structured", "arguments": "{}"},
                    }
                ],
            }
        if self.calls == 2:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-cards",
                        "function": {"name": "browser_extract_cards", "arguments": "{}"},
                    }
                ],
            }
        if self.calls == 3:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-find-link",
                        "function": {
                            "name": "browser_find_link_by_text",
                            "arguments": json.dumps({"text": "Execution Model"}, ensure_ascii=False),
                        },
                    }
                ],
            }
        if self.calls == 4:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-navigate",
                        "function": {
                            "name": "browser_navigate",
                            "arguments": json.dumps({"url": "https://example.com/docs/execution"}, ensure_ascii=False),
                        },
                    },
                    {
                        "id": "call-click",
                        "function": {
                            "name": "browser_click",
                            "arguments": json.dumps({"index": 0}, ensure_ascii=False),
                        },
                    },
                ],
            }
        return {
            "content": json.dumps({"success": True, "result": "已优先点击进入详情", "attachments": []}, ensure_ascii=False),
        }


class _BrowserListingWrongClickLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-structured",
                        "function": {"name": "browser_read_current_page_structured", "arguments": "{}"},
                    }
                ],
            }
        if self.calls == 2:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-cards",
                        "function": {"name": "browser_extract_cards", "arguments": "{}"},
                    }
                ],
            }
        if self.calls == 3:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-find-link",
                        "function": {
                            "name": "browser_find_link_by_text",
                            "arguments": json.dumps({"text": "Execution Model"}, ensure_ascii=False),
                        },
                    }
                ],
            }
        if self.calls == 4:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-wrong-click",
                        "function": {
                            "name": "browser_click",
                            "arguments": json.dumps({"index": 1}, ensure_ascii=False),
                        },
                    }
                ],
            }
        if self.calls == 5:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-correct-click",
                        "function": {
                            "name": "browser_click",
                            "arguments": json.dumps({"index": 0}, ensure_ascii=False),
                        },
                    }
                ],
            }
        return {
            "content": json.dumps({"success": True, "result": "已按匹配目标完成点击", "attachments": []}, ensure_ascii=False),
        }


def test_execute_step_with_prompt_should_prefer_browser_high_level_tool_for_web_reading() -> None:
    llm = _WebReadingPriorityLLM()
    tool = _BrowserPriorityTool()

    payload, _ = asyncio.run(
        execute_step_with_prompt(
            llm=llm,
            step=Step(description="阅读当前页面正文并提炼要点"),
            runtime_tools=[tool],
            task_mode="web_reading",
            user_content=[{"type": "text", "text": "请阅读当前页面正文并提炼要点"}],
        )
    )

    assert payload["success"] is True
    assert tool.invocations == ["browser_extract_main_content"]


def test_execute_step_with_prompt_should_prefer_actionable_scan_for_browser_interaction() -> None:
    llm = _BrowserInteractionPriorityLLM()
    tool = _BrowserPriorityTool()

    payload, _ = asyncio.run(
        execute_step_with_prompt(
            llm=llm,
            step=Step(description="查看当前表单并确认可点击元素"),
            runtime_tools=[tool],
            task_mode="browser_interaction",
            user_content=[{"type": "text", "text": "请查看当前表单并确认可点击元素"}],
        )
    )

    assert payload["success"] is True
    assert tool.invocations == ["browser_find_actionable_elements"]


def test_execute_step_with_prompt_should_degrade_to_atomic_browser_tool_after_high_level_failure() -> None:
    llm = _BrowserDegradeLLM()
    tool = _BrowserPriorityTool(main_content_success=False)

    payload, events = asyncio.run(
        execute_step_with_prompt(
            llm=llm,
            step=Step(description="阅读当前页面内容"),
            runtime_tools=[tool],
            task_mode="web_reading",
            user_content=[{"type": "text", "text": "请阅读当前页面内容"}],
        )
    )

    assert payload["success"] is True
    assert tool.invocations == ["browser_extract_main_content", "browser_view"]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.data["degrade_reason"] == "browser_extract_main_content_failed"


def test_execute_step_with_prompt_should_block_atomic_browser_tool_before_high_level_reading_path() -> None:
    llm = _AtomicOnlyWebReadingLLM()
    tool = _BrowserRoutingTool(page_type="document")

    payload, events = asyncio.run(
        execute_step_with_prompt(
            llm=llm,
            step=Step(description="阅读当前页面正文并提炼要点"),
            runtime_tools=[tool],
            task_mode="web_reading",
            user_content=[{"type": "text", "text": "请阅读当前页面正文并提炼要点"}],
        )
    )

    assert payload["success"] is True
    assert tool.invocations == ["browser_read_current_page_structured", "browser_extract_main_content"]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert called_events[0].function_name == "browser_view"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "请先调用 browser_read_current_page_structured" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_follow_listing_page_fixed_path() -> None:
    llm = _BrowserListingRouteLLM()
    tool = _BrowserRoutingTool(page_type="listing")

    payload, events = asyncio.run(
        execute_step_with_prompt(
            llm=llm,
            step=Step(description="从当前列表页中定位 Execution Model 详情"),
            runtime_tools=[tool],
            task_mode="web_reading",
            user_content=[{"type": "text", "text": "请从当前列表页中定位 Execution Model 详情"}],
        )
    )

    assert payload["success"] is True
    assert tool.invocations == [
        "browser_read_current_page_structured",
        "browser_extract_cards",
        "browser_find_link_by_text",
    ]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert [event.function_name for event in called_events] == [
        "browser_read_current_page_structured",
        "browser_extract_cards",
        "browser_find_link_by_text",
    ]


def test_execute_step_with_prompt_should_allow_find_link_retry_with_new_query() -> None:
    llm = _BrowserListingRetryLLM()
    tool = _BrowserRoutingTool(
        page_type="listing",
        link_match_fail_queries={"No Match"},
    )

    payload, events = asyncio.run(
        execute_step_with_prompt(
            llm=llm,
            step=Step(description="从当前列表页中重试不同关键词定位详情"),
            runtime_tools=[tool],
            task_mode="web_reading",
            user_content=[{"type": "text", "text": "请从当前列表页中重试不同关键词定位详情"}],
        )
    )

    assert payload["success"] is True
    assert tool.invocations == [
        "browser_read_current_page_structured",
        "browser_extract_cards",
        "browser_find_link_by_text",
        "browser_find_link_by_text",
    ]
    assert tool.link_queries == ["No Match", "Execution Model"]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert called_events[2].function_result is not None
    assert called_events[2].function_result.success is False
    assert called_events[3].function_result is not None
    assert called_events[3].function_result.success is True


def test_execute_step_with_prompt_should_prefer_click_after_listing_link_match() -> None:
    llm = _BrowserListingClickPriorityLLM()
    tool = _BrowserRoutingTool(page_type="listing")

    payload, events = asyncio.run(
        execute_step_with_prompt(
            llm=llm,
            step=Step(description="从当前列表页中点击进入 Execution Model 详情"),
            runtime_tools=[tool],
            task_mode="web_reading",
            user_content=[{"type": "text", "text": "请从当前列表页中点击进入 Execution Model 详情"}],
        )
    )

    assert payload["success"] is True
    assert tool.invocations == [
        "browser_read_current_page_structured",
        "browser_extract_cards",
        "browser_find_link_by_text",
        "browser_click",
    ]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert [event.function_name for event in called_events] == [
        "browser_read_current_page_structured",
        "browser_extract_cards",
        "browser_find_link_by_text",
        "browser_click",
    ]


def test_execute_step_with_prompt_should_block_wrong_click_index_after_listing_link_match() -> None:
    llm = _BrowserListingWrongClickLLM()
    tool = _BrowserRoutingTool(page_type="listing")

    payload, events = asyncio.run(
        execute_step_with_prompt(
            llm=llm,
            step=Step(description="从当前列表页中点击匹配到的详情项"),
            runtime_tools=[tool],
            task_mode="web_reading",
            max_tool_iterations=6,
            user_content=[{"type": "text", "text": "请从当前列表页中点击匹配到的详情项"}],
        )
    )

    assert payload["success"] is True
    assert tool.invocations == [
        "browser_read_current_page_structured",
        "browser_extract_cards",
        "browser_find_link_by_text",
        "browser_click",
    ]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert [event.function_name for event in called_events] == [
        "browser_read_current_page_structured",
        "browser_extract_cards",
        "browser_find_link_by_text",
        "browser_click",
        "browser_click",
    ]
    assert called_events[3].function_result is not None
    assert called_events[3].function_result.success is False
    assert "只允许点击刚刚匹配到的目标" in str(called_events[3].function_result.message or "")
    assert called_events[4].function_result is not None
    assert called_events[4].function_result.success is True
