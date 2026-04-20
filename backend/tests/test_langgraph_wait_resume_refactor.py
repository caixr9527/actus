import asyncio
import json

from app.domain.models import ExecutionStatus, Plan, Step, ToolEventStatus, ToolResult
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.tools.base import BaseTool, tool
from app.domain.services.tools.message import MessageTool
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes import (
    direct_wait_node,
    execute_step_node as _execute_step_node,
    wait_for_human_node,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.routing import (
    route_after_execute,
    route_after_wait,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.tools import (
    execute_step_with_prompt,
)


_TEST_RUNTIME_CONTEXT_SERVICE = RuntimeContextService()


async def execute_step_node(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return await _execute_step_node(
        *args,
        **kwargs,
    )


class _DummyTool(BaseTool):
    name = "dummy"

    @tool(
        name="dummy_tool",
        description="dummy tool",
        parameters={},
        required=[],
    )
    async def dummy_tool(self):
        return {"success": True, "data": {"ok": True}}


class _WriteFileTool(BaseTool):
    name = "file"

    @tool(
        name="write_file",
        description="write file",
        parameters={
            "filepath": {"type": "string"},
            "content": {"type": "string"},
        },
        required=["filepath", "content"],
    )
    async def write_file(self, filepath: str, content: str):
        return {"success": True, "data": {"filepath": filepath, "content": content}}


class _SearchTool(BaseTool):
    name = "search"

    @tool(
        name="search_web",
        description="search web",
        parameters={"query": {"type": "string"}},
        required=["query"],
    )
    async def search_web(self, query: str):
        return ToolResult(success=True, data={"query": query, "results": [{"url": "https://example.com"}]})


class _NonInterruptMessageTool(BaseTool):
    name = "message"

    @tool(
        name="message_ask_user",
        description="ask user without interrupt",
        parameters={"text": {"type": "string"}},
        required=["text"],
    )
    async def message_ask_user(self, text: str):
        return ToolResult(success=True, message=text, data={"ack": True})


class _FinalResultLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        return {
            "role": "assistant",
            "content": '{"success": true, "result": "当前步骤已完成", "attachments": []}',
        }


class _IllegalAskUserLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "message_ask_user",
                            "arguments": '{"text":"请选择一门课程","kind":"select","options":[{"label":"课程A","resume_value":"a"}]}',
                        },
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": '{"success": true, "result": "当前步骤只完成搜索，不提前让用户选择", "attachments": []}',
        }


class _AllowedAskUserLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-allow-ask-user",
                    "type": "function",
                    "function": {
                        "name": "message_ask_user",
                        "arguments": (
                            '{"text":"请确认是否开始搜索课程","kind":"confirm",'
                            '"confirm_label":"继续","cancel_label":"取消"}'
                        ),
                    },
                }
            ],
        }


class _WriteFileAttemptLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-write-file",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"filepath":"/tmp/course.txt","content":"课程列表"}',
                        },
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": '{"success": true, "result": "当前步骤已改为直接返回文本结果", "attachments": []}',
        }


class _ReadOnlyWriteAttemptLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-write-file-read-only",
                        "type": "function",
                        "function": {
                            "name": "write_file",
                            "arguments": '{"filepath":"/tmp/course.txt","content":"课程列表"}',
                        },
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": '{"success": true, "result": "已改为直接读取并输出结果", "attachments": []}',
        }


class _HumanWaitSearchDriftLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-search",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": json.dumps({"query": "imooc ai agent"}, ensure_ascii=False),
                        },
                    }
                ],
            }
        return {
            "role": "assistant",
            "content": '{"success": true, "result": "已直接结束等待步骤", "attachments": []}',
        }


class _HumanWaitAskUserWithoutInterruptLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-ask-user-no-interrupt",
                    "type": "function",
                    "function": {
                        "name": "message_ask_user",
                        "arguments": json.dumps({"text": "请选择课程"}, ensure_ascii=False),
                    },
                }
            ],
        }


class _ReasoningToolReplayLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": "",
                "reasoning_content": "这段推理字段必须在下一轮保留",
                "tool_calls": [
                    {
                        "id": "tool-msg-1",
                        "call_id": "tool-call-1",
                        "type": "function",
                        "function": {
                            "name": "search_web",
                            "arguments": json.dumps({"query": "imooc ai agent"}, ensure_ascii=False),
                        },
                    }
                ],
            }

        assistant_message = next(item for item in messages if item.get("role") == "assistant")
        tool_message = next(item for item in messages if item.get("role") == "tool")
        assert assistant_message["reasoning_content"] == "这段推理字段必须在下一轮保留"
        assert assistant_message["tool_calls"][0]["call_id"] == "tool-call-1"
        assert tool_message["tool_call_id"] == "tool-msg-1"
        assert tool_message["call_id"] == "tool-call-1"
        return {
            "role": "assistant",
            "content": '{"success": true, "result": "已完成课程检索", "attachments": []}',
        }


def test_execute_step_with_prompt_should_return_final_result_without_redundant_iterations() -> None:
    llm = _FinalResultLLM()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="搜索课程"),
            runtime_tools=[_DummyTool()],
            max_tool_iterations=5,
            on_tool_event=None,
            user_content=[{"type": "text", "text": "只完成当前步骤"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["result"] == "当前步骤已完成"
    assert events == []
    assert llm.calls == 1


def test_execute_step_with_prompt_should_block_ask_user_for_non_wait_step() -> None:
    llm = _IllegalAskUserLLM()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="搜索并记录 AI 课程信息"),
            runtime_tools=[MessageTool()],
            max_tool_iterations=5,
            on_tool_event=None,
            user_content=[{"type": "text", "text": "只完成搜索步骤"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["result"] == "当前步骤只完成搜索，不提前让用户选择"
    assert llm.calls == 2
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "message_ask_user"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "当前步骤不允许向用户提问" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_allow_ask_user_for_planner_wait_step() -> None:
    llm = _AllowedAskUserLLM()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="向用户请求确认是否开始使用搜索工具查找课程。"),
            runtime_tools=[MessageTool()],
            max_tool_iterations=5,
            on_tool_event=None,
            user_content=[{"type": "text", "text": "仅执行当前确认步骤"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is True
    assert payload["interrupt_request"]["kind"] == "confirm"
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "message_ask_user"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is True


def test_execute_step_with_prompt_should_allow_ask_user_for_structured_wait_hint() -> None:
    llm = _AllowedAskUserLLM()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="根据用户选择继续处理课程详情。",
                task_mode_hint="human_wait",
            ),
            runtime_tools=[MessageTool()],
            max_tool_iterations=5,
            on_tool_event=None,
            user_content=[{"type": "text", "text": "仅执行当前等待步骤"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is True
    assert payload["interrupt_request"]["kind"] == "confirm"
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "message_ask_user"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is True


def test_execute_step_with_prompt_should_block_write_file_when_artifact_policy_forbids_file_output() -> None:
    llm = _WriteFileAttemptLLM()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="搜索并整理课程信息。",
                task_mode_hint="research",
                output_mode="none",
                artifact_policy="forbid_file_output",
            ),
            runtime_tools=[_WriteFileTool()],
            max_tool_iterations=5,
            task_mode="research",
            on_tool_event=None,
            user_content=[{"type": "text", "text": "只返回文本结论，不要写文件"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["result"] == "当前步骤已改为直接返回文本结果"
    assert llm.calls == 2
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "write_file"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "结构化产物策略禁止文件产出" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_block_write_file_for_final_inline_delivery_by_default() -> None:
    llm = _WriteFileAttemptLLM()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="直接输出最终答复",
                task_mode_hint="general",
                output_mode="inline",
                delivery_role="final",
                delivery_context_state="ready",
                artifact_policy="default",
            ),
            runtime_tools=[_WriteFileTool()],
            max_tool_iterations=5,
            task_mode="general",
            on_tool_event=None,
            user_content=[{"type": "text", "text": "请直接给我最终答案"}],
            user_message="请直接给我最终答案，不要写文件",
            has_available_file_context=True,
        )

    payload, events = asyncio.run(_run())

    assert payload["result"] == "当前步骤已改为直接返回文本结果"
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "write_file"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "最终内联交付" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_allow_write_file_when_user_explicitly_requests_file_delivery() -> None:
    llm = _WriteFileAttemptLLM()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="整理最终结果并交付",
                task_mode_hint="general",
                output_mode="inline",
                delivery_role="final",
                delivery_context_state="ready",
                artifact_policy="default",
            ),
            runtime_tools=[_WriteFileTool()],
            max_tool_iterations=5,
            task_mode="general",
            on_tool_event=None,
            user_content=[{"type": "text", "text": "把最终内容保存到文件"}],
            user_message="请把最终结果保存到 result.md 文件并返回",
            has_available_file_context=True,
        )

    payload, events = asyncio.run(_run())

    assert payload["result"] == "当前步骤已改为直接返回文本结果"
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "write_file"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is True


def test_execute_step_with_prompt_should_block_write_file_for_read_only_file_request() -> None:
    llm = _ReadOnlyWriteAttemptLLM()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="读取 /tmp/course.txt 内容并返回结果，不要改文件。",
                task_mode_hint="file_processing",
                output_mode="inline",
            ),
            runtime_tools=[_WriteFileTool()],
            max_tool_iterations=5,
            task_mode="file_processing",
            on_tool_event=None,
            user_content=[{"type": "text", "text": "只读取文件内容，不要写入或修改"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["result"] == "已改为直接读取并输出结果"
    assert llm.calls == 2
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "write_file"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "只读文件请求" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_block_non_interrupt_tool_for_human_wait_step() -> None:
    llm = _HumanWaitSearchDriftLLM()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="等待用户从候选课程中选择一门",
                task_mode_hint="human_wait",
            ),
            runtime_tools=[_SearchTool(), MessageTool()],
            max_tool_iterations=3,
            task_mode="human_wait",
            on_tool_event=None,
            user_content=[{"type": "text", "text": "仅执行当前等待步骤"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is False
    assert payload["blockers"] == ["当前步骤需要等待用户确认/选择，但尚未成功发起等待请求。"]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 1
    assert called_events[0].function_name == "search_web"
    assert called_events[0].function_result is not None
    assert called_events[0].function_result.success is False
    assert "只允许调用 message_ask_user" in str(called_events[0].function_result.message or "")


def test_execute_step_with_prompt_should_fail_human_wait_step_without_interrupt_even_if_ask_user_called() -> None:
    llm = _HumanWaitAskUserWithoutInterruptLLM()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(
                description="等待用户选择课程",
                task_mode_hint="human_wait",
            ),
            runtime_tools=[_NonInterruptMessageTool()],
            max_tool_iterations=2,
            task_mode="human_wait",
            on_tool_event=None,
            user_content=[{"type": "text", "text": "仅执行当前等待步骤"}],
        )

    payload, events = asyncio.run(_run())

    assert payload["success"] is False
    assert payload["blockers"] == ["当前步骤需要等待用户确认/选择，但尚未成功发起等待请求。"]
    called_events = [event for event in events if event.status == ToolEventStatus.CALLED]
    assert len(called_events) == 2
    assert all(event.function_name == "message_ask_user" for event in called_events)
    assert all(event.function_result is not None and event.function_result.success is True for event in called_events)


def test_execute_step_with_prompt_should_preserve_assistant_reasoning_fields_during_tool_replay() -> None:
    llm = _ReasoningToolReplayLLM()

    async def _run():
        return await execute_step_with_prompt(
            llm=llm,
            step=Step(description="搜索课程"),
            runtime_tools=[_SearchTool()],
            max_tool_iterations=3,
            on_tool_event=None,
            user_content=[{"type": "text", "text": "查一下课程"}],
        )

    payload, _ = asyncio.run(_run())

    assert payload["success"] is True
    assert payload["result"] == "已完成课程检索"


def test_wait_for_human_node_should_complete_waiting_step_after_resume(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.interrupt",
        lambda payload: "AI 人工智能算法工程师体系课",
    )

    plan = Plan(
        title="课程推荐",
        goal="推荐课程并等待用户选择",
        language="zh",
        steps=[
            Step(
                id="step-1",
                title="等待用户选择课程",
                description="展示三门课程并等待用户选择",
                status=ExecutionStatus.RUNNING,
            ),
            Step(
                id="step-2",
                title="继续处理",
                description="根据用户选择继续后续步骤",
                status=ExecutionStatus.PENDING,
            ),
        ],
    )

    next_state = asyncio.run(
        wait_for_human_node(
            {
                "plan": plan,
                "current_step_id": "step-1",
                "pending_interrupt": {
                    "kind": "select",
                    "prompt": "请选择一门课程",
                    "options": [
                        {"label": "AI 人工智能算法工程师体系课", "resume_value": "AI 人工智能算法工程师体系课"},
                    ],
                },
                "graph_metadata": {},
                "message_window": [],
                "working_memory": {},
                "execution_count": 0,
            }
        )
    )

    assert next_state["pending_interrupt"] == {}
    assert next_state["execution_count"] == 1
    assert next_state["user_message"] == "AI 人工智能算法工程师体系课"
    assert next_state["current_step_id"] == "step-2"
    assert next_state["last_executed_step"].id == "step-1"
    assert next_state["last_executed_step"].status == ExecutionStatus.COMPLETED
    assert "已收到用户选择" in str(next_state["last_executed_step"].outcome.summary)
    assert "wait_resume_action" not in next_state["graph_metadata"].get("control", {})
    assert route_after_wait(next_state) == "guard_step_reuse"
    next_step = next_state["plan"].get_next_step()
    assert next_step is not None
    assert next_step.id == "step-2"
    assert next_state["working_memory"]["confirmed_facts"]["latest_user_input"] == "AI 人工智能算法工程师体系课"
    assert next_state["message_window"][-1]["message"] == "AI 人工智能算法工程师体系课"


def test_wait_for_human_node_should_append_execute_step_when_plan_only_contains_wait_step(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.interrupt",
        lambda payload: "从杭州出发，两天一夜",
    )

    plan = Plan(
        title="出行规划",
        goal="先补充关键信息",
        language="zh",
        steps=[
            Step(
                id="wait-step",
                title="等待用户补充信息",
                description="请补充出发地和时长",
                status=ExecutionStatus.RUNNING,
                task_mode_hint="human_wait",
            )
        ],
    )

    next_state = asyncio.run(
        wait_for_human_node(
            {
                "plan": plan,
                "current_step_id": "wait-step",
                "pending_interrupt": {
                    "kind": "input_text",
                    "prompt": "请补充出发地和时长",
                },
                "graph_metadata": {},
                "message_window": [],
                "working_memory": {},
                "execution_count": 0,
                "user_message": "帮我规划一个周末出行方案",
            }
        )
    )

    next_step = next_state["plan"].get_next_step()
    assert next_step is not None
    assert next_step.id != "wait-step"
    assert next_step.task_mode_hint == "research"
    assert next_step.description == "帮我规划一个周末出行方案"
    assert next_state["current_step_id"] == next_step.id
    assert next_state["user_message"] == "帮我规划一个周末出行方案"
    assert route_after_wait(next_state) == "guard_step_reuse"


def test_wait_for_human_node_should_preserve_real_task_mode_when_appending_post_wait_step(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.interrupt",
        lambda payload: "重点关注近两年的主流产品",
    )

    plan = Plan(
        title="AI 编程助手调研",
        goal="先补充调研范围",
        language="zh",
        steps=[
            Step(
                id="wait-step",
                title="等待用户补充信息",
                description="请补充调研范围",
                status=ExecutionStatus.RUNNING,
                task_mode_hint="human_wait",
            )
        ],
    )

    next_state = asyncio.run(
        wait_for_human_node(
            {
                "plan": plan,
                "current_step_id": "wait-step",
                "pending_interrupt": {
                    "kind": "input_text",
                    "prompt": "请补充调研范围",
                },
                "graph_metadata": {},
                "message_window": [],
                "working_memory": {},
                "execution_count": 0,
                "user_message": "调研主流 AI 编程助手及其支持的 IDE，并汇总差异",
            }
        )
    )

    next_step = next_state["plan"].get_next_step()
    assert next_step is not None
    assert next_step.id != "wait-step"
    assert next_step.task_mode_hint == "research"
    assert next_step.description == "调研主流 AI 编程助手及其支持的 IDE，并汇总差异"


def test_wait_for_human_node_should_cancel_waiting_step_and_replan_after_confirm_cancel(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.interrupt",
        lambda payload: False,
    )

    plan = Plan(
        title="课程推荐",
        goal="推荐课程并等待用户确认",
        language="zh",
        steps=[
            Step(
                id="step-1",
                title="等待用户确认",
                description="请确认是否继续当前批次",
                status=ExecutionStatus.RUNNING,
            ),
            Step(
                id="step-2",
                title="继续处理",
                description="根据确认结果继续后续步骤",
                status=ExecutionStatus.PENDING,
            ),
        ],
    )

    next_state = asyncio.run(
        wait_for_human_node(
            {
                "plan": plan,
                "current_step_id": "step-1",
                "pending_interrupt": {
                    "kind": "confirm",
                    "prompt": "确认继续执行？",
                    "confirm_resume_value": True,
                    "cancel_resume_value": False,
                    "confirm_label": "继续",
                    "cancel_label": "取消",
                },
                "graph_metadata": {},
                "message_window": [],
                "working_memory": {},
                "execution_count": 0,
            }
        )
    )

    assert next_state["pending_interrupt"] == {}
    assert next_state["execution_count"] == 1
    assert next_state["current_step_id"] is None
    assert next_state["last_executed_step"].id == "step-1"
    assert next_state["last_executed_step"].status == ExecutionStatus.CANCELLED
    assert next_state["graph_metadata"]["control"]["wait_resume_action"] == "replan"
    assert "已被用户取消" in str(next_state["last_executed_step"].outcome.summary)
    assert route_after_wait(next_state) == "replan"


def test_direct_wait_cancel_should_clear_direct_wait_control_state(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.interrupt",
        lambda payload: False,
    )

    waiting_state = asyncio.run(
        direct_wait_node(
            {
                "user_message": "先让我确认后再继续搜索课程",
                "graph_metadata": {},
                "message_window": [],
                "working_memory": {},
                "execution_count": 0,
            }
        )
    )

    cancelled_state = asyncio.run(wait_for_human_node(waiting_state))

    control = cancelled_state["graph_metadata"]["control"]
    assert cancelled_state["last_executed_step"].id == "direct-wait-confirm"
    assert cancelled_state["last_executed_step"].status == ExecutionStatus.CANCELLED
    assert control["wait_resume_action"] == "replan"
    assert "entry_strategy" not in control
    assert "skip_replan_when_plan_finished" not in control
    assert "direct_wait_original_message" not in control
    assert "direct_wait_execute_task_mode" not in control
    assert "direct_wait_original_task_executed" not in control
    assert route_after_wait(cancelled_state) == "replan"


def test_direct_wait_cancel_replan_execute_should_route_to_replan_instead_of_summarize(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.interrupt",
        lambda payload: False,
    )

    waiting_state = asyncio.run(
        direct_wait_node(
            {
                "user_message": "先让我确认后再继续搜索课程",
                "graph_metadata": {},
                "message_window": [],
                "working_memory": {},
                "execution_count": 0,
            }
        )
    )
    cancelled_state = asyncio.run(wait_for_human_node(waiting_state))
    replanned_plan = Plan(
        title="新的计划",
        goal="按取消后的新意图继续处理",
        language="zh",
        steps=[
            Step(
                id="step-new-1",
                title="执行新的步骤",
                description="执行新的步骤",
                status=ExecutionStatus.COMPLETED,
            )
        ],
        status=ExecutionStatus.PENDING,
    )

    next_state = {
        **cancelled_state,
        "plan": replanned_plan,
        "pending_interrupt": {},
        "current_step_id": None,
        "final_message": "新的计划批次已完成",
    }

    assert route_after_execute(next_state) == "replan"


def test_route_after_execute_should_fail_fast_to_summarize_on_direct_execute_failed() -> None:
    failed_plan = Plan(
        title="直接执行",
        goal="读取文件",
        language="zh",
        steps=[
            Step(
                id="direct-execute-step",
                title="直接执行用户请求",
                description="读取文件",
                status=ExecutionStatus.FAILED,
            )
        ],
        status=ExecutionStatus.PENDING,
    )
    next_state = {
        "plan": failed_plan,
        "last_executed_step": failed_plan.steps[0],
        "pending_interrupt": {},
        "execution_count": 1,
        "max_execution_steps": 20,
        "graph_metadata": {"control": {"entry_strategy": "direct_execute", "skip_replan_when_plan_finished": True}},
    }
    assert route_after_execute(next_state) == "summarize"


def test_route_after_execute_should_fail_fast_to_replan_when_failed_step_has_pending_followup() -> None:
    failed_plan = Plan(
        title="批次执行",
        goal="先失败再重规划",
        language="zh",
        steps=[
            Step(
                id="step-failed",
                title="失败步骤",
                description="失败步骤",
                status=ExecutionStatus.FAILED,
            ),
            Step(
                id="step-pending",
                title="后续步骤",
                description="后续步骤",
                status=ExecutionStatus.PENDING,
            ),
        ],
        status=ExecutionStatus.PENDING,
    )
    next_state = {
        "plan": failed_plan,
        "last_executed_step": failed_plan.steps[0],
        "pending_interrupt": {},
        "execution_count": 1,
        "max_execution_steps": 20,
        "graph_metadata": {"control": {}},
    }
    assert route_after_execute(next_state) == "replan"


def test_route_after_execute_should_skip_replan_when_final_inline_step_completed() -> None:
    completed_plan = Plan(
        title="批次执行",
        goal="最终交付",
        language="zh",
        steps=[
            Step(
                id="step-final-inline",
                title="输出最终结果",
                description="直接输出最终结果",
                output_mode="inline",
                delivery_role="final",
                status=ExecutionStatus.COMPLETED,
            )
        ],
        status=ExecutionStatus.PENDING,
    )
    next_state = {
        "plan": completed_plan,
        "last_executed_step": completed_plan.steps[0],
        "pending_interrupt": {},
        "execution_count": 1,
        "max_execution_steps": 20,
        "graph_metadata": {"control": {}},
    }
    assert route_after_execute(next_state) == "summarize"


def test_direct_wait_should_execute_original_task_after_confirm(monkeypatch) -> None:
    user_message = "先让我确认后再继续搜索课程"
    llm = _IllegalAskUserLLM()
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.interrupt",
        lambda payload: True,
    )

    waiting_state = asyncio.run(
        direct_wait_node(
            {
                "user_message": user_message,
                "graph_metadata": {},
                "message_window": [],
                "working_memory": {},
                "execution_count": 0,
            }
        )
    )

    resumed_state = asyncio.run(wait_for_human_node(waiting_state))

    assert resumed_state["current_step_id"] == "direct-wait-execute"
    assert resumed_state["user_message"] == user_message
    assert resumed_state["message_window"][-1]["message"] == "继续"

    executed_state = asyncio.run(
        execute_step_node(
            resumed_state,
            llm,
            runtime_tools=[MessageTool()],
        )
    )

    control = executed_state["graph_metadata"]["control"]
    assert llm.calls == 2
    assert executed_state["pending_interrupt"] == {}
    assert executed_state["current_step_id"] is None
    assert executed_state["user_message"] == user_message
    assert executed_state["final_message"] == "当前步骤只完成搜索，不提前让用户选择"
    assert executed_state["last_executed_step"].id == "direct-wait-execute"
    assert executed_state["last_executed_step"].status == ExecutionStatus.COMPLETED
    assert control["direct_wait_original_task_executed"] is True
    assert "wait_resume_action" not in control
    assert "direct_wait_execute_task_mode" not in control
    assert "direct_wait_original_message" not in control


def test_route_after_wait_should_continue_current_batch_before_replan() -> None:
    plan = Plan(
        title="课程推荐",
        goal="推荐课程并等待用户选择",
        language="zh",
        steps=[
            Step(
                id="step-1",
                title="等待用户选择课程",
                description="展示三门课程并等待用户选择",
                status=ExecutionStatus.COMPLETED,
            ),
            Step(
                id="step-2",
                title="继续处理",
                description="根据用户选择继续后续步骤",
                status=ExecutionStatus.PENDING,
            ),
        ],
    )

    assert route_after_wait({"plan": plan, "execution_count": 1, "max_execution_steps": 20}) == "guard_step_reuse"

    plan.steps[1].status = ExecutionStatus.COMPLETED
    assert route_after_wait({"plan": plan, "execution_count": 2, "max_execution_steps": 20}) == "replan"


def test_execute_step_node_should_append_confirmed_facts_to_description_context() -> None:
    class _CapturePromptLLM:
        def __init__(self) -> None:
            self.last_prompt_text = ""

        async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
            for message in messages:
                if message.get("role") != "user":
                    continue
                content = message.get("content")
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            self.last_prompt_text = str(item.get("text") or "")
                            break
                elif isinstance(content, str):
                    self.last_prompt_text = content
            return {
                "role": "assistant",
                "content": '{"success": true, "result": "执行完成", "attachments": []}',
            }

    llm = _CapturePromptLLM()
    plan = Plan(
        title="模板渲染",
        goal="验证执行模板渲染",
        language="zh",
        steps=[
            Step(
                id="step-render",
                title="执行模板步骤",
                description="根据用户输入继续处理",
                status=ExecutionStatus.PENDING,
            )
        ],
    )
    state = {
        "plan": plan,
        "current_step_id": "step-render",
        "pending_interrupt": {},
        "graph_metadata": {},
        "message_window": [],
        "working_memory": {"confirmed_facts": {"course_name": "AI 工程师课程", "budget": 3000}},
        "execution_count": 0,
        "user_message": "继续执行",
        "input_parts": [],
    }

    next_state = asyncio.run(
        execute_step_node(
            state,
            llm,
            runtime_tools=[],
        )
    )

    assert next_state["last_executed_step"].status == ExecutionStatus.COMPLETED
    assert "根据用户输入继续处理" in llm.last_prompt_text
    assert "已确认用户输入（仅作当前步骤执行上下文，不改写原步骤描述）" in llm.last_prompt_text
    assert "- course_name: AI 工程师课程" in llm.last_prompt_text
    assert "- budget: 3000" in llm.last_prompt_text
