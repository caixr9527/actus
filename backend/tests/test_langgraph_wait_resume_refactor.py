import asyncio

from app.domain.models import ExecutionStatus, Plan, Step, ToolEventStatus
from app.domain.services.tools.base import BaseTool, tool
from app.domain.services.tools.message import MessageTool
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.nodes import (
    wait_for_human_node,
)
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.routing import (
    route_after_wait,
)
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.tools import (
    execute_step_with_prompt,
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


def test_wait_for_human_node_should_complete_waiting_step_after_resume(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.nodes.interrupt",
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
                "step_local_memory": {
                    "current_step_id": "step-1",
                    "waiting_step_id": "step-1",
                },
                "graph_metadata": {
                    "pending_interrupts": [
                        {
                            "interrupt_id": "interrupt-1",
                            "payload": {
                                "kind": "select",
                                "prompt": "请选择一门课程",
                                "options": [
                                    {"label": "AI 人工智能算法工程师体系课", "resume_value": "AI 人工智能算法工程师体系课"},
                                ],
                            },
                        }
                    ]
                },
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
    assert "已收到用户回复" in str(next_state["last_executed_step"].outcome.summary)
    assert next_state["graph_metadata"].get("pending_interrupts") is None
    assert next_state["message_window"][-1]["message"] == "AI 人工智能算法工程师体系课"


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
