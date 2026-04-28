import asyncio
import json
from typing import Any

from app.domain.models import (
    DoneEvent,
    ExecutionStatus,
    MessageEvent,
    Plan,
    PlanEvent,
    PlanEventStatus,
    Step,
    StepOutcome,
    ToolResult,
)
from app.domain.services.tools import BaseTool, MessageTool
from app.domain.services.tools.base import tool
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.workspace_runtime.entry import EntryCompiler
from app.domain.services.workspace_runtime.policies.step_contract_policy import filter_final_delivery_steps
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes import (
    atomic_action_node,
    create_or_reuse_plan_node as _create_or_reuse_plan_node,
    direct_answer_node,
    direct_wait_node,
    execute_step_node as _execute_step_node,
    finalize_node,
    replan_node as _replan_node,
    summarize_node as _summarize_node,
    wait_for_human_node,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.routing import (
    route_after_execute,
    route_after_plan,
    route_after_wait,
)


_TEST_RUNTIME_CONTEXT_SERVICE = RuntimeContextService()


def _entry_contract_control(user_message: str, **state_overrides: Any) -> dict:
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


def _base_state(user_message: str, *, control: dict | None = None) -> dict:
    return {
        "session_id": "session-1",
        "user_id": "user-1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "user_message": user_message,
        "graph_metadata": {"control": control or _entry_contract_control(user_message)},
        "working_memory": {},
        "message_window": [],
        "input_parts": [],
        "selected_artifacts": [],
        "historical_artifact_paths": [],
        "retrieved_memories": [],
        "pending_memory_writes": [],
        "recent_run_briefs": [],
        "recent_attempt_briefs": [],
        "session_open_questions": [],
        "session_blockers": [],
        "step_states": [],
        "pending_interrupt": {},
        "emitted_events": [],
        "execution_count": 0,
        "max_execution_steps": 20,
        "current_step_id": None,
        "last_executed_step": None,
        "final_message": "",
        "conversation_summary": "",
    }


def _single_step_plan(*, step: Step, title: str = "P0-2 回归计划") -> Plan:
    return Plan(
        title=title,
        goal="验证 P0-2 最终出口边界",
        language="zh",
        steps=[step],
        status=ExecutionStatus.PENDING,
    )


async def create_or_reuse_plan_node(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return await _create_or_reuse_plan_node(*args, **kwargs)


async def execute_step_node(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return await _execute_step_node(*args, **kwargs)


async def replan_node(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return await _replan_node(*args, **kwargs)


async def summarize_node(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return await _summarize_node(*args, **kwargs)


class _ResearchExecuteLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "已完成调研步骤，只沉淀执行摘要。",
                    "facts_learned": ["资料 A 支持结论一", "资料 B 支持结论二"],
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


class _SummaryLLM:
    def __init__(self, *, attachments: list[str] | None = None) -> None:
        self.attachments = list(attachments or [])

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "message": "任务已总结完成。",
                    "final_answer_text": "这是 summary 重新组织后的最终正文。",
                    "attachments": self.attachments,
                    "facts_in_session": [],
                    "user_preferences": {},
                    "memory_candidates": [],
                },
                ensure_ascii=False,
            )
        }


class _DirectAnswerLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {"content": json.dumps({"message": "你好，我在。"}, ensure_ascii=False)}


class _PlanOnlyPlannerLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "message": "已按要求只交付计划，不执行。",
                    "goal": "规划北京三日游",
                    "title": "北京三日游计划",
                    "language": "zh",
                    "steps": [
                        {
                            "id": "1",
                            "description": "检索北京景点信息",
                            "task_mode_hint": "research",
                            "output_mode": "none",
                            "artifact_policy": "forbid_file_output",
                        }
                    ],
                },
                ensure_ascii=False,
            )
        }


class _FinalDeliveryPlannerLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "message": "已生成计划。",
                    "goal": "调研并输出结果",
                    "title": "最终交付过滤测试",
                    "language": "zh",
                    "steps": [
                        {
                            "id": "1",
                            "description": "整理最终回答并交付给用户",
                            "task_mode_hint": "general",
                            "output_mode": "none",
                            "artifact_policy": "forbid_file_output",
                        },
                        {
                            "id": "2",
                            "description": "调研候选资料并提取事实",
                            "task_mode_hint": "research",
                            "output_mode": "none",
                            "artifact_policy": "forbid_file_output",
                        },
                    ],
                },
                ensure_ascii=False,
            )
        }


class _FinalDeliveryReplanLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "steps": [
                        {
                            "id": "replan-final",
                            "description": "整理最终回答并交付给用户",
                            "task_mode_hint": "general",
                            "output_mode": "none",
                            "artifact_policy": "forbid_file_output",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        }


class _WriteFileTool(BaseTool):
    name = "file"

    @tool(
        name="write_file",
        description="写入文件",
        parameters={
            "filepath": {"type": "string"},
            "content": {"type": "string"},
        },
        required=["filepath", "content"],
    )
    async def write_file(self, filepath: str, content: str) -> ToolResult:
        return ToolResult(success=True, data={"filepath": filepath, "content_length": len(content)})


class _WriteFileExecuteLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-write",
                        "function": {
                            "name": "write_file",
                            "arguments": json.dumps(
                                {
                                    "filepath": "/home/ubuntu/final-report.md",
                                    "content": "# 最终报告\n\n已生成。",
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "报告文件已生成。",
                    "attachments": [],
                    "deliver_result_as_attachment": True,
                },
                ensure_ascii=False,
            ),
            "tool_calls": [],
        }


class _IllegalAskUserThenCompleteLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-ask",
                        "function": {
                            "name": "message_ask_user",
                            "arguments": json.dumps(
                                {
                                    "text": "请选择一个选项",
                                    "kind": "select",
                                    "options": [{"label": "A", "resume_value": "a"}],
                                },
                                ensure_ascii=False,
                            ),
                        },
                    }
                ],
            }
        return {
            "content": json.dumps(
                {
                    "success": True,
                    "summary": "确认后已执行原始任务。",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


def _assert_final_before_done(state: dict) -> None:
    events = list(state.get("emitted_events") or [])
    final_indexes = [
        index
        for index, event in enumerate(events)
        if isinstance(event, MessageEvent) and event.stage == "final"
    ]
    done_indexes = [index for index, event in enumerate(events) if isinstance(event, DoneEvent)]
    assert final_indexes
    assert done_indexes
    assert final_indexes[-1] < done_indexes[-1]


def test_multistep_research_should_only_deliver_final_text_from_summary() -> None:
    step = Step(
        id="research-step",
        title="调研资料",
        description="调研资料并提取事实",
        task_mode_hint="research",
        output_mode="none",
        artifact_policy="forbid_file_output",
        status=ExecutionStatus.PENDING,
    )
    state = _base_state("请调研资料后给我结论")
    state["plan"] = _single_step_plan(step=step)

    executed_state = asyncio.run(execute_step_node(state, _ResearchExecuteLLM(), runtime_tools=[]))

    assert executed_state["last_executed_step"].outcome.summary == "已完成调研步骤，只沉淀执行摘要。"
    assert executed_state["final_message"] == ""
    assert "final_answer_text" not in executed_state
    assert not any(isinstance(event, MessageEvent) and event.stage == "final" for event in executed_state["emitted_events"])
    assert route_after_execute(executed_state) == "replan"

    summarized_state = asyncio.run(summarize_node(executed_state, _SummaryLLM()))
    finalized_state = asyncio.run(finalize_node(summarized_state))

    assert summarized_state["final_message"] == "任务已总结完成。"
    assert summarized_state["final_answer_text"] == "这是 summary 重新组织后的最终正文。"
    assert summarized_state["final_answer_text"] != executed_state["last_executed_step"].outcome.summary
    assert isinstance(summarized_state["emitted_events"][-2], MessageEvent)
    assert summarized_state["emitted_events"][-2].stage == "final"
    _assert_final_before_done(finalized_state)


def test_file_generation_should_keep_execute_artifact_candidate_until_summary_selects_attachment() -> None:
    step = Step(
        id="file-step",
        title="生成报告",
        description="生成最终 Markdown 报告文件",
        task_mode_hint="coding",
        output_mode="file",
        artifact_policy="allow_file_output",
        status=ExecutionStatus.PENDING,
    )
    state = _base_state("请生成一份最终 Markdown 报告文件")
    state["plan"] = _single_step_plan(step=step, title="文件生成回归")

    executed_state = asyncio.run(
        execute_step_node(
            state,
            _WriteFileExecuteLLM(),
            runtime_tools=[_WriteFileTool()],
        )
    )

    assert executed_state["last_executed_step"].outcome.produced_artifacts == ["/home/ubuntu/final-report.md"]
    assert executed_state["selected_artifacts"] == []
    assert "final_answer_text" not in executed_state

    summarized_state = asyncio.run(
        summarize_node(
            executed_state,
            _SummaryLLM(attachments=["/home/ubuntu/final-report.md"]),
        )
    )

    assert summarized_state["selected_artifacts"] == ["/home/ubuntu/final-report.md"]
    final_message_event = next(
        event
        for event in summarized_state["emitted_events"]
        if isinstance(event, MessageEvent) and event.stage == "final"
    )
    assert [attachment.filepath for attachment in final_message_event.attachments] == ["/home/ubuntu/final-report.md"]


def test_atomic_action_should_execute_step_then_summarize_instead_of_direct_final_output() -> None:
    state = asyncio.run(
        atomic_action_node(
            _base_state(
                "搜索 OpenAI 官网",
                control=_entry_contract_control("搜索 OpenAI 官网"),
            )
        )
    )

    assert state["plan"] is not None
    assert state["final_message"] == ""
    assert route_after_plan(state) == "guard_step_reuse"

    executed_state = asyncio.run(execute_step_node(state, _ResearchExecuteLLM(), runtime_tools=[]))

    assert executed_state["final_message"] == ""
    assert "final_answer_text" not in executed_state
    assert route_after_execute(executed_state) == "summarize"

    summarized_state = asyncio.run(summarize_node(executed_state, _SummaryLLM()))

    assert summarized_state["final_answer_text"] == "这是 summary 重新组织后的最终正文。"
    assert any(isinstance(event, MessageEvent) and event.stage == "final" for event in summarized_state["emitted_events"])


def test_direct_wait_should_execute_after_confirm_then_summarize(monkeypatch) -> None:
    user_message = "先让我确认后再继续搜索课程"
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.interrupt",
        lambda payload: True,
    )
    waiting_state = asyncio.run(
        direct_wait_node(
            _base_state(
                user_message,
                control=_entry_contract_control(user_message),
            )
        )
    )
    resumed_state = asyncio.run(wait_for_human_node(waiting_state))

    assert resumed_state["current_step_id"] == "direct-wait-execute"
    assert route_after_wait(resumed_state) == "guard_step_reuse"

    executed_state = asyncio.run(
        execute_step_node(
            resumed_state,
            _IllegalAskUserThenCompleteLLM(),
            runtime_tools=[MessageTool()],
        )
    )

    assert executed_state["last_executed_step"].id == "direct-wait-execute"
    assert executed_state["final_message"] == ""
    assert "final_answer_text" not in executed_state
    assert route_after_execute(executed_state) == "summarize"

    summarized_state = asyncio.run(summarize_node(executed_state, _SummaryLLM()))

    assert summarized_state["final_answer_text"] == "这是 summary 重新组织后的最终正文。"
    assert any(isinstance(event, MessageEvent) and event.stage == "final" for event in summarized_state["emitted_events"])


def test_direct_answer_should_not_enter_planner_or_execute_and_should_clear_artifacts() -> None:
    state = _base_state(
        "你好",
        control=_entry_contract_control("你好"),
    )
    state["selected_artifacts"] = ["/home/ubuntu/old-report.md"]

    answered_state = asyncio.run(
        direct_answer_node(
            state,
            _DirectAnswerLLM(),
            runtime_context_service=_TEST_RUNTIME_CONTEXT_SERVICE,
        )
    )

    assert answered_state["plan"].steps == []
    assert answered_state["plan"].status == ExecutionStatus.COMPLETED
    assert answered_state["current_step_id"] is None
    assert answered_state["final_message"] == "你好，我在。"
    assert answered_state["final_answer_text"] == "你好，我在。"
    assert answered_state["selected_artifacts"] == []
    assert route_after_plan(answered_state) == "consolidate_memory"
    assert [event.type for event in answered_state["emitted_events"]] == ["title", "message"]


def test_plan_only_should_deliver_plan_as_final_text_without_execute_or_summary() -> None:
    user_message = "帮我规划北京三日游，先不要执行，只给步骤。"
    planned_state = asyncio.run(
        create_or_reuse_plan_node(
            _base_state(
                user_message,
                control=_entry_contract_control(user_message),
            ),
            _PlanOnlyPlannerLLM(),
        )
    )

    assert planned_state["plan"].status == ExecutionStatus.COMPLETED
    assert planned_state["current_step_id"] is None
    assert planned_state["final_answer_text"] == planned_state["final_message"]
    assert planned_state["selected_artifacts"] == []
    assert isinstance(planned_state["emitted_events"][1], PlanEvent)
    assert planned_state["emitted_events"][1].status == PlanEventStatus.CREATED
    assert isinstance(planned_state["emitted_events"][-1], MessageEvent)
    assert planned_state["emitted_events"][-1].stage == "final"
    assert route_after_plan(planned_state) == "consolidate_memory"


def test_planner_and_replan_should_filter_final_delivery_steps() -> None:
    user_message = "请调研资料后给我结论"
    planned_state = asyncio.run(
        create_or_reuse_plan_node(
            _base_state(
                user_message,
                control=_entry_contract_control(user_message),
            ),
            _FinalDeliveryPlannerLLM(),
        )
    )

    assert [step.description for step in planned_state["plan"].steps] == ["调研候选资料并提取事实"]
    assert planned_state["current_step_id"] == planned_state["plan"].steps[0].id
    assert not any(
        isinstance(event, MessageEvent) and event.stage == "final"
        for event in planned_state["emitted_events"]
    )

    planned_state["plan"].steps[0].status = ExecutionStatus.COMPLETED
    planned_state["plan"].steps[0].outcome = StepOutcome(done=True, summary="已完成调研")
    planned_state["last_executed_step"] = planned_state["plan"].steps[0].model_copy(deep=True)
    replanned_state = asyncio.run(replan_node(planned_state, _FinalDeliveryReplanLLM()))

    assert len(replanned_state["plan"].steps) == 1
    assert replanned_state["plan"].steps[0].description == "调研候选资料并提取事实"
    assert replanned_state["current_step_id"] is None
    assert route_after_plan(replanned_state) == "summarize"


def test_final_delivery_filter_should_keep_explicit_final_report_file_step() -> None:
    step = Step(
        id="file-report",
        title="生成最终 Markdown 报告文件",
        description="整理调研结果并生成最终 Markdown 报告文件",
        output_mode="file",
        artifact_policy="allow_file_output",
    )

    filtered_steps, dropped_count = filter_final_delivery_steps(
        steps=[step],
        user_message="请调研后生成一份最终 Markdown 报告文件",
    )

    assert dropped_count == 0
    assert filtered_steps == [step]
