import asyncio
import json

from app.domain.models import (
    ExecutionStatus,
    MessageEvent,
    Plan,
    PlanEvent,
    PlanEventStatus,
    Step,
    StepArtifactPolicy,
    StepEvent,
    StepOutcome,
    StepOutputMode,
    StepTaskModeHint,
    TextStreamStartEvent,
)
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.workspace_runtime.entry import EntryCompiler
from app.domain.services.workspace_runtime.policies import (
    StepContractCompilationIssue,
    compile_step_contracts,
    filter_final_delivery_steps,
)
from app.infrastructure.runtime.langgraph.graphs import bind_live_event_sink, unbind_live_event_sink
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes import (
    create_or_reuse_plan_node as _create_or_reuse_plan_node,
    replan_node as _replan_node,
    summarize_node as _summarize_node,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.routing import route_after_plan


_TEST_RUNTIME_CONTEXT_SERVICE = RuntimeContextService()


async def create_or_reuse_plan_node(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return await _create_or_reuse_plan_node(*args, **kwargs)


async def replan_node(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return await _replan_node(*args, **kwargs)


async def summarize_node(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return await _summarize_node(*args, **kwargs)


def _entry_contract_control(user_message: str) -> dict:
    contract = EntryCompiler().compile_state(
        {
            "user_message": user_message,
            "input_parts": [],
            "message_window": [],
            "recent_run_briefs": [],
            "conversation_summary": "",
        }
    )
    return {"entry_contract": contract.model_dump(mode="json")}


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
                        }
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


class _EmptyPlannerLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "message": "Planner 未返回步骤。",
                    "goal": "生成冲突兜底步骤",
                    "title": "冲突兜底",
                    "language": "zh",
                    "steps": [],
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
                            "id": "final-step",
                            "title": "整理最终回答给用户",
                            "description": "根据已收集资料整理最终回答并交付给用户",
                            "task_mode_hint": "general",
                            "output_mode": "none",
                            "artifact_policy": "forbid_file_output",
                        }
                    ]
                },
                ensure_ascii=False,
            )
        }


class _SuccessSummaryLLM:
    async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
        return {
            "content": json.dumps(
                {
                    "message": "任务已完成。",
                    "final_answer_text": "任务已成功完成。",
                    "attachments": [],
                },
                ensure_ascii=False,
            )
        }


def _base_planner_state(user_message: str, *, graph_metadata: dict | None = None) -> dict:
    return {
        "user_message": user_message,
        "graph_metadata": graph_metadata if graph_metadata is not None else {"control": _entry_contract_control(user_message)},
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
        "final_answer_text": "",
        "thread_id": "thread-1",
        "conversation_summary": "",
    }


def test_filter_final_delivery_steps_should_drop_english_final_delivery_steps() -> None:
    steps, issues, corrected_count = compile_step_contracts(
        steps=[
            Step(
                id="1",
                description="Summarize the collected research findings into a final answer for the user",
                task_mode_hint="general",
                output_mode="none",
                artifact_policy="forbid_file_output",
            ),
            Step(
                id="2",
                description="Read the source pages and extract facts for later summary",
                task_mode_hint="research",
                output_mode="none",
                artifact_policy="forbid_file_output",
            ),
        ],
        user_message="Research LangGraph HITL patterns and give me five takeaways",
    )

    filtered_steps, dropped_count = filter_final_delivery_steps(
        steps=steps,
        user_message="Research LangGraph HITL patterns and give me five takeaways",
    )

    assert issues == []
    assert corrected_count == 0
    assert dropped_count == 1
    assert [step.id for step in filtered_steps] == ["2"]


def test_filter_final_delivery_steps_should_keep_explicit_file_report_request() -> None:
    steps, _, _ = compile_step_contracts(
        steps=[
            Step(
                id="1",
                description="整理调研结果并生成 Markdown 报告文件",
                task_mode_hint="coding",
                output_mode="file",
                artifact_policy="allow_file_output",
            )
        ],
        user_message="请调研后生成一份 Markdown 报告文件",
    )

    filtered_steps, dropped_count = filter_final_delivery_steps(
        steps=steps,
        user_message="请调研后生成一份 Markdown 报告文件",
    )

    assert dropped_count == 0
    assert len(filtered_steps) == 1


def test_filter_final_delivery_steps_should_keep_explicit_final_report_file_request() -> None:
    user_message = "请调研后生成一份最终 Markdown 报告文件"
    steps, _, _ = compile_step_contracts(
        steps=[
            Step(
                id="1",
                description="整理调研结果并生成最终 Markdown 报告文件",
                task_mode_hint="coding",
                output_mode="file",
                artifact_policy="allow_file_output",
            )
        ],
        user_message=user_message,
    )

    filtered_steps, dropped_count = filter_final_delivery_steps(
        steps=steps,
        user_message=user_message,
    )

    assert dropped_count == 0
    assert len(filtered_steps) == 1
    assert filtered_steps[0].id == "1"


def test_planner_plan_only_should_emit_final_event_and_final_answer_text() -> None:
    captured_events = []

    async def _sink(event):
        captured_events.append(event)

    user_message = "帮我规划一个北京 3 天旅游安排，先不要执行，只给步骤。"
    token = bind_live_event_sink(_sink)
    try:
        state = asyncio.run(
            create_or_reuse_plan_node(
                _base_planner_state(user_message),
                _PlanOnlyPlannerLLM(),
            )
        )
    finally:
        unbind_live_event_sink(token)

    assert state["plan"] is not None
    assert state["plan"].status == ExecutionStatus.COMPLETED
    assert state["current_step_id"] is None
    assert state["final_message"] == "好的，我将先给出执行步骤，不直接开始执行。"
    assert state["final_answer_text"] == state["final_message"]
    assert state["selected_artifacts"] == []
    assert state["emitted_events"][1].status == PlanEventStatus.CREATED
    assert state["emitted_events"][1].plan.status == ExecutionStatus.COMPLETED
    assert isinstance(captured_events[0], TextStreamStartEvent)
    assert captured_events[0].channel.value == "final_message"
    assert captured_events[0].stage == "final"
    assert isinstance(state["emitted_events"][-1], MessageEvent)
    assert state["emitted_events"][-1].stage == "final"
    assert route_after_plan(state) == "consolidate_memory"


def test_planner_non_planned_empty_steps_should_use_direct_fallback_contract() -> None:
    state = asyncio.run(
        create_or_reuse_plan_node(
            _base_planner_state(
                "再详细一点，给我 3 个典型使用场景。",
                graph_metadata={},
            ),
            _NoStepPlannerWithExtraTextLLM(),
        )
    )

    assert state["plan"] is not None
    assert state["plan"].status == ExecutionStatus.COMPLETED
    assert "三个典型使用场景" in state["final_message"]
    assert state["final_answer_text"] == state["final_message"]
    assert state["emitted_events"][-1].type == "message"
    assert state["emitted_events"][-1].stage == "final"


def test_planner_empty_steps_with_invalid_fallback_should_build_failed_synthetic_step(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.planner_nodes.collect_step_contract_hard_issues",
        lambda steps: [
            StepContractCompilationIssue(
                step_id="fallback-execution-step",
                issue_code="synthetic_contract_issue",
                issue_message="兜底步骤契约不可执行。",
            )
        ],
    )
    user_message = "请规划并分阶段执行：生成冲突兜底步骤"

    state = asyncio.run(
        create_or_reuse_plan_node(
            _base_planner_state(user_message),
            _EmptyPlannerLLM(),
        )
    )

    assert state["plan"] is not None
    assert state["plan"].status == ExecutionStatus.FAILED
    assert len(state["plan"].steps) == 1
    failed_step = state["plan"].steps[0]
    assert failed_step.id == "planner-failed-step"
    assert failed_step.status == ExecutionStatus.FAILED
    assert failed_step.outcome is not None
    assert failed_step.outcome.done is False
    assert "Planner 未能生成可执行步骤" in failed_step.outcome.blockers
    assert state["last_executed_step"].id == "planner-failed-step"
    assert state["current_step_id"] is None
    assert state["final_message"] == ""
    assert [event.type for event in state["emitted_events"]] == ["title", "plan"]
    assert state["emitted_events"][1].status == PlanEventStatus.CREATED
    assert route_after_plan(state) == "summarize"


def test_failed_synthetic_step_should_be_summarized_as_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.planner_nodes.collect_step_contract_hard_issues",
        lambda steps: [
            StepContractCompilationIssue(
                step_id="fallback-execution-step",
                issue_code="synthetic_contract_issue",
                issue_message="兜底步骤契约不可执行。",
            )
        ],
    )
    user_message = "请规划并分阶段执行：生成冲突兜底步骤"
    planned_state = asyncio.run(
        create_or_reuse_plan_node(
            _base_planner_state(user_message),
            _EmptyPlannerLLM(),
        )
    )

    summarized_state = asyncio.run(summarize_node(planned_state, _SuccessSummaryLLM()))

    assert "未完整完成" in summarized_state["final_answer_text"]
    assert "执行失败" in summarized_state["final_answer_text"]
    assert "任务已成功完成" not in summarized_state["final_answer_text"]
    assert summarized_state["emitted_events"][-2].stage == "final"


def test_replan_final_delivery_only_should_keep_plan_and_route_to_summary() -> None:
    completed_step = Step(
        id="step-done",
        title="完成调研",
        description="完成调研",
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(done=True, summary="已完成调研"),
    )
    plan = Plan(
        title="重规划最终交付过滤",
        goal="整理最终答案",
        language="zh",
        steps=[completed_step],
    )
    state = {
        "plan": plan,
        "last_executed_step": completed_step.model_copy(deep=True),
        "user_message": "整理最终答案给我",
        "emitted_events": [],
    }

    next_state = asyncio.run(replan_node(state, _FinalDeliveryReplanLLM()))

    assert next_state["plan"].steps == [completed_step]
    assert next_state["current_step_id"] is None
    assert next_state.get("emitted_events", []) == []


def test_replan_final_delivery_only_should_keep_existing_next_step() -> None:
    completed_step = Step(
        id="step-done",
        title="完成调研",
        description="完成调研",
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(done=True, summary="已完成调研"),
    )
    pending_step = Step(
        id="step-next",
        title="继续读取来源",
        description="继续读取来源",
        status=ExecutionStatus.PENDING,
    )
    plan = Plan(
        title="重规划最终交付过滤",
        goal="继续读取来源",
        language="zh",
        steps=[completed_step, pending_step],
    )
    state = {
        "plan": plan,
        "last_executed_step": completed_step.model_copy(deep=True),
        "user_message": "继续读取来源",
        "emitted_events": [],
    }

    next_state = asyncio.run(replan_node(state, _FinalDeliveryReplanLLM()))

    assert next_state["plan"].steps == [completed_step, pending_step]
    assert next_state["current_step_id"] == "step-next"
    assert not any(isinstance(event, PlanEvent) for event in next_state.get("emitted_events", []))
    assert not any(isinstance(event, StepEvent) for event in next_state.get("emitted_events", []))


def test_replan_final_delivery_only_should_clear_one_shot_control_metadata() -> None:
    completed_step = Step(
        id="step-done",
        title="完成原子动作",
        description="完成原子动作",
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(done=True, summary="需要升级后继续规划"),
    )
    plan = Plan(
        title="重规划一次性信号清理",
        goal="继续规划",
        language="zh",
        steps=[completed_step],
    )
    state = {
        "plan": plan,
        "last_executed_step": completed_step.model_copy(deep=True),
        "user_message": "继续规划",
        "graph_metadata": {
            "control": {
                **_entry_contract_control("继续规划"),
                "entry_upgrade": {
                    "reason_code": "open_questions_require_planner",
                    "source_route": "atomic_action",
                    "target_route": "planned_task",
                    "evidence": {"open_questions": ["需要继续规划"]},
                },
                "wait_resume_action": "replan",
            }
        },
        "emitted_events": [],
    }

    next_state = asyncio.run(replan_node(state, _FinalDeliveryReplanLLM()))

    control = next_state["graph_metadata"]["control"]
    assert "entry_upgrade" not in control
    assert "wait_resume_action" not in control
