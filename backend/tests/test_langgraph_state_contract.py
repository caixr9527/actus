import asyncio
from datetime import datetime

from app.domain.models import (
    BrowserToolContent,
    DoneEvent,
    ExecutionStatus,
    File,
    Plan,
    PlanEvent,
    PlanEventStatus,
    Session,
    Step,
    StepArtifactPolicy,
    StepDeliveryContextState,
    StepDeliveryRole,
    StepEvent,
    StepEventStatus,
    StepOutcome,
    StepOutputMode,
    StepTaskModeHint,
    ToolEvent,
    ToolEventStatus,
    ToolResult,
    WaitEvent,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowRunSummary,
    SessionContextSnapshot,
)
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.domain.services.runtime.langgraph_state import (
    GRAPH_STATE_CONTRACT_SCHEMA_VERSION,
    GraphStateContractMapper,
)
from app.infrastructure.runtime.langgraph_graphs import bind_live_event_sink, unbind_live_event_sink
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.nodes import (
    create_or_reuse_plan_node as _create_or_reuse_plan_node,
)


_TEST_RUNTIME_CONTEXT_SERVICE = RuntimeContextService()


async def create_or_reuse_plan_node(*args, **kwargs):
    kwargs.setdefault("runtime_context_service", _TEST_RUNTIME_CONTEXT_SERVICE)
    return await _create_or_reuse_plan_node(
        *args,
        **kwargs,
    )


def _build_plan(step_status: ExecutionStatus = ExecutionStatus.PENDING) -> Plan:
    return Plan(
        title="测试计划",
        goal="验证状态契约",
        language="zh",
        message="plan message",
        steps=[
            Step(
                id="step-1",
                title="执行第一步",
                description="执行第一步",
                task_mode_hint=StepTaskModeHint.RESEARCH,
                output_mode=StepOutputMode.NONE,
                artifact_policy=StepArtifactPolicy.FORBID_FILE_OUTPUT,
                delivery_role=StepDeliveryRole.FINAL,
                delivery_context_state=StepDeliveryContextState.NEEDS_PREPARATION,
                objective_key="objective-step-1",
                success_criteria=["执行第一步完成"],
                status=step_status,
            )
        ],
        status=ExecutionStatus.PENDING,
    )


def test_graph_state_contract_should_build_initial_state_from_workflow_run_snapshot() -> None:
    session = Session(id="session-1", workspace_id="workspace-1", current_run_id="run-1")
    plan = _build_plan()
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        thread_id="thread-1",
        checkpoint_namespace="",
        checkpoint_id="cp-1",
        runtime_metadata={
            "graph_state_contract": {
                "graph_state": {
                    "plan": plan.model_dump(mode="json"),
                    "current_step_id": "step-1",
                    "message_window": [
                        {"role": "user", "message": "上一次用户输入", "input_part_count": 0},
                        {
                            "role": "assistant",
                            "message": "上一轮输出过附件",
                            "attachment_paths": [
                                "artifact-id-1",
                                "https://example.com/final.md",
                                "final.md",
                                "/tmp/final.md",
                            ],
                        },
                    ],
                    "conversation_summary": "历史对话摘要",
                    "working_memory": {
                        "goal": "验证状态契约",
                        "constraints": ["保持兼容"],
                    },
                    "task_mode": "research",
                    "environment_digest": {
                        "task_mode": "research",
                        "payload": {"recent_search_queries": ["langgraph persistence"]},
                    },
                    "observation_digest": {
                        "task_mode": "research",
                        "payload": {"last_step_result": "已完成上一轮检索"},
                    },
                    "recent_action_digest": {
                        "task_mode": "research",
                        "payload": {"last_user_wait_reason": "请补充上下文"},
                    },
                    "retrieved_memories": [
                        {"id": "mem-1", "summary": "用户偏好中文"},
                    ],
                    "pending_memory_writes": [
                        {"id": "pending-1", "summary": "待写入事实"},
                    ],
                    "selected_artifacts": [
                        "artifact-id-1",
                        "https://example.com/final.md",
                        "final.md",
                        "/tmp/final.md",
                    ],
                    "pending_interrupt": {
                        "kind": "input_text",
                        "prompt": "请补充上下文",
                        "attachments": ["/tmp/context.md"],
                        "response_key": "message",
                    },
                    "metadata": {"control": {"entry_strategy": "recall_memory_context"}},
                    "step_states": [
                        {
                            "step_id": "step-1",
                            "step_index": 0,
                            "title": "执行第一步",
                            "description": "执行第一步",
                            "status": ExecutionStatus.PENDING.value,
                        }
                    ],
                }
            },
            "artifacts": ["file-1", "file-1", "file-2"],
        },
    )

    state = GraphStateContractMapper.build_initial_state(
        session=session,
        run=run,
        completed_run_summaries=[
            WorkflowRunSummary(
                run_id="run-completed",
                session_id="session-1",
                status=WorkflowRunStatus.COMPLETED,
                title="已完成运行",
                final_answer_summary="已经输出过结论",
                final_answer_text="这是历史运行里真正交付的完整正文。",
                open_questions=["已完成运行待确认"],
                artifacts=["/tmp/history-artifact.md"],
            )
        ],
        recent_attempt_summaries=[
            WorkflowRunSummary(
                run_id="run-failed",
                session_id="session-1",
                status=WorkflowRunStatus.FAILED,
                title="失败运行",
                final_answer_summary="上一次执行失败",
                open_questions=["失败运行待确认"],
                blockers=["远端接口不可用"],
                artifacts=["/tmp/failed-artifact.md"],
            )
        ],
        session_context_snapshot=SessionContextSnapshot(
            session_id="session-1",
            summary_text="会话级摘要",
            recent_run_briefs=[
                {
                    "run_id": "run-prev",
                    "title": "前序运行",
                    "final_answer_summary": "已经做过调研",
                }
            ],
            open_questions=["还需确认范围"],
            artifact_paths=["/tmp/snapshot-artifact.md"],
        ),
        user_message="你好",
        workspace_id="workspace-1",
        thread_id="thread-1",
    )

    assert state["run_id"] == "run-1"
    assert state["workspace_id"] == "workspace-1"
    assert state["thread_id"] == "thread-1"
    assert state["current_step_id"] == "step-1"
    assert state["message_window"][0]["message"] == "上一次用户输入"
    assert state["message_window"][1]["attachment_paths"] == ["/tmp/final.md"]
    assert state["message_window"][-1]["message"] == "你好"
    assert state["conversation_summary"] == "历史对话摘要"
    assert state["working_memory"]["goal"] == "验证状态契约"
    assert state["task_mode"] == StepTaskModeHint.RESEARCH.value
    assert state["environment_digest"]["task_mode"] == StepTaskModeHint.RESEARCH.value
    assert state["environment_digest"]["payload"]["recent_search_queries"] == ["langgraph persistence"]
    assert state["observation_digest"]["payload"]["last_step_result"] == "已完成上一轮检索"
    assert state["recent_action_digest"]["payload"]["last_user_wait_reason"] == "请补充上下文"
    assert state["retrieved_memories"][0]["id"] == "mem-1"
    assert state["pending_memory_writes"][0]["id"] == "pending-1"
    assert state["recent_run_briefs"][0]["run_id"] == "run-completed"
    assert state["recent_run_briefs"][0]["final_answer_summary"] == "已经输出过结论"
    assert state["recent_run_briefs"][0]["final_answer_text_excerpt"] == "这是历史运行里真正交付的完整正文。"
    assert state["recent_attempt_briefs"][0]["run_id"] == "run-failed"
    assert state["recent_attempt_briefs"][0]["status"] == WorkflowRunStatus.FAILED.value
    assert state["session_open_questions"] == ["已完成运行待确认", "失败运行待确认", "还需确认范围"]
    assert state["session_blockers"] == ["远端接口不可用"]
    assert state["selected_artifacts"] == ["/tmp/final.md"]
    assert state["historical_artifact_paths"] == [
        "/tmp/snapshot-artifact.md",
        "/tmp/history-artifact.md",
        "/tmp/failed-artifact.md",
    ]
    assert len(state["step_states"]) == 1
    assert state["step_states"][0]["step_id"] == "step-1"
    assert state["step_states"][0]["status"] == ExecutionStatus.PENDING.value
    assert sorted(state["step_states"][0].keys()) == [
        "artifact_policy",
        "delivery_context_state",
        "delivery_role",
        "description",
        "output_mode",
        "status",
        "step_id",
        "step_index",
        "task_mode_hint",
        "title",
    ]
    assert state["step_states"][0]["task_mode_hint"] == StepTaskModeHint.RESEARCH.value
    assert state["step_states"][0]["delivery_context_state"] == StepDeliveryContextState.NEEDS_PREPARATION.value
    assert sorted(state["retrieved_memories"][0].keys()) == ["content", "id", "memory_type", "summary", "tags"]
    assert state["pending_interrupt"]["prompt"] == "请补充上下文"
    assert state["graph_metadata"]["control"]["entry_strategy"] == "recall_memory_context"
    runtime_metadata = GraphStateContractMapper.build_runtime_metadata(state)
    assert runtime_metadata["graph_state_contract"]["graph_state"]["workspace_id"] == "workspace-1"


def test_graph_state_contract_should_reopen_cancelled_plan_for_explicit_command() -> None:
    cancelled_plan = Plan(
        title="被取消的任务",
        goal="继续原任务",
        language="zh",
        message="原任务计划",
        steps=[
            Step(
                id="step-1",
                title="第一步",
                description="已完成步骤",
                status=ExecutionStatus.COMPLETED,
                outcome=StepOutcome(done=True, summary="已完成"),
            ),
            Step(
                id="step-2",
                title="第二步",
                description="被取消步骤",
                status=ExecutionStatus.CANCELLED,
                outcome=StepOutcome(done=False, summary="任务已取消"),
            ),
        ],
        status=ExecutionStatus.CANCELLED,
    )
    session = Session(
        id="session-1",
        current_run_id="run-2",
        events=[PlanEvent(plan=cancelled_plan, status=PlanEventStatus.CANCELLED)],
    )
    run = WorkflowRun(
        id="run-2",
        session_id="session-1",
        status=WorkflowRunStatus.RUNNING,
    )

    state = GraphStateContractMapper.build_initial_state(
        session=session,
        run=run,
        completed_run_summaries=[],
        recent_attempt_summaries=[],
        session_context_snapshot=None,
        user_message="",
        continue_cancelled_task=True,
        thread_id="thread-1",
    )

    reopened_plan = state["plan"]
    assert reopened_plan is not None
    assert reopened_plan.status == ExecutionStatus.PENDING
    assert [step.status for step in reopened_plan.steps] == [
        ExecutionStatus.COMPLETED,
        ExecutionStatus.PENDING,
    ]
    assert reopened_plan.steps[1].outcome is None
    assert state["current_step_id"] == "step-2"
    assert state["graph_metadata"]["control"]["continued_from_cancelled_plan"] is True
    assert state["step_states"][1]["status"] == ExecutionStatus.PENDING.value


def test_create_or_reuse_plan_node_should_emit_updated_plan_when_continuing_cancelled_task() -> None:
    state = {
        "session_id": "session-1",
        "user_message": "继续",
        "plan": Plan(
            title="继续原任务",
            goal="继续执行",
            language="zh",
            message="继续原任务",
            steps=[
                Step(id="step-2", description="继续步骤", status=ExecutionStatus.PENDING),
            ],
            status=ExecutionStatus.PENDING,
        ),
        "graph_metadata": {
            "control": {"continued_from_cancelled_plan": True},
        },
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
    }
    emitted_events = []

    async def _sink(event):
        emitted_events.append(event)

    token = bind_live_event_sink(_sink)
    try:
        next_state = asyncio.run(create_or_reuse_plan_node(state, llm=object()))
    finally:
        unbind_live_event_sink(token)

    assert len(emitted_events) == 1
    assert isinstance(emitted_events[0], PlanEvent)
    assert emitted_events[0].status == PlanEventStatus.UPDATED
    assert next_state["current_step_id"] == "step-2"
    assert next_state["graph_metadata"].get("control", {}).get("continued_from_cancelled_plan") is None
    assert len(next_state["emitted_events"]) == 1


def test_graph_state_contract_should_reduce_wait_interrupt_and_generate_runtime_metadata() -> None:
    session = Session(id="session-1", current_run_id="run-1")
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        thread_id="thread-1",
        checkpoint_namespace="",
        checkpoint_id="cp-1",
    )
    created_plan = _build_plan(step_status=ExecutionStatus.PENDING)
    completed_step = created_plan.steps[0].model_copy(deep=True)
    completed_step.status = ExecutionStatus.COMPLETED
    completed_step.outcome = StepOutcome(
        done=True,
        summary="完成第一步",
        produced_artifacts=["/tmp/file-1.md"],
    )

    wait_event = WaitEvent.from_interrupt(
        interrupt_id="interrupt-1",
        payload={
            "kind": "confirm",
            "prompt": "请确认是否继续？",
            "attachments": ["/tmp/reference.md"],
            "suggest_user_takeover": "browser",
            "confirm_label": "继续",
            "cancel_label": "取消",
        },
    )
    tool_event = ToolEvent(
        tool_call_id="call-1",
        tool_name="search",
        function_name="search_web",
        function_args={"q": "langgraph persistence"},
        function_result=ToolResult[dict](success=True, data={"count": 1}),
        status=ToolEventStatus.CALLED,
    )

    state = GraphStateContractMapper.build_initial_state(
        session=session,
        run=run,
        completed_run_summaries=[
            WorkflowRunSummary(
                run_id="run-prev",
                session_id="session-1",
                    status=WorkflowRunStatus.COMPLETED,
                    title="前序运行",
                    final_answer_summary="已完成前置分析",
                    open_questions=["上一轮遗留问题"],
                    artifacts=["/tmp/artifact-prev.md"],
                )
            ],
        recent_attempt_summaries=[
            WorkflowRunSummary(
                run_id="run-failed",
                session_id="session-1",
                status=WorkflowRunStatus.FAILED,
                title="失败尝试",
                final_answer_summary="卡在权限校验",
                open_questions=["是否放宽权限"],
                blockers=["缺少凭证"],
            )
        ],
        session_context_snapshot=None,
        user_message="帮我调研一下",
        thread_id="thread-1",
    )
    state["emitted_events"] = [
        PlanEvent(plan=created_plan, status=PlanEventStatus.CREATED),
        StepEvent(step=completed_step),
        tool_event,
        wait_event,
    ]
    state["message_window"] = [
        {"role": "user", "message": "帮我调研一下", "input_part_count": 0},
        {
            "role": "assistant",
            "message": "计划已生成",
            "attachment_paths": [
                "artifact-id-1",
                "https://example.com/final.md",
                "final.md",
                "/tmp/final.md",
            ],
            "input_part_count": 0,
        },
    ]
    state["conversation_summary"] = "用户希望调研 LangGraph 持久化"
    state["working_memory"] = {
        "goal": "调研 LangGraph 持久化",
        "constraints": ["仅看后端"],
    }
    state["task_mode"] = StepTaskModeHint.RESEARCH.value
    state["environment_digest"] = {
        "task_mode": StepTaskModeHint.RESEARCH.value,
        "payload": {"recent_search_queries": ["langgraph persistence"]},
    }
    state["observation_digest"] = {
        "task_mode": StepTaskModeHint.RESEARCH.value,
        "payload": {"last_step_result": "完成第一步"},
    }
    state["recent_action_digest"] = {
        "task_mode": StepTaskModeHint.RESEARCH.value,
        "payload": {"last_user_wait_reason": "请确认是否继续？"},
    }
    state["retrieved_memories"] = [
        {"id": "mem-1", "summary": "用户偏好中文回复"},
    ]
    state["pending_memory_writes"] = [
        {"id": "pending-1", "summary": "需要沉淀的后端约束"},
    ]
    state["selected_artifacts"] = [
        "artifact-id-1",
        "https://example.com/final.md",
        "final.md",
        "/tmp/final.md",
    ]
    state["graph_metadata"] = {}

    reduced_state = GraphStateContractMapper.apply_emitted_events(state=state)
    runtime_metadata = GraphStateContractMapper.build_runtime_metadata(reduced_state)

    assert reduced_state["current_step_id"] is None
    assert reduced_state["pending_interrupt"]["prompt"] == "请确认是否继续？"
    assert reduced_state["pending_interrupt"]["attachments"] == ["/tmp/reference.md"]
    assert reduced_state["pending_interrupt"]["suggest_user_takeover"] == "browser"

    contract = runtime_metadata["graph_state_contract"]
    memory_metadata = runtime_metadata["memory"]
    assert contract["schema_version"] == GRAPH_STATE_CONTRACT_SCHEMA_VERSION
    assert contract["audit"]["event_count"] == len(state["emitted_events"])
    assert contract["graph_state"]["current_step_id"] is None
    assert contract["graph_state"]["message_window"][0]["message"] == "帮我调研一下"
    assert contract["graph_state"]["message_window"][1]["attachment_paths"] == ["/tmp/final.md"]
    assert contract["graph_state"]["conversation_summary"] == "用户希望调研 LangGraph 持久化"
    assert contract["graph_state"]["working_memory"]["goal"] == "调研 LangGraph 持久化"
    assert contract["graph_state"]["task_mode"] == StepTaskModeHint.RESEARCH.value
    assert contract["graph_state"]["environment_digest"]["task_mode"] == StepTaskModeHint.RESEARCH.value
    assert contract["graph_state"]["environment_digest"]["payload"]["recent_search_queries"] == ["langgraph persistence"]
    assert contract["graph_state"]["observation_digest"]["payload"]["last_step_result"] == "完成第一步"
    assert contract["graph_state"]["recent_action_digest"]["payload"]["last_user_wait_reason"] == "请确认是否继续？"
    assert contract["graph_state"]["retrieved_memories"][0]["id"] == "mem-1"
    assert sorted(contract["graph_state"]["retrieved_memories"][0].keys()) == [
        "content",
        "id",
        "memory_type",
        "summary",
        "tags",
    ]
    assert contract["graph_state"]["pending_memory_writes"][0]["id"] == "pending-1"
    assert contract["graph_state"]["selected_artifacts"] == ["/tmp/final.md"]
    assert sorted(contract["graph_state"]["step_states"][0].keys()) == [
        "artifact_policy",
        "delivery_context_state",
        "delivery_role",
        "description",
        "outcome",
        "output_mode",
        "status",
        "step_id",
        "step_index",
        "task_mode_hint",
        "title",
    ]
    assert sorted(contract["graph_state"]["step_states"][0]["outcome"].keys()) == [
        "produced_artifacts",
        "summary",
    ]
    assert contract["graph_state"]["recent_attempt_briefs"][0]["run_id"] == "run-failed"
    assert contract["graph_state"]["session_blockers"] == ["缺少凭证"]
    assert contract["graph_state"]["historical_artifact_paths"] == ["/tmp/artifact-prev.md"]
    assert contract["graph_state"]["pending_interrupt"]["prompt"] == "请确认是否继续？"
    assert contract["graph_state"]["metadata"]["projection"]["run_status"] == "waiting"
    assert "graph_state_fields" not in contract["planes"]
    assert "task_mode" in contract["planes"]["prompt_visible_fields"]
    assert contract["planes"]["projection_only_fields"] == ["sessions.title/latest_message/status"]
    assert memory_metadata["recall_count"] == 1
    assert memory_metadata["recall_ids"] == ["mem-1"]
    assert memory_metadata["write_count"] == 1
    assert memory_metadata["write_ids"] == ["pending-1"]


def test_build_initial_state_should_rebuild_step_states_from_plan_when_metadata_step_states_are_stale() -> None:
    session = Session(id="session-1", current_run_id="run-1")
    plan = _build_plan()
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        runtime_metadata={
            "graph_state_contract": {
                "graph_state": {
                    "plan": plan.model_dump(mode="json"),
                    "step_states": [
                        {
                            "step_id": "step-1",
                            "step_index": 0,
                            "title": "执行第一步",
                            "description": "执行第一步",
                            "status": ExecutionStatus.PENDING.value,
                        }
                    ],
                }
            }
        },
    )

    state = GraphStateContractMapper.build_initial_state(
        session=session,
        run=run,
        completed_run_summaries=[],
        recent_attempt_summaries=[],
        session_context_snapshot=None,
        user_message="继续",
        thread_id="thread-1",
    )

    assert state["step_states"][0]["task_mode_hint"] == StepTaskModeHint.RESEARCH.value
    assert state["step_states"][0]["output_mode"] == StepOutputMode.NONE.value
    assert state["step_states"][0]["artifact_policy"] == StepArtifactPolicy.FORBID_FILE_OUTPUT.value
    assert state["step_states"][0]["delivery_role"] == StepDeliveryRole.FINAL.value
    assert state["step_states"][0]["delivery_context_state"] == StepDeliveryContextState.NEEDS_PREPARATION.value


def test_graph_state_contract_should_clear_pending_interrupt_after_done() -> None:
    state = {
        "pending_interrupt": {
            "kind": "input_text",
            "prompt": "还需要确认",
            "response_key": "message",
        },
        "graph_metadata": {},
        "step_states": [],
        "emitted_events": [DoneEvent(created_at=datetime(2026, 3, 22, 12, 0, 0))],
    }

    reduced_state = GraphStateContractMapper.apply_emitted_events(state=state)
    runtime_metadata = GraphStateContractMapper.build_runtime_metadata(reduced_state)

    assert reduced_state["pending_interrupt"] == {}
    assert runtime_metadata["graph_state_contract"]["graph_state"]["metadata"]["projection"]["run_status"] == "completed"


def test_apply_emitted_events_should_keep_plan_in_sync_after_step_completed() -> None:
    created_plan = Plan(
        title="测试计划",
        goal="验证步骤同步",
        language="zh",
        message="生成计划",
        steps=[
            Step(
                id="step-1",
                title="第一步",
                description="第一步",
                objective_key="objective-step-1",
                success_criteria=["第一步完成"],
                status=ExecutionStatus.PENDING,
            ),
            Step(
                id="step-2",
                title="第二步",
                description="第二步",
                objective_key="objective-step-2",
                success_criteria=["第二步完成"],
                status=ExecutionStatus.PENDING,
            ),
        ],
    )
    completed_step = created_plan.steps[0].model_copy(deep=True)
    completed_step.status = ExecutionStatus.COMPLETED
    completed_step.outcome = StepOutcome(done=True, summary="第一步完成")
    state = {
        "plan": created_plan.model_copy(deep=True),
        "step_states": [],
        "graph_metadata": {},
        "pending_interrupt": {},
        "emitted_events": [
            PlanEvent(plan=created_plan.model_copy(deep=True), status=PlanEventStatus.CREATED),
            StepEvent(step=completed_step.model_copy(deep=True), status=StepEventStatus.COMPLETED),
        ],
    }

    reduced_state = GraphStateContractMapper.apply_emitted_events(state=state)

    assert reduced_state["current_step_id"] == "step-2"
    assert reduced_state["plan"] is not None
    assert reduced_state["plan"].steps[0].status == ExecutionStatus.COMPLETED
    assert reduced_state["plan"].steps[0].outcome is not None
    assert reduced_state["plan"].steps[0].outcome.summary == "第一步完成"
    assert reduced_state["plan"].steps[1].status == ExecutionStatus.PENDING
    assert reduced_state["plan"].get_next_step() is not None
    assert reduced_state["plan"].get_next_step().id == "step-2"


def test_graph_state_contract_should_normalize_plan_and_last_step_payload_when_building_runtime_metadata() -> None:
    dirty_outcome = StepOutcome(
        done=True,
        summary="结果已生成",
        produced_artifacts=["artifact-id-1", "/tmp/final.md"],
    )
    state = {
        "session_id": "session-1",
        "thread_id": "thread-1",
        "plan": Plan(
            title="生成最终结果",
            goal="输出最终正文",
            steps=[
                Step(
                    id="step-1",
                    title="整理正文",
                    description="整理正文",
                    objective_key="objective-step-1",
                    success_criteria=["输出最终正文"],
                    status=ExecutionStatus.COMPLETED,
                    outcome=dirty_outcome.model_copy(deep=True),
                )
            ],
        ),
        "last_executed_step": Step(
            id="step-1",
            title="整理正文",
            description="整理正文",
            objective_key="objective-step-1",
            success_criteria=["输出最终正文"],
            status=ExecutionStatus.COMPLETED,
            outcome=dirty_outcome.model_copy(deep=True),
        ),
        "step_states": [],
        "message_window": [],
        "conversation_summary": "",
        "working_memory": {},
        "task_mode": "research",
        "environment_digest": {
            "task_mode": "research",
            "payload": {"recent_search_queries": ["langgraph persistence"]},
        },
        "observation_digest": {
            "task_mode": "research",
            "payload": {"last_step_result": "结果已生成"},
        },
        "recent_action_digest": {
            "task_mode": "research",
            "payload": {"last_user_wait_reason": "请确认"},
        },
        "retrieved_memories": [],
        "pending_memory_writes": [],
        "recent_run_briefs": [],
        "recent_attempt_briefs": [],
        "session_open_questions": [],
        "session_blockers": [],
        "selected_artifacts": [],
        "historical_artifact_paths": [],
        "current_step_id": None,
        "execution_count": 1,
        "max_execution_steps": 20,
        "pending_interrupt": {},
        "graph_metadata": {},
        "emitted_events": [],
        "input_parts": [],
        "user_id": None,
        "run_id": None,
        "user_message": "",
        "final_message": "",
    }

    runtime_metadata = GraphStateContractMapper.build_runtime_metadata(state)
    graph_state = runtime_metadata["graph_state_contract"]["graph_state"]

    assert graph_state["plan"]["steps"][0]["outcome"]["produced_artifacts"] == ["/tmp/final.md"]
    assert graph_state["last_executed_step"]["outcome"]["produced_artifacts"] == ["/tmp/final.md"]
    assert graph_state["task_mode"] == StepTaskModeHint.RESEARCH.value
    assert graph_state["environment_digest"]["task_mode"] == StepTaskModeHint.RESEARCH.value
    assert graph_state["environment_digest"]["payload"]["recent_search_queries"] == ["langgraph persistence"]
    assert graph_state["observation_digest"]["payload"]["last_step_result"] == "结果已生成"
    assert graph_state["recent_action_digest"]["payload"]["last_user_wait_reason"] == "请确认"


def test_normalize_runtime_state_should_rebuild_step_states_and_current_step_from_plan() -> None:
    normalized_state = GraphStateContractMapper.normalize_runtime_state(
        {
            "session_id": "session-1",
            "task_mode": "research",
            "environment_digest": {
                "task_mode": "research",
                "payload": {"recent_search_queries": ["langgraph persistence"]},
            },
            "observation_digest": {
                "task_mode": "research",
                "payload": {"last_step_result": "第一步完成"},
            },
            "recent_action_digest": {
                "task_mode": "research",
                "payload": {"last_user_wait_reason": "请确认"},
            },
            "plan": {
                "title": "测试计划",
                "steps": [
                    {
                        "id": "step-1",
                        "title": "第一步",
                        "description": "第一步",
                        "status": ExecutionStatus.COMPLETED.value,
                        "task_mode_hint": StepTaskModeHint.RESEARCH.value,
                        "output_mode": StepOutputMode.NONE.value,
                        "artifact_policy": StepArtifactPolicy.FORBID_FILE_OUTPUT.value,
                        "delivery_role": StepDeliveryRole.INTERMEDIATE.value,
                        "delivery_context_state": StepDeliveryContextState.NONE.value,
                        "outcome": {
                            "done": True,
                            "summary": "第一步完成",
                            "produced_artifacts": ["artifact-id-1", "/tmp/first.md"],
                        },
                    },
                    {
                        "id": "step-2",
                        "title": "第二步",
                        "description": "第二步",
                        "status": ExecutionStatus.PENDING.value,
                        "task_mode_hint": StepTaskModeHint.GENERAL.value,
                        "output_mode": StepOutputMode.INLINE.value,
                        "artifact_policy": StepArtifactPolicy.DEFAULT.value,
                        "delivery_role": StepDeliveryRole.FINAL.value,
                        "delivery_context_state": StepDeliveryContextState.READY.value,
                    },
                ],
            },
            "step_states": [
                {
                    "step_id": "stale-step",
                    "title": "旧步骤",
                    "description": "旧步骤",
                    "status": ExecutionStatus.COMPLETED.value,
                }
            ],
            "current_step_id": "stale-step",
        }
    )

    assert normalized_state["current_step_id"] == "step-2"
    assert len(normalized_state["step_states"]) == 2
    assert normalized_state["step_states"][0]["step_id"] == "step-1"
    assert normalized_state["step_states"][0]["outcome"]["produced_artifacts"] == ["/tmp/first.md"]
    assert normalized_state["step_states"][1]["step_id"] == "step-2"
    assert normalized_state["task_mode"] == StepTaskModeHint.RESEARCH.value
    assert normalized_state["environment_digest"]["task_mode"] == StepTaskModeHint.RESEARCH.value
    assert normalized_state["environment_digest"]["payload"]["recent_search_queries"] == ["langgraph persistence"]
    assert normalized_state["observation_digest"]["payload"]["last_step_result"] == "第一步完成"
    assert normalized_state["recent_action_digest"]["payload"]["last_user_wait_reason"] == "请确认"


def test_build_initial_state_should_not_fallback_to_session_current_run_id_when_run_missing() -> None:
    session = Session(id="session-1", workspace_id="workspace-1", current_run_id="run-legacy")

    state = GraphStateContractMapper.build_initial_state(
        session=session,
        run=None,
        completed_run_summaries=[],
        recent_attempt_summaries=[],
        session_context_snapshot=None,
        user_message="hello",
        workspace_id="workspace-1",
        thread_id="thread-1",
    )

    assert state["run_id"] is None
