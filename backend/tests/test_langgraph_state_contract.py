from datetime import datetime

from app.domain.models import (
    DoneEvent,
    ExecutionStatus,
    File,
    Plan,
    PlanEvent,
    PlanEventStatus,
    Session,
    Step,
    StepEvent,
    StepOutcome,
    ToolEvent,
    ToolEventStatus,
    ToolResult,
    WaitEvent,
    WorkflowRun,
    WorkflowRunStatus,
    WorkflowRunSummary,
    SessionContextSnapshot,
)
from app.domain.services.runtime.langgraph_state import (
    GRAPH_STATE_CONTRACT_SCHEMA_VERSION,
    GraphStateContractMapper,
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
                objective_key="objective-step-1",
                success_criteria=["执行第一步完成"],
                status=step_status,
            )
        ],
        status=ExecutionStatus.PENDING,
    )


def test_graph_state_contract_should_build_initial_state_from_workflow_run_snapshot() -> None:
    session = Session(id="session-1", current_run_id="run-1")
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
                    ],
                    "conversation_summary": "历史对话摘要",
                    "working_memory": {
                        "goal": "验证状态契约",
                        "constraints": ["保持兼容"],
                    },
                    "retrieved_memories": [
                        {"id": "mem-1", "summary": "用户偏好中文"},
                    ],
                    "pending_memory_writes": [
                        {"id": "pending-1", "summary": "待写入事实"},
                    ],
                    "planner_local_memory": {
                        "plan_brief": "已有规划摘要",
                    },
                    "step_local_memory": {
                        "current_step_id": "step-1",
                    },
                    "summary_local_memory": {
                        "answer_outline": "总结提纲",
                    },
                    "memory_context_version": "ctx-v1",
                    "pending_interrupt": {
                        "kind": "input_text",
                        "prompt": "请补充上下文",
                        "attachments": ["/tmp/context.md"],
                        "response_key": "message",
                    },
                    "tool_invocations": {
                        "call-1": {
                            "invocation_id": "call-1",
                            "event_id": "evt-tool",
                            "tool_name": "search",
                            "function_name": "search_web",
                            "status": "called",
                            "function_args": {"q": "langgraph"},
                            "function_result": {"success": True},
                            "created_at": "2026-03-21T12:00:00",
                            "updated_at": "2026-03-21T12:00:00",
                        }
                    },
                    "metadata": {"carry_over": "yes"},
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
                open_questions=["已完成运行待确认"],
                artifacts=["history-artifact"],
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
                artifacts=["failed-artifact"],
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
            artifact_refs=["artifact-from-snapshot"],
        ),
        user_message="你好",
        thread_id="thread-1",
        checkpoint_namespace="",
        checkpoint_id="cp-1",
    )

    assert state["schema_version"] == GRAPH_STATE_CONTRACT_SCHEMA_VERSION
    assert state["run_id"] == "run-1"
    assert state["thread_id"] == "thread-1"
    assert state["current_step_id"] == "step-1"
    assert state["message_window"][0]["message"] == "上一次用户输入"
    assert state["message_window"][-1]["message"] == "你好"
    assert state["conversation_summary"] == "历史对话摘要"
    assert state["working_memory"]["goal"] == "验证状态契约"
    assert state["retrieved_memories"][0]["id"] == "mem-1"
    assert state["pending_memory_writes"][0]["id"] == "pending-1"
    assert state["planner_local_memory"]["plan_brief"] == "已有规划摘要"
    assert state["step_local_memory"]["current_step_id"] == "step-1"
    assert state["summary_local_memory"]["answer_outline"] == "总结提纲"
    assert state["memory_context_version"] == "ctx-v1"
    assert state["recent_run_briefs"][0]["run_id"] == "run-completed"
    assert state["recent_run_briefs"][0]["final_answer_summary"] == "已经输出过结论"
    assert state["recent_attempt_briefs"][0]["run_id"] == "run-failed"
    assert state["recent_attempt_briefs"][0]["status"] == WorkflowRunStatus.FAILED.value
    assert state["session_open_questions"] == ["已完成运行待确认", "失败运行待确认", "还需确认范围"]
    assert state["session_blockers"] == ["远端接口不可用"]
    assert state["selected_artifacts"] == []
    assert state["historical_artifact_refs"] == ["artifact-from-snapshot", "history-artifact", "failed-artifact"]
    assert len(state["step_states"]) == 1
    assert state["step_states"][0]["step_id"] == "step-1"
    assert state["step_states"][0]["objective_key"] == "objective-step-1"
    assert state["step_states"][0]["status"] == ExecutionStatus.PENDING.value
    assert state["pending_interrupt"]["prompt"] == "请补充上下文"
    assert state["tool_invocations"]["call-1"]["tool_name"] == "search"
    assert state["graph_metadata"]["carry_over"] == "yes"
    assert state["artifact_refs"] == ["file-1", "file-2"]


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
    completed_step = Step(
        id="step-1",
        title="执行第一步",
        description="执行第一步",
        objective_key="objective-step-1",
        success_criteria=["执行第一步完成"],
        status=ExecutionStatus.COMPLETED,
        outcome=StepOutcome(
            done=True,
            summary="完成第一步",
            produced_artifacts=["file-1"],
        ),
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
                artifacts=["artifact-prev"],
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
        checkpoint_namespace="",
        checkpoint_id="cp-1",
    )
    state["emitted_events"] = [
        PlanEvent(plan=created_plan, status=PlanEventStatus.CREATED),
        StepEvent(step=completed_step),
        tool_event,
        wait_event,
    ]
    state["message_window"] = [
        {"role": "user", "message": "帮我调研一下", "input_part_count": 0},
        {"role": "assistant", "message": "计划已生成", "input_part_count": 0},
    ]
    state["conversation_summary"] = "用户希望调研 LangGraph 持久化"
    state["working_memory"] = {
        "goal": "调研 LangGraph 持久化",
        "constraints": ["仅看后端"],
    }
    state["retrieved_memories"] = [
        {"id": "mem-1", "summary": "用户偏好中文回复"},
    ]
    state["pending_memory_writes"] = [
        {"id": "pending-1", "summary": "需要沉淀的后端约束"},
    ]
    state["planner_local_memory"] = {"plan_brief": "调研方案"}
    state["step_local_memory"] = {"current_step_id": "step-1"}
    state["summary_local_memory"] = {"answer_outline": "最终答复"}
    state["memory_context_version"] = "ctx-v2"
    state["graph_metadata"] = {
        "memory_compacted": True,
        "memory_last_compaction_at": "2026-03-29T12:00:00",
    }

    reduced_state = GraphStateContractMapper.apply_emitted_events(state=state)
    runtime_metadata = GraphStateContractMapper.build_runtime_metadata(reduced_state)

    assert reduced_state["current_step_id"] is None
    assert len(reduced_state["audit_events"]) == len(state["emitted_events"])
    assert reduced_state["tool_invocations"]["call-1"]["status"] == ToolEventStatus.CALLED.value
    assert reduced_state["artifact_refs"] == ["file-1"]
    assert reduced_state["pending_interrupt"]["prompt"] == "请确认是否继续？"
    assert reduced_state["pending_interrupt"]["attachments"] == ["/tmp/reference.md"]
    assert reduced_state["pending_interrupt"]["suggest_user_takeover"] == "browser"

    contract = runtime_metadata["graph_state_contract"]
    memory_metadata = runtime_metadata["memory"]
    assert contract["schema_version"] == GRAPH_STATE_CONTRACT_SCHEMA_VERSION
    assert contract["audit"]["event_count"] == len(state["emitted_events"])
    assert contract["graph_state"]["current_step_id"] is None
    assert contract["graph_state"]["message_window"][0]["message"] == "帮我调研一下"
    assert contract["graph_state"]["conversation_summary"] == "用户希望调研 LangGraph 持久化"
    assert contract["graph_state"]["working_memory"]["goal"] == "调研 LangGraph 持久化"
    assert contract["graph_state"]["retrieved_memories"][0]["id"] == "mem-1"
    assert contract["graph_state"]["pending_memory_writes"][0]["id"] == "pending-1"
    assert contract["graph_state"]["planner_local_memory"]["plan_brief"] == "调研方案"
    assert contract["graph_state"]["step_local_memory"]["current_step_id"] == "step-1"
    assert contract["graph_state"]["summary_local_memory"]["answer_outline"] == "最终答复"
    assert contract["graph_state"]["memory_context_version"] == "ctx-v2"
    assert contract["graph_state"]["recent_attempt_briefs"][0]["run_id"] == "run-failed"
    assert contract["graph_state"]["session_blockers"] == ["缺少凭证"]
    assert contract["graph_state"]["historical_artifact_refs"] == ["artifact-prev"]
    assert contract["graph_state"]["pending_interrupt"]["prompt"] == "请确认是否继续？"
    assert contract["graph_state"]["metadata"]["pending_interrupts"][0]["interrupt_id"] == "interrupt-1"
    assert contract["planes"]["projection_only_fields"] == ["sessions.title/latest_message/status"]
    assert memory_metadata["recall_count"] == 1
    assert memory_metadata["recall_ids"] == ["mem-1"]
    assert memory_metadata["write_count"] == 1
    assert memory_metadata["write_ids"] == ["pending-1"]
    assert memory_metadata["compacted"] is True
    assert memory_metadata["last_compaction_at"] == "2026-03-29T12:00:00"
    assert memory_metadata["summary_version"] == "ctx-v2"


def test_graph_state_contract_should_clear_pending_interrupt_after_done() -> None:
    state = {
        "schema_version": GRAPH_STATE_CONTRACT_SCHEMA_VERSION,
        "pending_interrupt": {
            "kind": "input_text",
            "prompt": "还需要确认",
            "response_key": "message",
        },
        "graph_metadata": {
            "pending_interrupts": [
                {
                    "interrupt_id": "interrupt-1",
                    "payload": {"kind": "input_text", "prompt": "还需要确认", "response_key": "message"},
                }
            ]
        },
        "step_states": [],
        "tool_invocations": {},
        "artifact_refs": [],
        "audit_events": [],
        "emitted_events": [DoneEvent(created_at=datetime(2026, 3, 22, 12, 0, 0))],
    }

    reduced_state = GraphStateContractMapper.apply_emitted_events(state=state)
    runtime_metadata = GraphStateContractMapper.build_runtime_metadata(reduced_state)

    assert reduced_state["pending_interrupt"] == {}
    assert "pending_interrupts" not in runtime_metadata["graph_state_contract"]["graph_state"]["metadata"]
