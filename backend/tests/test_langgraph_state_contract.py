from datetime import datetime

from app.domain.models import (
    DoneEvent,
    ExecutionStatus,
    File,
    HumanTask,
    HumanTaskResumeCommand,
    HumanTaskResumePoint,
    HumanTaskTimeoutPolicy,
    MessageEvent,
    Plan,
    PlanEvent,
    PlanEventStatus,
    Session,
    Step,
    StepEvent,
    ToolEvent,
    ToolEventStatus,
    ToolResult,
    WaitEvent,
    WorkflowRun,
)
from app.domain.services.runtime.langgraph_state import (
    GRAPH_STATE_CONTRACT_SCHEMA_VERSION,
    GraphStateContractMapper,
    HumanTaskStatus,
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
                description="执行第一步",
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
                    "human_tasks": {
                        "wait:1": {
                            "task_id": "wait:1",
                            "status": "waiting",
                            "reason": "wait_event",
                            "created_at": "2026-03-21T12:00:00",
                            "updated_at": "2026-03-21T12:00:00",
                            "wait_event_id": "evt-wait",
                            "resume_event_id": None,
                        }
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
    assert len(state["step_states"]) == 1
    assert state["step_states"][0]["step_id"] == "step-1"
    assert state["step_states"][0]["status"] == ExecutionStatus.PENDING.value
    assert state["human_tasks"]["wait:1"]["status"] == HumanTaskStatus.WAITING.value
    assert state["tool_invocations"]["call-1"]["tool_name"] == "search"
    assert state["graph_metadata"]["carry_over"] == "yes"
    assert state["artifact_refs"] == ["file-1", "file-2"]


def test_graph_state_contract_should_reduce_emitted_events_and_generate_runtime_metadata() -> None:
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
        description="执行第一步",
        status=ExecutionStatus.COMPLETED,
        success=True,
        result="完成第一步",
    )

    wait_event = WaitEvent.build_for_user_input(
        session_id="session-1",
        question="请确认是否继续？",
        reason="ask_user",
        attachments=["/tmp/reference.md"],
        suggest_user_takeover="browser",
        timeout_seconds=300,
        run_id="run-1",
        thread_id="thread-1",
        checkpoint_namespace="",
        checkpoint_id="cp-1",
        current_step_id="step-1",
        resume_token="resume-token-1",
    )
    user_reply_event = MessageEvent(
        role="user",
        message="我补充一下需求",
        attachments=[File(id="attachment-1", filename="ctx.txt")],
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
        user_message="帮我调研一下",
        thread_id="thread-1",
        checkpoint_namespace="",
        checkpoint_id="cp-1",
    )
    state["emitted_events"] = [
        PlanEvent(plan=created_plan, status=PlanEventStatus.CREATED),
        StepEvent(step=completed_step),
        wait_event,
        user_reply_event,
        tool_event,
        DoneEvent(),
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
    assert reduced_state["artifact_refs"] == ["attachment-1"]
    assert any(
        task.get("status") == HumanTaskStatus.RESUMED.value
        for task in reduced_state["human_tasks"].values()
    )
    resumed_task = next(iter(reduced_state["human_tasks"].values()))
    assert resumed_task["question"] == "请确认是否继续？"
    assert resumed_task["resume_token"] == "resume-token-1"
    assert resumed_task["suggest_user_takeover"] == "browser"
    assert resumed_task["resume_point"]["run_id"] == "run-1"

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
    assert contract["planes"]["projection_only_fields"] == ["sessions.title/latest_message/status"]
    assert memory_metadata["recall_count"] == 1
    assert memory_metadata["recall_ids"] == ["mem-1"]
    assert memory_metadata["write_count"] == 1
    assert memory_metadata["write_ids"] == ["pending-1"]
    assert memory_metadata["compacted"] is True
    assert memory_metadata["last_compaction_at"] == "2026-03-29T12:00:00"
    assert memory_metadata["summary_version"] == "ctx-v2"


def test_graph_state_contract_should_mark_waiting_task_timeout_when_reference_time_passed() -> None:
    wait_event = WaitEvent(
        id="evt-wait-timeout",
        human_task=HumanTask(
            id="human-task-timeout",
            reason="ask_user",
            question="请补充上下文",
            resume_token="resume-timeout",
            resume_command=HumanTaskResumeCommand(
                session_id="session-1",
                resume_token="resume-timeout",
            ),
            resume_point=HumanTaskResumePoint(
                session_id="session-1",
                run_id="run-1",
                thread_id="thread-1",
            ),
            timeout=HumanTaskTimeoutPolicy(
                timeout_seconds=60,
                timeout_at=datetime(2026, 3, 22, 12, 0, 0),
            ),
        ),
        created_at=datetime(2026, 3, 22, 11, 59, 0),
    )

    human_tasks = GraphStateContractMapper.reduce_human_tasks_from_events(
        events=[wait_event],
        reference_at=datetime(2026, 3, 22, 12, 1, 0),
    )

    assert human_tasks["human-task-timeout"]["status"] == HumanTaskStatus.TIMEOUT.value
    assert GraphStateContractMapper.find_latest_waiting_human_task(human_tasks) is None
