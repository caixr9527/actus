#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : langgraph_state.py
"""
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel

from app.domain.models import (
    BaseEvent,
    normalize_wait_payload,
    DoneEvent,
    ErrorEvent,
    ExecutionStatus,
    MessageEvent,
    Plan,
    PlanEvent,
    Session,
    Step,
    StepArtifactPolicy,
    StepDeliveryContextState,
    StepDeliveryRole,
    StepOutcome,
    StepOutputMode,
    StepEvent,
    StepTaskModeHint,
    ToolEvent,
    WaitEvent,
    WorkflowRun,
    WorkflowRunSummary,
    SessionContextSnapshot,
)
from app.domain.services.runtime.normalizers import (
    merge_unique_strings,
    normalize_controlled_value,
    normalize_file_path_list,
    normalize_message_window_entry,
    normalize_plan_payload,
    normalize_ref_list,
    normalize_step_payload,
    normalize_step_outcome_payload,
    normalize_text_list,
)

logger = logging.getLogger(__name__)

# BE-LG-04 契约版本。
# v7 为 P2 增加 task_mode 与 digest 状态面，确保恢复链路与 Prompt 上下文使用同一份摘要真相源。
GRAPH_STATE_CONTRACT_SCHEMA_VERSION = "be-lg-04.v7"


class StepState(TypedDict, total=False):
    """Graph 内部步骤状态快照。"""

    step_id: str
    step_index: int
    title: str
    description: str
    task_mode_hint: str
    output_mode: str
    artifact_policy: str
    delivery_role: str
    delivery_context_state: str
    status: str
    outcome: Optional["StepOutcomeState"]


class StepOutcomeState(TypedDict, total=False):
    """Graph 内部步骤结果快照，仅保留真实消费字段。"""

    summary: str
    produced_artifacts: List[str]
    blockers: List[str]
    facts_learned: List[str]
    open_questions: List[str]


class RetrievedMemoryState(TypedDict, total=False):
    """Graph 内部召回记忆快照，仅保留 prompt/runtime 真正消费的字段。"""

    id: str
    memory_type: str
    summary: str
    content: Dict[str, Any]
    tags: List[str]


class RuntimeDigestState(TypedDict, total=False):
    """按 task_mode 包装的运行时 digest。"""

    task_mode: str
    payload: Dict[str, Any]


class GraphControlState(TypedDict, total=False):
    """图运行中的控制状态。"""

    entry_strategy: str
    skip_replan_when_plan_finished: bool
    step_reuse_hit: bool
    wait_resume_action: str
    continued_from_cancelled_plan: bool
    direct_wait_original_message: str
    direct_wait_execute_task_mode: str
    direct_wait_original_task_executed: bool


class GraphProjectionState(TypedDict, total=False):
    """图运行结束后的投影状态。"""

    run_status: str


class GraphMetadataState(TypedDict, total=False):
    """Graph 内部元状态，仅保留控制与投影字段。"""

    control: GraphControlState
    projection: GraphProjectionState


def normalize_graph_metadata(raw: Any) -> GraphMetadataState:
    """标准化 graph metadata，仅保留 control/projection 两个平面。"""
    if not isinstance(raw, dict):
        return {}

    metadata: GraphMetadataState = {}
    raw_control = raw.get("control")
    if isinstance(raw_control, dict):
        control: GraphControlState = {}
        entry_strategy = str(raw_control.get("entry_strategy") or "").strip()
        if entry_strategy:
            control["entry_strategy"] = entry_strategy
        if "skip_replan_when_plan_finished" in raw_control:
            control["skip_replan_when_plan_finished"] = bool(raw_control.get("skip_replan_when_plan_finished"))
        if "step_reuse_hit" in raw_control:
            control["step_reuse_hit"] = bool(raw_control.get("step_reuse_hit"))
        wait_resume_action = str(raw_control.get("wait_resume_action") or "").strip()
        if wait_resume_action:
            control["wait_resume_action"] = wait_resume_action
        if "continued_from_cancelled_plan" in raw_control:
            control["continued_from_cancelled_plan"] = bool(raw_control.get("continued_from_cancelled_plan"))
        direct_wait_original_message = str(raw_control.get("direct_wait_original_message") or "").strip()
        if direct_wait_original_message:
            control["direct_wait_original_message"] = direct_wait_original_message
        direct_wait_execute_task_mode = str(raw_control.get("direct_wait_execute_task_mode") or "").strip()
        if direct_wait_execute_task_mode:
            control["direct_wait_execute_task_mode"] = direct_wait_execute_task_mode
        if "direct_wait_original_task_executed" in raw_control:
            control["direct_wait_original_task_executed"] = bool(raw_control.get("direct_wait_original_task_executed"))
        if control:
            metadata["control"] = control

    raw_projection = raw.get("projection")
    if isinstance(raw_projection, dict):
        projection: GraphProjectionState = {}
        run_status = str(raw_projection.get("run_status") or "").strip()
        if run_status:
            projection["run_status"] = run_status
        if projection:
            metadata["projection"] = projection

    return metadata


def get_graph_control(metadata: Any) -> GraphControlState:
    normalized = normalize_graph_metadata(metadata)
    return dict(normalized.get("control") or {})


def get_graph_projection(metadata: Any) -> GraphProjectionState:
    normalized = normalize_graph_metadata(metadata)
    return dict(normalized.get("projection") or {})


def replace_graph_control(metadata: Any, control: Dict[str, Any]) -> GraphMetadataState:
    normalized = normalize_graph_metadata(metadata)
    next_metadata: GraphMetadataState = {}
    next_control = dict(control or {})
    next_projection = dict(normalized.get("projection") or {})
    if next_control:
        next_metadata["control"] = next_control
    if next_projection:
        next_metadata["projection"] = next_projection
    return next_metadata


def replace_graph_projection(metadata: Any, projection: Dict[str, Any]) -> GraphMetadataState:
    normalized = normalize_graph_metadata(metadata)
    next_metadata: GraphMetadataState = {}
    next_control = dict(normalized.get("control") or {})
    next_projection = dict(projection or {})
    if next_control:
        next_metadata["control"] = next_control
    if next_projection:
        next_metadata["projection"] = next_projection
    return next_metadata


class PlannerReActLangGraphState(TypedDict, total=False):
    """LangGraph 状态对象"""

    """当前会话 ID；用于绑定 session 级上下文、记忆命名空间和运行投影。"""
    session_id: str
    """当前用户 ID；用于用户级长期记忆读写和运行归属。"""
    user_id: Optional[str]
    """当前运行 ID；用于状态回写、事件归并和运行内复用标识。"""
    run_id: Optional[str]
    """LangGraph 线程 ID；用于 checkpoint 隔离与恢复。"""
    thread_id: str
    """当前轮用户输入文本；用于入口路由、规划、执行和总结提示词。"""
    user_message: str
    """当前轮输入附件/多模态片段；用于构建多模态提示和附件上下文。"""
    input_parts: List[Dict[str, Any]]
    """当前线程的消息窗口快照；用于 prompt 上下文拼装和会话摘要压缩。"""
    message_window: List[Dict[str, Any]]
    """当前线程的压缩对话摘要；用于跨多轮保留核心上下文。"""
    conversation_summary: str
    """当前运行的工作记忆；用于沉淀 goal、事实、偏好、开放问题等可变上下文。"""
    working_memory: Dict[str, Any]
    """当前阶段采用的真实任务模式；作为上下文工程与执行路由的主索引字段。"""
    task_mode: str
    """环境层摘要；用于 Prompt 快速理解当前环境状态，而不是回放原始轨迹。"""
    environment_digest: RuntimeDigestState
    """最近有效观察摘要；用于 Prompt 快速理解已经看到了什么。"""
    observation_digest: RuntimeDigestState
    """最近动作与失败/等待摘要；用于 Prompt 快速理解刚做过什么、不该再做什么。"""
    recent_action_digest: RuntimeDigestState
    """本轮召回的长期记忆快照；用于 prompt 注入和偏好提取。"""
    retrieved_memories: List[RetrievedMemoryState]
    """待写入长期记忆仓库的候选项；用于总结后统一治理、落库或重试。"""
    pending_memory_writes: List[Dict[str, Any]]
    """同会话近期成功运行摘要；用于补充已完成历史上下文。"""
    recent_run_briefs: List[Dict[str, Any]]
    """同会话近期失败/取消运行摘要；用于补充失败尝试与阻塞上下文。"""
    recent_attempt_briefs: List[Dict[str, Any]]
    """会话级待解决问题集合；用于 prompt 上下文和工作记忆初始化。"""
    session_open_questions: List[str]
    """会话级阻塞项集合；用于 prompt 上下文补充历史受阻信息。"""
    session_blockers: List[str]
    """当前运行明确选中/确认的产物引用；用于总结输出和历史上下文投影。"""
    selected_artifacts: List[str]
    """历史运行产物引用；用于当前运行 prompt 中补充跨轮产物上下文。"""
    historical_artifact_refs: List[str]
    """当前运行的计划快照；承载步骤编排，是执行主路径的核心状态。"""
    plan: Optional[Plan]
    """当前待执行或等待恢复的步骤 ID；是 wait/cancel/projection 的快捷指针，不是步骤真相源。"""
    current_step_id: Optional[str]
    """当前运行已完成的步骤执行次数；用于执行上限控制和总结展示。"""
    execution_count: int
    """当前运行允许的最大步骤执行次数；用于防止图内无限循环。"""
    max_execution_steps: int
    """最近一次执行完成的步骤快照；用于总结、产物回填和后续重规划。"""
    last_executed_step: Optional[Step]
    """步骤运行态 ledger；从 plan 派生，用于 prompt/summary/projection，不独立承载步骤真相。"""
    step_states: List[StepState]
    """当前待恢复的等待态载荷；用于 wait 路由、恢复校验和继续执行。"""
    pending_interrupt: Dict[str, Any]
    """图内控制/投影元状态；用于路由控制和 run_status 投影。"""
    graph_metadata: GraphMetadataState
    """当前运行从事件层收敛出的原始产物引用集合；仅作附件兜底池，不等于最终交付附件。"""
    artifact_refs: List[str]
    """当前运行最新的面向用户结果文本；用于投影、总结和消息窗口收敛。"""
    final_message: str
    """当前 graph 已发射但尚未完成外部归并的事件序列；用于状态归并和流式事件去重。"""
    emitted_events: List[BaseEvent]


class GraphStateContractMapper:
    """Graph State 与领域对象映射器。"""

    GRAPH_STATE_FIELDS: tuple[str, ...] = (
        "session_id",
        "user_id",
        "run_id",
        "thread_id",
        "user_message",
        "input_parts",
        "message_window",
        "conversation_summary",
        "working_memory",
        "task_mode",
        "environment_digest",
        "observation_digest",
        "recent_action_digest",
        "retrieved_memories",
        "pending_memory_writes",
        "recent_run_briefs",
        "recent_attempt_briefs",
        "session_open_questions",
        "session_blockers",
        "selected_artifacts",
        "historical_artifact_refs",
        "plan",
        "current_step_id",
        "execution_count",
        "max_execution_steps",
        "last_executed_step",
        "step_states",
        "pending_interrupt",
        "graph_metadata",
    )

    PROJECTION_ONLY_FIELDS: tuple[str, ...] = (
        "sessions.title/latest_message/status",
    )

    AUDIT_ONLY_FIELDS: tuple[str, ...] = (
        "workflow_run_events.event_payload",
        "workflow_run_events.event_id",
        "workflow_run_events.event_type",
        "workflow_run_events.created_at",
    )

    ARTIFACT_ONLY_FIELDS: tuple[str, ...] = (
        "generated_file_ids",
        "browser_screenshot_urls",
        "message_attachment_ids",
    )

    @staticmethod
    def _to_iso(dt: Optional[datetime]) -> str:
        if dt is None:
            return datetime.now().isoformat()
        return dt.isoformat()

    @classmethod
    def _to_json_safe(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(k): cls._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._to_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [cls._to_json_safe(item) for item in value]
        return value

    @classmethod
    def _step_to_state(cls, step: Step, step_index: int) -> StepState:
        step_state: StepState = {
            "step_id": step.id,
            "step_index": step_index,
            "title": step.title,
            "description": step.description,
            "status": step.status.value,
        }
        step_state.update(cls._normalize_step_control_state(step))
        normalized_outcome = cls._normalize_step_outcome_state(step.outcome)
        if normalized_outcome is not None:
            step_state["outcome"] = normalized_outcome
        return step_state

    @classmethod
    def _normalize_step_control_state(cls, raw: Any) -> Dict[str, str]:
        """统一规整步骤结构化语义，保证 plan、step_states、恢复链路使用同一套字段。"""
        if isinstance(raw, Step):
            raw_values = {
                "task_mode_hint": getattr(raw, "task_mode_hint", None),
                "output_mode": getattr(raw, "output_mode", None),
                "artifact_policy": getattr(raw, "artifact_policy", None),
                "delivery_role": getattr(raw, "delivery_role", None),
                "delivery_context_state": getattr(raw, "delivery_context_state", None),
            }
        elif isinstance(raw, dict):
            raw_values = {
                "task_mode_hint": raw.get("task_mode_hint"),
                "output_mode": raw.get("output_mode"),
                "artifact_policy": raw.get("artifact_policy"),
                "delivery_role": raw.get("delivery_role"),
                "delivery_context_state": raw.get("delivery_context_state"),
            }
        else:
            return {}

        normalized_values = {
            "task_mode_hint": normalize_controlled_value(raw_values.get("task_mode_hint"), StepTaskModeHint),
            "output_mode": normalize_controlled_value(raw_values.get("output_mode"), StepOutputMode),
            "artifact_policy": normalize_controlled_value(raw_values.get("artifact_policy"), StepArtifactPolicy),
            "delivery_role": normalize_controlled_value(raw_values.get("delivery_role"), StepDeliveryRole),
            "delivery_context_state": normalize_controlled_value(
                raw_values.get("delivery_context_state"),
                StepDeliveryContextState,
            ),
        }
        return {
            key: value
            for key, value in normalized_values.items()
            if value
        }

    @classmethod
    def _normalize_step_outcome_state(cls, raw: Any) -> Optional[StepOutcomeState]:
        normalized_payload = normalize_step_outcome_payload(raw)
        if normalized_payload is None:
            return None

        normalized_outcome = {}
        summary = str(normalized_payload.get("summary") or "").strip()
        if summary:
            normalized_outcome["summary"] = summary
        produced_artifacts = normalize_file_path_list(normalized_payload.get("produced_artifacts"))
        if produced_artifacts:
            normalized_outcome["produced_artifacts"] = produced_artifacts
        blockers = normalize_text_list(normalized_payload.get("blockers"))
        if blockers:
            normalized_outcome["blockers"] = blockers
        facts_learned = normalize_text_list(normalized_payload.get("facts_learned"))
        if facts_learned:
            normalized_outcome["facts_learned"] = facts_learned
        open_questions = normalize_text_list(normalized_payload.get("open_questions"))
        if open_questions:
            normalized_outcome["open_questions"] = open_questions
        return normalized_outcome or None

    @classmethod
    def _normalize_step_states(cls, raw: Any) -> List[StepState]:
        if not isinstance(raw, list):
            return []
        normalized_states: List[StepState] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            step_id = str(item.get("step_id") or "").strip()
            if not step_id:
                continue
            normalized_state: StepState = {
                "step_id": step_id,
                "step_index": int(item.get("step_index", index) or index),
                "title": str(item.get("title") or "").strip(),
                "description": str(item.get("description") or "").strip(),
                "status": str(item.get("status") or "").strip(),
            }
            normalized_state.update(cls._normalize_step_control_state(item))
            normalized_outcome = cls._normalize_step_outcome_state(item.get("outcome"))
            if normalized_outcome is not None:
                normalized_state["outcome"] = normalized_outcome
            normalized_states.append(normalized_state)
        return normalized_states

    @classmethod
    def _normalize_retrieved_memory(cls, raw: Any) -> Optional[RetrievedMemoryState]:
        if not isinstance(raw, dict):
            return None
        memory_id = str(raw.get("id") or "").strip()
        if not memory_id:
            return None

        content = raw.get("content") if isinstance(raw.get("content"), dict) else {}
        tags = normalize_text_list(raw.get("tags"))
        return {
            "id": memory_id,
            "memory_type": str(raw.get("memory_type") or "").strip(),
            "summary": str(raw.get("summary") or "").strip(),
            "content": cls._normalize_dict_memory(content),
            "tags": tags,
        }

    @classmethod
    def _normalize_retrieved_memories(cls, raw: Any) -> List[RetrievedMemoryState]:
        if not isinstance(raw, list):
            return []
        normalized_items: List[RetrievedMemoryState] = []
        for item in raw:
            normalized_item = cls._normalize_retrieved_memory(item)
            if normalized_item is not None:
                normalized_items.append(normalized_item)
        return normalized_items

    @classmethod
    def _build_step_states_from_plan(cls, plan: Optional[Plan]) -> List[StepState]:
        if plan is None:
            return []
        return [cls._step_to_state(step=step, step_index=index) for index, step in enumerate(plan.steps)]

    @classmethod
    def _resolve_plan_snapshot(
            cls,
            session: Session,
            run: Optional[WorkflowRun],
    ) -> Optional[Plan]:
        graph_state = cls._extract_contract_graph_state(run=run)
        graph_plan = graph_state.get("plan")
        if isinstance(graph_plan, dict) and graph_plan:
            try:
                return Plan.model_validate(graph_plan)
            except Exception as e:
                logger.warning(
                    "运行[%s]graph_state_contract.plan反序列化失败，回退Session事件中的计划: %s",
                    run.id if run is not None else "",
                    e,
                )

        latest_plan = session.get_latest_plan()
        return latest_plan.model_copy(deep=True) if latest_plan is not None else None

    @classmethod
    def _extract_contract_graph_state(
            cls,
            run: Optional[WorkflowRun],
    ) -> Dict[str, Any]:
        if run is None:
            return {}
        contract = (run.runtime_metadata or {}).get("graph_state_contract")
        if not isinstance(contract, dict):
            return {}
        graph_state = contract.get("graph_state")
        if not isinstance(graph_state, dict):
            return {}
        return dict(graph_state)

    @staticmethod
    def _normalize_text(raw: Any) -> str:
        return str(raw or "")

    @staticmethod
    def _should_clear_cancellation_outcome(outcome: Optional[StepOutcome]) -> bool:
        if outcome is None:
            return False
        return (
                str(outcome.summary or "").strip() == "任务已取消"
                and len(list(outcome.produced_artifacts or [])) == 0
                and len(list(outcome.blockers or [])) == 0
                and len(list(outcome.facts_learned or [])) == 0
                and len(list(outcome.open_questions or [])) == 0
                and not str(outcome.next_hint or "").strip()
                and not str(outcome.reused_from_run_id or "").strip()
                and not str(outcome.reused_from_step_id or "").strip()
        )

    @classmethod
    def _reopen_cancelled_plan_for_continuation(cls, plan: Plan) -> Optional[Plan]:
        if plan.status != ExecutionStatus.CANCELLED:
            return None

        reopened_plan = plan.model_copy(deep=True)
        reopened_any_step = False
        for step in reopened_plan.steps:
            if step.status != ExecutionStatus.CANCELLED:
                continue
            step.status = ExecutionStatus.PENDING
            if cls._should_clear_cancellation_outcome(step.outcome):
                step.outcome = None
            reopened_any_step = True

        if not reopened_any_step:
            return None

        reopened_plan.status = ExecutionStatus.PENDING
        return reopened_plan

    @classmethod
    def _normalize_dict_memory(cls, raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        return cls._to_json_safe(raw)

    @classmethod
    def _normalize_task_mode(cls, raw: Any) -> str:
        return normalize_controlled_value(raw, StepTaskModeHint)

    @classmethod
    def _normalize_runtime_digest(cls, raw: Any) -> RuntimeDigestState:
        """统一规整 digest 包装结构，避免旧字段残留再次流回主链。"""
        if not isinstance(raw, dict):
            return {}

        task_mode = cls._normalize_task_mode(raw.get("task_mode"))
        payload = cls._normalize_dict_memory(raw.get("payload"))
        if not task_mode and not payload:
            return {}

        normalized_digest: RuntimeDigestState = {}
        if task_mode:
            normalized_digest["task_mode"] = task_mode
        if payload:
            normalized_digest["payload"] = payload
        return normalized_digest

    @classmethod
    def _normalize_list_memory(cls, raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in raw:
            if isinstance(item, dict):
                normalized.append(cls._to_json_safe(item))
        return normalized

    @classmethod
    def _normalize_input_parts(cls, raw: Any) -> List[Dict[str, Any]]:
        return cls._normalize_list_memory(raw)

    @classmethod
    def _normalize_message_window(cls, raw: Any) -> List[Dict[str, Any]]:
        normalized_window: List[Dict[str, Any]] = []
        for item in cls._normalize_list_memory(raw):
            normalized_entry = normalize_message_window_entry(item, default_role="assistant")
            if normalized_entry is None:
                continue
            normalized_window.append(normalized_entry)
        return normalized_window

    @classmethod
    def _normalize_recent_run_briefs(cls, raw: Any) -> List[Dict[str, Any]]:
        normalized_briefs: List[Dict[str, Any]] = []
        for item in cls._normalize_list_memory(raw):
            run_id = str(item.get("run_id") or "").strip()
            if not run_id:
                continue
            normalized_briefs.append(
                {
                    "run_id": run_id,
                    "title": str(item.get("title") or "").strip(),
                    "goal": str(item.get("goal") or "").strip(),
                    "status": str(item.get("status") or "").strip(),
                    "final_answer_summary": str(item.get("final_answer_summary") or "").strip(),
                    "final_answer_text_excerpt": str(item.get("final_answer_text_excerpt") or "").strip(),
                }
            )
        return normalized_briefs

    @classmethod
    def _normalize_recent_attempt_briefs(cls, raw: Any) -> List[Dict[str, Any]]:
        return cls._normalize_recent_run_briefs(raw)

    @staticmethod
    def _truncate_brief_summary(raw: Any, *, max_chars: int = 200) -> str:
        normalized = str(raw or "").strip()
        if len(normalized) <= max_chars:
            return normalized
        return f"{normalized[:max_chars].rstrip()}..."

    @classmethod
    def _build_briefs_from_summaries(
            cls,
            summaries: Optional[List[WorkflowRunSummary]],
    ) -> List[Dict[str, Any]]:
        briefs: List[Dict[str, Any]] = []
        for summary in list(summaries or []):
            if summary is None:
                continue
            briefs.append(
                {
                    "run_id": summary.run_id,
                    "title": summary.title,
                    "goal": summary.goal,
                    "status": summary.status.value,
                    "final_answer_summary": cls._truncate_brief_summary(summary.final_answer_summary),
                    "final_answer_text_excerpt": cls._truncate_brief_summary(summary.final_answer_text),
                }
            )
        return briefs

    @classmethod
    def _build_open_questions_from_summaries(
            cls,
            summaries: Optional[List[WorkflowRunSummary]],
    ) -> List[str]:
        questions: List[str] = []
        for summary in list(summaries or []):
            if summary is None:
                continue
            questions = merge_unique_strings(questions, getattr(summary, "open_questions", []))
        return questions

    @classmethod
    def _build_blockers_from_summaries(
            cls,
            summaries: Optional[List[WorkflowRunSummary]],
    ) -> List[str]:
        blockers: List[str] = []
        for summary in list(summaries or []):
            if summary is None:
                continue
            blockers = merge_unique_strings(blockers, getattr(summary, "blockers", []))
        return blockers

    @classmethod
    def _build_artifact_refs_from_summaries(
            cls,
            summaries: Optional[List[WorkflowRunSummary]],
    ) -> List[str]:
        artifact_refs: List[str] = []
        for summary in list(summaries or []):
            if summary is None:
                continue
            artifact_refs = merge_unique_strings(artifact_refs, normalize_ref_list(getattr(summary, "artifacts", [])))
        return artifact_refs

    @classmethod
    def _normalize_pending_interrupt(cls, raw: Any) -> Dict[str, Any]:
        return normalize_wait_payload(raw)

    @classmethod
    def _normalize_graph_metadata(cls, raw: Any) -> GraphMetadataState:
        return normalize_graph_metadata(raw)

    @classmethod
    def get_pending_interrupt(
            cls,
            state: Optional[PlannerReActLangGraphState],
    ) -> Dict[str, Any]:
        """统一读取并标准化当前 checkpoint 中的等待态。"""
        if not isinstance(state, dict):
            return {}
        return cls._normalize_pending_interrupt(state.get("pending_interrupt"))

    @classmethod
    def _normalize_last_executed_step(cls, raw: Any) -> Optional[Step]:
        if not isinstance(raw, dict) or not raw:
            return None
        try:
            step = Step.model_validate(raw)
        except Exception:
            return None
        normalized_outcome = normalize_step_outcome_payload(step.outcome)
        if normalized_outcome is not None:
            step.outcome = StepOutcome.model_validate(normalized_outcome)
        return step

    @classmethod
    def _append_current_user_message(
            cls,
            message_window: List[Dict[str, Any]],
            user_message: str,
            input_parts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        normalized_message = str(user_message or "").strip()
        if not normalized_message and not input_parts:
            return list(message_window)

        updated_window = list(message_window)
        next_message = {
            "role": "user",
            "message": user_message,
            "input_part_count": len(input_parts),
        }
        if updated_window:
            latest_message = dict(updated_window[-1])
            if (
                    latest_message.get("role") == next_message["role"]
                    and latest_message.get("message") == next_message["message"]
                    and int(latest_message.get("input_part_count") or 0) == next_message["input_part_count"]
            ):
                return updated_window

        updated_window.append(next_message)
        return updated_window

    @staticmethod
    def _extract_memory_ids(items: List[Dict[str, Any]]) -> List[str]:
        ids: List[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            memory_id = item.get("id")
            if memory_id is None:
                continue
            memory_id_str = str(memory_id).strip()
            if memory_id_str:
                ids.append(memory_id_str)
        return list(dict.fromkeys(ids))

    @classmethod
    def _extract_pending_interrupt_from_metadata(
            cls,
            graph_state_from_metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return cls._normalize_pending_interrupt(graph_state_from_metadata.get("pending_interrupt"))

    @classmethod
    def build_initial_state(
            cls,
            session: Session,
            run: Optional[WorkflowRun],
            completed_run_summaries: Optional[List[WorkflowRunSummary]],
            recent_attempt_summaries: Optional[List[WorkflowRunSummary]],
            session_context_snapshot: Optional[SessionContextSnapshot],
            user_message: str,
            input_parts: Optional[List[Dict[str, Any]]] = None,
            continue_cancelled_task: bool = False,
            thread_id: str = "",
    ) -> PlannerReActLangGraphState:
        """构建 BE-LG-04 契约化初始状态。"""
        graph_state_from_metadata = cls._extract_contract_graph_state(run=run)
        plan = cls._resolve_plan_snapshot(session=session, run=run)
        plan_resumed_from_cancelled = False
        if (
                plan is not None
                and len(graph_state_from_metadata) == 0
                and continue_cancelled_task
        ):
            reopened_plan = cls._reopen_cancelled_plan_for_continuation(plan)
            if reopened_plan is not None:
                plan = reopened_plan
                plan_resumed_from_cancelled = True

        raw_step_states = graph_state_from_metadata.get("step_states")
        if plan is not None:
            # 恢复时统一以 plan 为步骤真相源，避免持久化 step_states 与 plan 发生语义漂移。
            step_states = cls._build_step_states_from_plan(plan=plan)
        elif isinstance(raw_step_states, list) and not plan_resumed_from_cancelled:
            step_states = cls._normalize_step_states(raw_step_states)
        else:
            step_states = []

        current_step_id = graph_state_from_metadata.get("current_step_id")
        if current_step_id is not None:
            current_step_id = str(current_step_id)
        elif plan is not None:
            next_step = plan.get_next_step()
            current_step_id = next_step.id if next_step is not None else None

        recent_run_briefs = cls._normalize_recent_run_briefs(graph_state_from_metadata.get("recent_run_briefs"))
        if not recent_run_briefs:
            recent_run_briefs = cls._build_briefs_from_summaries(completed_run_summaries)
        if not recent_run_briefs:
            recent_run_briefs = cls._normalize_recent_run_briefs(
                getattr(session_context_snapshot, "recent_run_briefs", None)
            )

        recent_attempt_briefs = cls._normalize_recent_attempt_briefs(
            graph_state_from_metadata.get("recent_attempt_briefs")
        )
        if not recent_attempt_briefs:
            recent_attempt_briefs = cls._build_briefs_from_summaries(recent_attempt_summaries)

        session_open_questions = normalize_text_list(graph_state_from_metadata.get("session_open_questions"))
        if not session_open_questions:
            session_open_questions = merge_unique_strings(
                cls._build_open_questions_from_summaries(completed_run_summaries),
                cls._build_open_questions_from_summaries(recent_attempt_summaries),
                normalize_text_list(getattr(session_context_snapshot, "open_questions", None)),
            )

        session_blockers = normalize_text_list(graph_state_from_metadata.get("session_blockers"))
        if not session_blockers:
            session_blockers = cls._build_blockers_from_summaries(recent_attempt_summaries)

        # selected_artifacts 只承载最终可交付文件路径，避免历史普通引用污染附件链路。
        selected_artifacts = normalize_file_path_list(graph_state_from_metadata.get("selected_artifacts"))

        historical_artifact_refs = normalize_ref_list(graph_state_from_metadata.get("historical_artifact_refs"))
        if not historical_artifact_refs:
            historical_artifact_refs = merge_unique_strings(
                normalize_ref_list(getattr(session_context_snapshot, "artifact_refs", None)),
                cls._build_artifact_refs_from_summaries(completed_run_summaries),
                cls._build_artifact_refs_from_summaries(recent_attempt_summaries),
            )

        conversation_summary = cls._normalize_text(graph_state_from_metadata.get("conversation_summary"))
        if not conversation_summary and session_context_snapshot is not None:
            conversation_summary = cls._normalize_text(session_context_snapshot.summary_text)

        graph_metadata = cls._normalize_graph_metadata(graph_state_from_metadata.get("metadata"))
        if plan_resumed_from_cancelled:
            control = dict(graph_metadata.get("control") or {})
            control["continued_from_cancelled_plan"] = True
            graph_metadata["control"] = control

        return {
            "session_id": session.id,
            "user_id": (
                run.user_id
                if run is not None and run.user_id is not None
                else session.user_id
            ),
            "run_id": run.id if run is not None else session.current_run_id,
            "thread_id": thread_id,
            "user_message": user_message,
            "input_parts": cls._normalize_input_parts(input_parts),
            "message_window": cls._append_current_user_message(
                message_window=cls._normalize_message_window(graph_state_from_metadata.get("message_window")),
                user_message=user_message,
                input_parts=cls._normalize_input_parts(input_parts),
            ),
            "conversation_summary": conversation_summary,
            "working_memory": cls._normalize_dict_memory(graph_state_from_metadata.get("working_memory")),
            "task_mode": cls._normalize_task_mode(graph_state_from_metadata.get("task_mode")),
            "environment_digest": cls._normalize_runtime_digest(graph_state_from_metadata.get("environment_digest")),
            "observation_digest": cls._normalize_runtime_digest(graph_state_from_metadata.get("observation_digest")),
            "recent_action_digest": cls._normalize_runtime_digest(graph_state_from_metadata.get("recent_action_digest")),
            "retrieved_memories": cls._normalize_retrieved_memories(
                graph_state_from_metadata.get("retrieved_memories")),
            "pending_memory_writes": cls._normalize_list_memory(graph_state_from_metadata.get("pending_memory_writes")),
            "recent_run_briefs": recent_run_briefs,
            "recent_attempt_briefs": recent_attempt_briefs,
            "session_open_questions": session_open_questions,
            "session_blockers": session_blockers,
            "selected_artifacts": selected_artifacts,
            "historical_artifact_refs": historical_artifact_refs,
            "plan": plan,
            "current_step_id": current_step_id,
            "execution_count": int(graph_state_from_metadata.get("execution_count") or 0),
            "max_execution_steps": int(graph_state_from_metadata.get("max_execution_steps") or 20),
            "last_executed_step": cls._normalize_last_executed_step(
                graph_state_from_metadata.get("last_executed_step")
            ),
            "step_states": step_states,
            "pending_interrupt": cls._extract_pending_interrupt_from_metadata(graph_state_from_metadata),
            "graph_metadata": graph_metadata,
            "artifact_refs": normalize_ref_list(
                (run.runtime_metadata or {}).get("artifacts") if run is not None else []
            ),
            "emitted_events": [],
            "final_message": "",
        }

    @classmethod
    def _upsert_step_state(cls, step_states: List[StepState], step: Step) -> List[StepState]:
        step_id = str(step.id)
        updated = list(step_states)
        for index, current in enumerate(updated):
            if str(current.get("step_id")) == step_id:
                step_index = int(current.get("step_index", index))
                updated[index] = cls._step_to_state(step=step, step_index=step_index)
                return updated

        updated.append(cls._step_to_state(step=step, step_index=len(updated)))
        return updated

    @classmethod
    def _get_step_index_from_states(cls, step_states: List[StepState], step_id: str) -> Optional[int]:
        """优先复用现有步骤顺序，避免 StepEvent 破坏 plan 中的稳定索引。"""
        for index, step_state in enumerate(step_states):
            if str(step_state.get("step_id") or "") == step_id:
                return int(step_state.get("step_index", index))
        return None

    @classmethod
    def _upsert_step_into_plan(
            cls,
            plan: Optional[Plan],
            step: Step,
            *,
            step_states: List[StepState],
    ) -> Optional[Plan]:
        """把 StepEvent 的最新状态同步回 plan，避免 plan 与 step_states 分叉。"""
        if plan is None:
            return None

        updated_plan = plan.model_copy(deep=True)
        step_id = str(step.id)
        for index, current in enumerate(updated_plan.steps):
            if str(current.id) == step_id:
                updated_plan.steps[index] = step.model_copy(deep=True)
                return updated_plan

        insert_index = cls._get_step_index_from_states(step_states=step_states, step_id=step_id)
        normalized_index = len(updated_plan.steps) if insert_index is None else max(0, min(insert_index,
                                                                                           len(updated_plan.steps)))
        updated_plan.steps.insert(normalized_index, step.model_copy(deep=True))
        return updated_plan

    @classmethod
    def _derive_current_step_id_from_plan(cls, plan: Optional[Plan]) -> Optional[str]:
        """统一以 plan 快照推导当前步骤，避免 current_step_id 与 plan 各自漂移。"""
        if plan is None:
            return None
        next_step = plan.get_next_step()
        if next_step is None:
            return None
        return str(next_step.id or "") or None

    @classmethod
    def _extract_artifact_refs_from_event(cls, event: BaseEvent) -> List[str]:
        refs: List[str] = []

        if isinstance(event, MessageEvent):
            for attachment in event.attachments:
                if attachment.id:
                    refs.append(str(attachment.id))

        if isinstance(event, StepEvent):
            step_outcome = event.step.outcome
            if step_outcome is not None:
                refs.extend(
                    [
                        str(item)
                        for item in list(step_outcome.produced_artifacts or [])
                        if str(item).strip()
                    ]
                )

        if isinstance(event, ToolEvent) and event.tool_content is not None:
            screenshot = getattr(event.tool_content, "screenshot", None)
            if isinstance(screenshot, str) and screenshot.strip():
                refs.append(screenshot.strip())

        return refs

    @classmethod
    def apply_emitted_events(cls, state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        """根据 emitted events 收敛 step/tool/audit 状态。"""
        events = list(state.get("emitted_events") or [])
        next_state: PlannerReActLangGraphState = dict(state)
        plan = next_state.get("plan")
        step_states = cls._normalize_step_states(next_state.get("step_states"))
        graph_metadata = cls._normalize_graph_metadata(next_state.get("graph_metadata"))
        control = dict(graph_metadata.get("control") or {})
        projection = dict(graph_metadata.get("projection") or {})
        artifact_refs = list(next_state.get("artifact_refs") or [])
        pending_interrupt = cls._normalize_pending_interrupt(next_state.get("pending_interrupt"))
        waiting_for_replan = control.get("wait_resume_action") == "replan"

        for event in events:
            artifact_refs.extend(cls._extract_artifact_refs_from_event(event))

            if isinstance(event, PlanEvent):
                plan = event.plan.model_copy(deep=True)
                step_states = cls._build_step_states_from_plan(plan)
                next_state["current_step_id"] = None if waiting_for_replan else cls._derive_current_step_id_from_plan(
                    plan)
                continue

            if isinstance(event, StepEvent):
                step_states = cls._upsert_step_state(step_states=step_states, step=event.step)
                plan = cls._upsert_step_into_plan(plan=plan, step=event.step, step_states=step_states)
                next_state["current_step_id"] = None if waiting_for_replan else cls._derive_current_step_id_from_plan(
                    plan)
                continue

            if isinstance(event, WaitEvent):
                pending_interrupt = cls._normalize_pending_interrupt(event.payload)
                projection["run_status"] = "waiting"
                continue

            if isinstance(event, ErrorEvent):
                projection["run_status"] = "failed"
                continue

            if isinstance(event, DoneEvent):
                projection["run_status"] = "completed"
                pending_interrupt = {}

        next_state["plan"] = plan
        if not next_state.get("current_step_id") and not waiting_for_replan:
            next_state["current_step_id"] = cls._derive_current_step_id_from_plan(plan)
        if not next_state.get("current_step_id") and not waiting_for_replan:
            for step_state in step_states:
                if str(step_state.get("status")) == ExecutionStatus.RUNNING.value:
                    next_state["current_step_id"] = str(step_state.get("step_id") or "")
                    break
            if not next_state.get("current_step_id"):
                for step_state in step_states:
                    if str(step_state.get("status")) == ExecutionStatus.PENDING.value:
                        next_state["current_step_id"] = str(step_state.get("step_id") or "")
                        break

        next_state["step_states"] = step_states
        next_state["pending_interrupt"] = pending_interrupt
        next_graph_metadata: GraphMetadataState = {}
        if control:
            next_graph_metadata["control"] = control
        if projection:
            next_graph_metadata["projection"] = projection
        next_state["graph_metadata"] = next_graph_metadata
        next_state["artifact_refs"] = normalize_ref_list(artifact_refs)
        return next_state

    @classmethod
    def normalize_runtime_state(cls, raw: Any) -> PlannerReActLangGraphState:
        """统一规整 checkpoint/runtime 恢复态，避免旧脏值再次流回主链。"""
        if not isinstance(raw, dict):
            return {}

        normalized_state: PlannerReActLangGraphState = dict(raw)
        normalized_state["session_id"] = cls._normalize_text(raw.get("session_id"))
        normalized_state["user_id"] = (
            cls._normalize_text(raw.get("user_id"))
            if raw.get("user_id") is not None
            else None
        )
        normalized_state["run_id"] = (
            cls._normalize_text(raw.get("run_id"))
            if raw.get("run_id") is not None
            else None
        )
        normalized_state["thread_id"] = cls._normalize_text(raw.get("thread_id"))
        normalized_state["user_message"] = cls._normalize_text(raw.get("user_message"))
        normalized_state["input_parts"] = cls._normalize_input_parts(raw.get("input_parts"))
        normalized_state["message_window"] = cls._normalize_message_window(raw.get("message_window"))
        normalized_state["conversation_summary"] = cls._normalize_text(raw.get("conversation_summary"))
        normalized_state["working_memory"] = cls._normalize_dict_memory(raw.get("working_memory"))
        normalized_state["task_mode"] = cls._normalize_task_mode(raw.get("task_mode"))
        normalized_state["environment_digest"] = cls._normalize_runtime_digest(raw.get("environment_digest"))
        normalized_state["observation_digest"] = cls._normalize_runtime_digest(raw.get("observation_digest"))
        normalized_state["recent_action_digest"] = cls._normalize_runtime_digest(raw.get("recent_action_digest"))
        normalized_state["retrieved_memories"] = cls._normalize_retrieved_memories(raw.get("retrieved_memories"))
        normalized_state["pending_memory_writes"] = cls._normalize_list_memory(raw.get("pending_memory_writes"))
        normalized_state["recent_run_briefs"] = cls._normalize_recent_run_briefs(raw.get("recent_run_briefs"))
        normalized_state["recent_attempt_briefs"] = cls._normalize_recent_attempt_briefs(
            raw.get("recent_attempt_briefs")
        )
        normalized_state["session_open_questions"] = normalize_text_list(raw.get("session_open_questions"))
        normalized_state["session_blockers"] = normalize_text_list(raw.get("session_blockers"))
        normalized_state["selected_artifacts"] = normalize_file_path_list(raw.get("selected_artifacts"))
        normalized_state["historical_artifact_refs"] = normalize_ref_list(raw.get("historical_artifact_refs"))

        normalized_plan = normalize_plan_payload(raw.get("plan"))
        plan = Plan.model_validate(normalized_plan) if normalized_plan is not None else None
        normalized_state["plan"] = plan
        normalized_state["execution_count"] = int(raw.get("execution_count") or 0)
        normalized_state["max_execution_steps"] = int(raw.get("max_execution_steps") or 20)

        normalized_last_step = normalize_step_payload(raw.get("last_executed_step"))
        normalized_state["last_executed_step"] = (
            Step.model_validate(normalized_last_step)
            if normalized_last_step is not None
            else None
        )
        if plan is not None:
            # 恢复态一律以 plan 为步骤真相源，避免 stale step_states/current_step_id 再次写回 metadata。
            normalized_state["step_states"] = cls._build_step_states_from_plan(plan)
            normalized_state["current_step_id"] = cls._derive_current_step_id_from_plan(plan)
        else:
            normalized_state["step_states"] = cls._normalize_step_states(raw.get("step_states"))
            current_step_id = raw.get("current_step_id")
            normalized_state["current_step_id"] = (
                cls._normalize_text(current_step_id)
                if current_step_id is not None
                else None
            )
        normalized_state["pending_interrupt"] = cls._normalize_pending_interrupt(raw.get("pending_interrupt"))
        normalized_state["graph_metadata"] = cls._normalize_graph_metadata(
            raw.get("graph_metadata") if raw.get("graph_metadata") is not None else raw.get("metadata")
        )
        normalized_state["artifact_refs"] = normalize_ref_list(raw.get("artifact_refs"))
        normalized_state["final_message"] = cls._normalize_text(raw.get("final_message"))
        normalized_state["emitted_events"] = list(raw.get("emitted_events") or [])
        return normalized_state

    @classmethod
    def build_runtime_metadata(cls, state: PlannerReActLangGraphState) -> Dict[str, Any]:
        """将 graph state 收敛为 WorkflowRun.runtime_metadata。"""
        events = list(state.get("emitted_events") or [])
        last_event = events[-1] if events else None
        pending_interrupt = cls._normalize_pending_interrupt(state.get("pending_interrupt"))
        graph_metadata = cls._normalize_graph_metadata(state.get("graph_metadata"))
        step_states = cls._normalize_step_states(state.get("step_states"))
        retrieved_memories = cls._normalize_retrieved_memories(state.get("retrieved_memories"))

        return {
            "memory": {
                "recall_count": len(retrieved_memories),
                "recall_ids": cls._extract_memory_ids(retrieved_memories),
                "write_count": len(state.get("pending_memory_writes") or []),
                "write_ids": cls._extract_memory_ids(list(state.get("pending_memory_writes") or [])),
            },
            "graph_state_contract": {
                "schema_version": GRAPH_STATE_CONTRACT_SCHEMA_VERSION,
                "planes": {
                    "graph_state_fields": list(cls.GRAPH_STATE_FIELDS),
                    "projection_only_fields": list(cls.PROJECTION_ONLY_FIELDS),
                    "audit_only_fields": list(cls.AUDIT_ONLY_FIELDS),
                    "artifact_only_fields": list(cls.ARTIFACT_ONLY_FIELDS),
                },
                "graph_state": {
                    "session_id": state.get("session_id"),
                    "user_id": state.get("user_id"),
                    "run_id": state.get("run_id"),
                    "thread_id": state.get("thread_id"),
                    "input_parts": cls._to_json_safe(state.get("input_parts") or []),
                    "message_window": cls._to_json_safe(cls._normalize_message_window(state.get("message_window"))),
                    "conversation_summary": str(state.get("conversation_summary") or ""),
                    "working_memory": cls._to_json_safe(state.get("working_memory") or {}),
                    "task_mode": cls._normalize_task_mode(state.get("task_mode")),
                    "environment_digest": cls._to_json_safe(
                        cls._normalize_runtime_digest(state.get("environment_digest"))
                    ),
                    "observation_digest": cls._to_json_safe(
                        cls._normalize_runtime_digest(state.get("observation_digest"))
                    ),
                    "recent_action_digest": cls._to_json_safe(
                        cls._normalize_runtime_digest(state.get("recent_action_digest"))
                    ),
                    "retrieved_memories": cls._to_json_safe(retrieved_memories),
                    "pending_memory_writes": cls._to_json_safe(state.get("pending_memory_writes") or []),
                    "recent_run_briefs": cls._to_json_safe(state.get("recent_run_briefs") or []),
                    "recent_attempt_briefs": cls._to_json_safe(state.get("recent_attempt_briefs") or []),
                    "session_open_questions": cls._to_json_safe(state.get("session_open_questions") or []),
                    "session_blockers": cls._to_json_safe(state.get("session_blockers") or []),
                    "selected_artifacts": cls._to_json_safe(
                        normalize_file_path_list(state.get("selected_artifacts"))
                    ),
                    "historical_artifact_refs": cls._to_json_safe(state.get("historical_artifact_refs") or []),
                    "plan": cls._to_json_safe(normalize_plan_payload(state.get("plan"))),
                    "current_step_id": state.get("current_step_id"),
                    "execution_count": int(state.get("execution_count") or 0),
                    "max_execution_steps": int(state.get("max_execution_steps") or 20),
                    "last_executed_step": cls._to_json_safe(
                        normalize_step_payload(state.get("last_executed_step"))
                    ),
                    "step_states": cls._to_json_safe(step_states),
                    "pending_interrupt": cls._to_json_safe(pending_interrupt),
                    "metadata": cls._to_json_safe(graph_metadata),
                },
                "audit": {
                    "event_count": len(events),
                    "last_event_id": str(last_event.id) if last_event is not None else None,
                    "last_event_type": last_event.type if last_event is not None else None,
                    "last_event_at": cls._to_iso(last_event.created_at) if last_event is not None else None,
                },
            },
            "artifacts": cls._to_json_safe(state.get("artifact_refs") or []),
        }


def normalize_retrieved_memories(raw: Any) -> List[RetrievedMemoryState]:
    """统一裁剪 graph state 中的召回记忆快照。"""
    return GraphStateContractMapper._normalize_retrieved_memories(raw)
