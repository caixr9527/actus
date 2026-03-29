#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : langgraph_state.py
"""
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict

from pydantic import BaseModel

from app.domain.models import (
    BaseEvent,
    DoneEvent,
    ErrorEvent,
    ExecutionStatus,
    MessageEvent,
    Plan,
    PlanEvent,
    Session,
    Step,
    StepEvent,
    ToolEvent,
    WaitEvent,
    WorkflowRun,
)

logger = logging.getLogger(__name__)

# BE-LG-04 约定版本号。
# 后续契约发生结构化变更时，必须升级该版本并做兼容迁移策略。
GRAPH_STATE_CONTRACT_SCHEMA_VERSION = "be-lg-04.v1"


class HumanTaskStatus(str, Enum):
    """人机协作任务状态。"""

    WAITING = "waiting"
    RESUMED = "resumed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class StepState(TypedDict, total=False):
    """Graph 内部步骤状态快照。"""

    step_id: str
    step_index: int
    description: str
    status: str
    result: Optional[str]
    error: Optional[str]
    success: bool
    attachments: List[str]


class HumanTaskState(TypedDict, total=False):
    """Graph 内部人机协作任务状态。"""

    task_id: str
    status: str
    reason: str
    question: str
    attachments: List[str]
    suggest_user_takeover: str
    resume_token: Optional[str]
    resume_command: Dict[str, Any]
    resume_point: Dict[str, Any]
    timeout_seconds: Optional[int]
    timeout_at: Optional[str]
    created_at: str
    updated_at: str
    wait_event_id: Optional[str]
    resume_event_id: Optional[str]


class ToolInvocationState(TypedDict, total=False):
    """Graph 内部工具调用状态。"""

    invocation_id: str
    event_id: str
    tool_name: str
    function_name: str
    status: str
    function_args: Dict[str, Any]
    function_result: Optional[Any]
    error: Optional[str]
    created_at: str
    updated_at: str

class PlannerReActLangGraphState(TypedDict, total=False):
    """LangGraph 状态对象（BE-LG-04 契约版本）。"""

    schema_version: str  # 状态契约版本号，用于跨版本兼容与迁移判断。
    session_id: str  # 会话ID，标识当前对话上下文归属。
    run_id: Optional[str]  # 运行ID，对应 workflow_runs 主记录，可为空（新建前）。
    thread_id: str  # LangGraph/checkpoint 线程ID，用于恢复同一执行链路。
    checkpoint_ref_namespace: str  # checkpoint 命名空间，支持多图/多环境隔离。
    checkpoint_ref_id: Optional[str]  # 最近一次 checkpoint 引用ID，用于断点续跑。
    user_message: str  # 本轮用户输入的纯文本主消息。
    input_parts: List[Dict[str, Any]]  # 本轮统一输入片段（text/image/file/audio/video 等）。
    plan: Plan  # 当前执行计划快照（标题、步骤、状态等）。
    current_step_id: Optional[str]  # 当前正在执行或即将执行的步骤ID。
    execution_count: int  # 已执行步骤轮次计数，用于循环收敛与保护。
    max_execution_steps: int  # 最大允许执行步数，防止无限循环。
    last_executed_step: Optional[Step]  # 最近一次执行完成的步骤快照，供 replan/summarize 使用。
    step_states: List[StepState]  # 步骤状态平铺快照，便于投影与查询。
    human_tasks: Dict[str, HumanTaskState]  # 人机协作任务集合（wait/resume/timeout）。
    tool_invocations: Dict[str, ToolInvocationState]  # 工具调用轨迹集合（参数、结果、状态）。
    graph_metadata: Dict[str, Any]  # 图运行元信息（如 input_policy、调试扩展字段）。
    artifact_refs: List[str]  # 产物引用列表（文件ID、URL、附件引用等）。
    audit_events: List[BaseEvent]  # 审计事件缓存，仅用于追踪/复盘，不参与决策。
    final_message: str  # 当前已确定的最终回复候选文本。
    emitted_events: List[BaseEvent]  # 已发射事件序列，供回放/去重/最终落库。
    error: Optional[str]  # 图执行错误信息（可选），用于失败态透出与诊断。


class GraphStateContractMapper:
    """BE-LG-04：Graph State 与领域对象映射器。"""

    # graph state plane：允许进入 LangGraph checkpoint 的字段。
    GRAPH_STATE_FIELDS: tuple[str, ...] = (
        "schema_version",
        "session_id",
        "run_id",
        "thread_id",
        "checkpoint_ref_namespace",
        "checkpoint_ref_id",
        "user_message",
        "input_parts",
        "plan",
        "current_step_id",
        "execution_count",
        "max_execution_steps",
        "step_states",
        "human_tasks",
        "tool_invocations",
        "graph_metadata",
        "artifact_refs",
    )

    # projection plane：只作为查询投影，不进入 graph state。
    PROJECTION_ONLY_FIELDS: tuple[str, ...] = (
        "workflow_runs.plan_snapshot",
        "workflow_runs.files_snapshot",
        "workflow_runs.memories_snapshot",
        "sessions.title/latest_message/status",
    )

    # audit plane：只做审计与追踪，不参与图内决策状态。
    AUDIT_ONLY_FIELDS: tuple[str, ...] = (
        "workflow_run_events.event_payload",
        "workflow_run_events.event_id",
        "workflow_run_events.event_type",
        "workflow_run_events.created_at",
    )

    # artifact plane：产物引用，不进入 graph 主状态。
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
        """递归转换为可落 JSONB 的结构。"""
        if value is None:
            return None
        if isinstance(value, BaseModel):
            return value.model_dump(mode="json")
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(k): cls._to_json_safe(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._to_json_safe(item) for item in value]
        if isinstance(value, tuple):
            return [cls._to_json_safe(item) for item in value]
        return value

    @staticmethod
    def _step_to_state(step: Step, step_index: int) -> StepState:
        return {
            "step_id": step.id,
            "step_index": step_index,
            "description": step.description,
            "status": step.status.value,
            "result": step.result,
            "error": step.error,
            "success": step.success,
            "attachments": list(step.attachments or []),
        }

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
        # 优先使用 run 主线中的 plan 快照，保证 durable run 语义优先。
        if run is not None and run.plan_snapshot:
            try:
                return Plan.model_validate(run.plan_snapshot)
            except Exception as e:
                logger.warning("运行[%s]plan_snapshot反序列化失败，回退Session事件中的计划: %s", run.id, e)

        # run 不可用时回退到 session 事件投影。
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

    @classmethod
    def _normalize_human_tasks(cls, raw: Any) -> Dict[str, HumanTaskState]:
        if not isinstance(raw, dict):
            return {}
        normalized: Dict[str, HumanTaskState] = {}
        for task_id, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            raw_attachments = payload.get("attachments")
            if isinstance(raw_attachments, str):
                attachments = [raw_attachments] if raw_attachments.strip() else []
            elif isinstance(raw_attachments, list):
                attachments = [str(item) for item in raw_attachments if str(item).strip()]
            else:
                attachments = []
            normalized[str(task_id)] = {
                "task_id": str(payload.get("task_id") or task_id),
                "status": str(payload.get("status") or HumanTaskStatus.WAITING.value),
                "reason": str(payload.get("reason") or ""),
                "question": str(payload.get("question") or ""),
                "attachments": attachments,
                "suggest_user_takeover": str(payload.get("suggest_user_takeover") or "none"),
                "resume_token": str(payload.get("resume_token")) if payload.get("resume_token") else None,
                "resume_command": (
                    payload.get("resume_command")
                    if isinstance(payload.get("resume_command"), dict)
                    else {}
                ),
                "resume_point": (
                    payload.get("resume_point")
                    if isinstance(payload.get("resume_point"), dict)
                    else {}
                ),
                "timeout_seconds": (
                    int(payload.get("timeout_seconds"))
                    if payload.get("timeout_seconds") is not None
                    else None
                ),
                "timeout_at": str(payload.get("timeout_at")) if payload.get("timeout_at") else None,
                "created_at": str(payload.get("created_at") or datetime.now().isoformat()),
                "updated_at": str(payload.get("updated_at") or datetime.now().isoformat()),
                "wait_event_id": str(payload.get("wait_event_id")) if payload.get("wait_event_id") else None,
                "resume_event_id": str(payload.get("resume_event_id")) if payload.get("resume_event_id") else None,
            }
        return normalized

    @classmethod
    def _normalize_tool_invocations(cls, raw: Any) -> Dict[str, ToolInvocationState]:
        if not isinstance(raw, dict):
            return {}
        normalized: Dict[str, ToolInvocationState] = {}
        for invocation_id, payload in raw.items():
            if not isinstance(payload, dict):
                continue
            normalized[str(invocation_id)] = {
                "invocation_id": str(payload.get("invocation_id") or invocation_id),
                "event_id": str(payload.get("event_id") or ""),
                "tool_name": str(payload.get("tool_name") or ""),
                "function_name": str(payload.get("function_name") or ""),
                "status": str(payload.get("status") or ""),
                "function_args": payload.get("function_args") if isinstance(payload.get("function_args"), dict) else {},
                "function_result": cls._to_json_safe(payload.get("function_result")),
                "error": str(payload.get("error")) if payload.get("error") else None,
                "created_at": str(payload.get("created_at") or datetime.now().isoformat()),
                "updated_at": str(payload.get("updated_at") or datetime.now().isoformat()),
            }
        return normalized

    @staticmethod
    def _normalize_artifact_refs(raw: Any) -> List[str]:
        if not isinstance(raw, list):
            return []
        refs: List[str] = []
        for item in raw:
            if isinstance(item, str) and item.strip():
                refs.append(item.strip())
        # 保持顺序去重，避免重复刷写 runtime_metadata。
        return list(dict.fromkeys(refs))

    @classmethod
    def _normalize_input_parts(cls, raw: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw, list):
            return []
        normalized: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            normalized.append(cls._to_json_safe(item))
        return normalized

    @classmethod
    def build_initial_state(
            cls,
            session: Session,
            run: Optional[WorkflowRun],
            user_message: str,
            input_parts: Optional[List[Dict[str, Any]]] = None,
            thread_id: str = "",
            checkpoint_namespace: str = "",
            checkpoint_id: Optional[str] = None,
    ) -> PlannerReActLangGraphState:
        """构建 BE-LG-04 契约化初始状态。"""
        plan = cls._resolve_plan_snapshot(session=session, run=run)
        graph_state_from_metadata = cls._extract_contract_graph_state(run=run)

        step_states = graph_state_from_metadata.get("step_states")
        if not isinstance(step_states, list):
            step_states = cls._build_step_states_from_plan(plan=plan)

        current_step_id = graph_state_from_metadata.get("current_step_id")
        if current_step_id is not None:
            current_step_id = str(current_step_id)
        elif plan is not None:
            next_step = plan.get_next_step()
            current_step_id = next_step.id if next_step is not None else None

        state: PlannerReActLangGraphState = {
            "schema_version": GRAPH_STATE_CONTRACT_SCHEMA_VERSION,
            "session_id": session.id,
            "run_id": run.id if run is not None else session.current_run_id,
            "thread_id": thread_id,
            "checkpoint_ref_namespace": checkpoint_namespace,
            "checkpoint_ref_id": checkpoint_id,
            "user_message": user_message,
            "input_parts": cls._normalize_input_parts(input_parts),
            "plan": plan,
            "current_step_id": current_step_id,
            "execution_count": int(graph_state_from_metadata.get("execution_count") or 0),
            "max_execution_steps": int(graph_state_from_metadata.get("max_execution_steps") or 20),
            "last_executed_step": None,
            "step_states": step_states,
            "human_tasks": cls._normalize_human_tasks(graph_state_from_metadata.get("human_tasks")),
            "tool_invocations": cls._normalize_tool_invocations(graph_state_from_metadata.get("tool_invocations")),
            "graph_metadata": (
                graph_state_from_metadata.get("metadata")
                if isinstance(graph_state_from_metadata.get("metadata"), dict)
                else {}
            ),
            "artifact_refs": cls._normalize_artifact_refs(
                (run.runtime_metadata or {}).get("artifacts") if run is not None else []
            ),
            "audit_events": [],
            "emitted_events": [],
            "final_message": "",
            "error": None,
        }
        state["human_tasks"] = cls._mark_timeout_tasks(
            human_tasks=dict(state.get("human_tasks") or {}),
            reference_at=datetime.now(),
        )
        return state

    @classmethod
    def _upsert_step_state(cls, step_states: List[StepState], step: Step) -> List[StepState]:
        """按 step_id 更新步骤状态；不存在时追加。"""
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
    def _mark_latest_waiting_task_resumed(
            cls,
            human_tasks: Dict[str, HumanTaskState],
            resume_event_id: str,
            updated_at: str,
    ) -> Dict[str, HumanTaskState]:
        # 采用“最近等待任务优先恢复”的策略，兼容当前 wait/resume 线性语义。
        for task_id in reversed(list(human_tasks.keys())):
            task = dict(human_tasks[task_id])
            if task.get("status") == HumanTaskStatus.WAITING.value:
                task["status"] = HumanTaskStatus.RESUMED.value
                task["resume_event_id"] = resume_event_id
                task["updated_at"] = updated_at
                human_tasks[task_id] = task
                break
        return human_tasks

    @staticmethod
    def _parse_iso_datetime(raw: Any) -> Optional[datetime]:
        if not raw:
            return None
        if isinstance(raw, datetime):
            return raw
        try:
            return datetime.fromisoformat(str(raw))
        except Exception:
            return None

    @classmethod
    def _mark_timeout_tasks(
            cls,
            human_tasks: Dict[str, HumanTaskState],
            reference_at: datetime,
    ) -> Dict[str, HumanTaskState]:
        updated = dict(human_tasks)
        for task_id, payload in list(updated.items()):
            task = dict(payload)
            if task.get("status") != HumanTaskStatus.WAITING.value:
                continue
            timeout_at = cls._parse_iso_datetime(task.get("timeout_at"))
            if timeout_at is None or timeout_at > reference_at:
                continue
            task["status"] = HumanTaskStatus.TIMEOUT.value
            task["updated_at"] = reference_at.isoformat()
            updated[task_id] = task
        return updated

    @classmethod
    def _extract_artifact_refs_from_event(cls, event: BaseEvent) -> List[str]:
        refs: List[str] = []

        if isinstance(event, MessageEvent):
            for attachment in event.attachments:
                if attachment.id:
                    refs.append(str(attachment.id))

        if isinstance(event, StepEvent):
            refs.extend([str(item) for item in list(event.step.attachments or []) if str(item).strip()])

        if isinstance(event, ToolEvent) and event.tool_content is not None:
            screenshot = getattr(event.tool_content, "screenshot", None)
            if isinstance(screenshot, str) and screenshot.strip():
                refs.append(screenshot.strip())

        return refs

    @classmethod
    def apply_emitted_events(cls, state: PlannerReActLangGraphState) -> PlannerReActLangGraphState:
        """根据 emitted events 收敛 step/human/tool/audit 状态。"""
        events = list(state.get("emitted_events") or [])
        next_state: PlannerReActLangGraphState = dict(state)
        step_states = list(next_state.get("step_states") or [])
        human_tasks = dict(next_state.get("human_tasks") or {})
        tool_invocations = dict(next_state.get("tool_invocations") or {})
        graph_metadata = dict(next_state.get("graph_metadata") or {})
        artifact_refs = list(next_state.get("artifact_refs") or [])

        audit_events = list(next_state.get("audit_events") or [])
        seen_event_ids = {str(event.id) for event in audit_events}

        for event in events:
            artifact_refs.extend(cls._extract_artifact_refs_from_event(event))

            if str(event.id) not in seen_event_ids:
                audit_events.append(event)
                seen_event_ids.add(str(event.id))

            if isinstance(event, PlanEvent):
                next_state["plan"] = event.plan.model_copy(deep=True)
                step_states = cls._build_step_states_from_plan(next_state["plan"])
                next_step = event.plan.get_next_step()
                next_state["current_step_id"] = next_step.id if next_step is not None else None
                graph_metadata["latest_plan_event_id"] = str(event.id)
                continue

            if isinstance(event, StepEvent):
                step_states = cls._upsert_step_state(step_states=step_states, step=event.step)
                if event.step.status == ExecutionStatus.RUNNING:
                    next_state["current_step_id"] = event.step.id
                elif event.step.done and next_state.get("current_step_id") == event.step.id:
                    next_state["current_step_id"] = None
                graph_metadata["latest_step_event_id"] = str(event.id)
                continue

            if isinstance(event, ToolEvent):
                invocation_id = event.tool_call_id or str(event.id)
                function_result = (
                    event.function_result.model_dump(mode="json")
                    if event.function_result is not None
                    else None
                )
                tool_invocations[invocation_id] = {
                    "invocation_id": invocation_id,
                    "event_id": str(event.id),
                    "tool_name": event.tool_name,
                    "function_name": event.function_name,
                    "status": event.status.value,
                    "function_args": cls._to_json_safe(event.function_args or {}),
                    "function_result": cls._to_json_safe(function_result),
                    "error": None if event.function_result is not None else "tool_result_missing",
                    "created_at": cls._to_iso(event.created_at),
                    "updated_at": cls._to_iso(event.created_at),
                }
                continue

            if isinstance(event, WaitEvent):
                task = event.human_task
                task_id = str(task.id) if task is not None else f"wait:{event.id}"
                human_tasks[task_id] = {
                    "task_id": task_id,
                    "status": HumanTaskStatus.WAITING.value,
                    "reason": task.reason if task is not None else "wait_event",
                    "question": task.question if task is not None else "",
                    "attachments": list(task.attachments or []) if task is not None else [],
                    "suggest_user_takeover": (
                        task.suggest_user_takeover if task is not None else "none"
                    ),
                    "resume_token": task.resume_token if task is not None else None,
                    "resume_command": (
                        task.resume_command.model_dump(mode="json")
                        if task is not None and task.resume_command is not None
                        else {}
                    ),
                    "resume_point": (
                        task.resume_point.model_dump(mode="json")
                        if task is not None and task.resume_point is not None
                        else {}
                    ),
                    "timeout_seconds": (
                        task.timeout.timeout_seconds
                        if task is not None
                        else None
                    ),
                    "timeout_at": (
                        cls._to_iso(task.timeout.timeout_at)
                        if task is not None and task.timeout.timeout_at is not None
                        else None
                    ),
                    "created_at": cls._to_iso(event.created_at),
                    "updated_at": cls._to_iso(event.created_at),
                    "wait_event_id": str(event.id),
                    "resume_event_id": None,
                }
                graph_metadata["latest_wait_event_id"] = str(event.id)
                graph_metadata["latest_human_task_id"] = task_id
                continue

            if isinstance(event, MessageEvent) and event.role == "user":
                human_tasks = cls._mark_timeout_tasks(
                    human_tasks=human_tasks,
                    reference_at=event.created_at,
                )
                human_tasks = cls._mark_latest_waiting_task_resumed(
                    human_tasks=human_tasks,
                    resume_event_id=str(event.id),
                    updated_at=cls._to_iso(event.created_at),
                )
                continue

            if isinstance(event, ErrorEvent):
                graph_metadata["run_status"] = "failed"
                graph_metadata["last_error"] = event.error
                graph_metadata["last_error_event_id"] = str(event.id)
                continue

            if isinstance(event, DoneEvent):
                graph_metadata["run_status"] = "completed"
                graph_metadata["last_done_event_id"] = str(event.id)

        reference_at = events[-1].created_at if events else datetime.now()
        human_tasks = cls._mark_timeout_tasks(
            human_tasks=human_tasks,
            reference_at=reference_at,
        )

        if not next_state.get("current_step_id"):
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
        next_state["human_tasks"] = human_tasks
        next_state["tool_invocations"] = tool_invocations
        next_state["graph_metadata"] = graph_metadata
        next_state["artifact_refs"] = list(dict.fromkeys(artifact_refs))
        next_state["audit_events"] = audit_events
        return next_state

    @classmethod
    def reduce_human_tasks_from_events(
            cls,
            events: List[BaseEvent],
            reference_at: Optional[datetime] = None,
    ) -> Dict[str, HumanTaskState]:
        reduced = cls.apply_emitted_events(
            state={
                "emitted_events": list(events or []),
                "human_tasks": {},
                "step_states": [],
                "tool_invocations": {},
                "graph_metadata": {},
                "artifact_refs": [],
                "audit_events": [],
            }
        )
        human_tasks = dict(reduced.get("human_tasks") or {})
        if reference_at is not None:
            human_tasks = cls._mark_timeout_tasks(
                human_tasks=human_tasks,
                reference_at=reference_at,
            )
        return human_tasks

    @staticmethod
    def find_latest_waiting_human_task(
            human_tasks: Dict[str, HumanTaskState],
    ) -> Optional[HumanTaskState]:
        for task_id in reversed(list(human_tasks.keys())):
            task = dict(human_tasks[task_id])
            if task.get("status") == HumanTaskStatus.WAITING.value:
                return task
        return None

    @staticmethod
    def find_latest_human_task(
            human_tasks: Dict[str, HumanTaskState],
    ) -> Optional[HumanTaskState]:
        for task_id in reversed(list(human_tasks.keys())):
            return dict(human_tasks[task_id])
        return None

    @classmethod
    def build_runtime_metadata(cls, state: PlannerReActLangGraphState) -> Dict[str, Any]:
        """将 graph state 收敛为 WorkflowRun.runtime_metadata。"""
        events = list(state.get("emitted_events") or [])
        last_event = events[-1] if events else None

        return {
            "graph_state_contract": {
                "schema_version": str(
                    state.get("schema_version") or GRAPH_STATE_CONTRACT_SCHEMA_VERSION
                ),
                "planes": {
                    "graph_state_fields": list(cls.GRAPH_STATE_FIELDS),
                    "projection_only_fields": list(cls.PROJECTION_ONLY_FIELDS),
                    "audit_only_fields": list(cls.AUDIT_ONLY_FIELDS),
                    "artifact_only_fields": list(cls.ARTIFACT_ONLY_FIELDS),
                },
                "graph_state": {
                    "session_id": state.get("session_id"),
                    "run_id": state.get("run_id"),
                    "thread_id": state.get("thread_id"),
                    "checkpoint_ref_namespace": state.get("checkpoint_ref_namespace"),
                    "checkpoint_ref_id": state.get("checkpoint_ref_id"),
                    "input_parts": cls._to_json_safe(state.get("input_parts") or []),
                    "current_step_id": state.get("current_step_id"),
                    "execution_count": int(state.get("execution_count") or 0),
                    "max_execution_steps": int(state.get("max_execution_steps") or 20),
                    "step_states": cls._to_json_safe(state.get("step_states") or []),
                    "human_tasks": cls._to_json_safe(state.get("human_tasks") or {}),
                    "tool_invocations": cls._to_json_safe(state.get("tool_invocations") or {}),
                    "metadata": cls._to_json_safe(state.get("graph_metadata") or {}),
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
