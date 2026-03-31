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
    StepEvent,
    ToolEvent,
    WaitEvent,
    WorkflowRun,
)

logger = logging.getLogger(__name__)

# BE-LG-04 契约版本。
# v4 删除 human_tasks 真相源，等待态统一切到 LangGraph 原生 interrupt。
GRAPH_STATE_CONTRACT_SCHEMA_VERSION = "be-lg-04.v4"


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

    schema_version: str
    session_id: str
    user_id: Optional[str]
    run_id: Optional[str]
    thread_id: str
    checkpoint_ref_namespace: str
    checkpoint_ref_id: Optional[str]
    user_message: str
    input_parts: List[Dict[str, Any]]
    message_window: List[Dict[str, Any]]
    conversation_summary: str
    working_memory: Dict[str, Any]
    retrieved_memories: List[Dict[str, Any]]
    pending_memory_writes: List[Dict[str, Any]]
    planner_local_memory: Dict[str, Any]
    step_local_memory: Dict[str, Any]
    summary_local_memory: Dict[str, Any]
    memory_context_version: Optional[str]
    plan: Optional[Plan]
    current_step_id: Optional[str]
    execution_count: int
    max_execution_steps: int
    last_executed_step: Optional[Step]
    step_states: List[StepState]
    pending_interrupt: Dict[str, Any]
    tool_invocations: Dict[str, ToolInvocationState]
    graph_metadata: Dict[str, Any]
    artifact_refs: List[str]
    audit_events: List[BaseEvent]
    final_message: str
    emitted_events: List[BaseEvent]
    error: Optional[str]


class GraphStateContractMapper:
    """BE-LG-04：Graph State 与领域对象映射器。"""

    GRAPH_STATE_FIELDS: tuple[str, ...] = (
        "schema_version",
        "session_id",
        "user_id",
        "run_id",
        "thread_id",
        "checkpoint_ref_namespace",
        "checkpoint_ref_id",
        "user_message",
        "input_parts",
        "message_window",
        "conversation_summary",
        "working_memory",
        "retrieved_memories",
        "pending_memory_writes",
        "planner_local_memory",
        "step_local_memory",
        "summary_local_memory",
        "memory_context_version",
        "plan",
        "current_step_id",
        "execution_count",
        "max_execution_steps",
        "last_executed_step",
        "step_states",
        "pending_interrupt",
        "tool_invocations",
        "graph_metadata",
        "artifact_refs",
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

    @classmethod
    def _normalize_dict_memory(cls, raw: Any) -> Dict[str, Any]:
        if not isinstance(raw, dict):
            return {}
        return cls._to_json_safe(raw)

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
        return cls._normalize_list_memory(raw)

    @classmethod
    def _normalize_pending_interrupt(cls, raw: Any) -> Dict[str, Any]:
        return normalize_wait_payload(raw)

    @classmethod
    def get_pending_interrupt(
            cls,
            state: Optional[PlannerReActLangGraphState],
    ) -> Dict[str, Any]:
        """统一读取并标准化当前 checkpoint 中的等待态。"""
        if not isinstance(state, dict):
            return {}
        pending_interrupt = cls._normalize_pending_interrupt(state.get("pending_interrupt"))
        if pending_interrupt:
            return pending_interrupt

        graph_metadata = state.get("graph_metadata")
        if not isinstance(graph_metadata, dict):
            return {}
        pending_interrupts = graph_metadata.get("pending_interrupts")
        if not isinstance(pending_interrupts, list) or len(pending_interrupts) == 0:
            return {}
        first_interrupt = pending_interrupts[0]
        if not isinstance(first_interrupt, dict):
            return {}
        return cls._normalize_pending_interrupt(first_interrupt.get("payload"))

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
        return list(dict.fromkeys(refs))

    @classmethod
    def _normalize_last_executed_step(cls, raw: Any) -> Optional[Step]:
        if not isinstance(raw, dict) or not raw:
            return None
        try:
            return Step.model_validate(raw)
        except Exception:
            return None

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
        pending_interrupt = cls._normalize_pending_interrupt(graph_state_from_metadata.get("pending_interrupt"))
        if pending_interrupt:
            return pending_interrupt

        metadata = graph_state_from_metadata.get("metadata")
        if not isinstance(metadata, dict):
            return {}
        pending_interrupts = metadata.get("pending_interrupts")
        if not isinstance(pending_interrupts, list) or len(pending_interrupts) == 0:
            return {}

        first_interrupt = pending_interrupts[0]
        if not isinstance(first_interrupt, dict):
            return {}
        return cls._normalize_pending_interrupt(first_interrupt.get("payload"))

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

        return {
            "schema_version": GRAPH_STATE_CONTRACT_SCHEMA_VERSION,
            "session_id": session.id,
            "user_id": (
                run.user_id
                if run is not None and run.user_id is not None
                else session.user_id
            ),
            "run_id": run.id if run is not None else session.current_run_id,
            "thread_id": thread_id,
            "checkpoint_ref_namespace": checkpoint_namespace,
            "checkpoint_ref_id": checkpoint_id,
            "user_message": user_message,
            "input_parts": cls._normalize_input_parts(input_parts),
            "message_window": cls._append_current_user_message(
                message_window=cls._normalize_message_window(graph_state_from_metadata.get("message_window")),
                user_message=user_message,
                input_parts=cls._normalize_input_parts(input_parts),
            ),
            "conversation_summary": cls._normalize_text(graph_state_from_metadata.get("conversation_summary")),
            "working_memory": cls._normalize_dict_memory(graph_state_from_metadata.get("working_memory")),
            "retrieved_memories": cls._normalize_list_memory(graph_state_from_metadata.get("retrieved_memories")),
            "pending_memory_writes": cls._normalize_list_memory(graph_state_from_metadata.get("pending_memory_writes")),
            "planner_local_memory": cls._normalize_dict_memory(graph_state_from_metadata.get("planner_local_memory")),
            "step_local_memory": cls._normalize_dict_memory(graph_state_from_metadata.get("step_local_memory")),
            "summary_local_memory": cls._normalize_dict_memory(graph_state_from_metadata.get("summary_local_memory")),
            "memory_context_version": (
                str(graph_state_from_metadata.get("memory_context_version"))
                if graph_state_from_metadata.get("memory_context_version") is not None
                else None
            ),
            "plan": plan,
            "current_step_id": current_step_id,
            "execution_count": int(graph_state_from_metadata.get("execution_count") or 0),
            "max_execution_steps": int(graph_state_from_metadata.get("max_execution_steps") or 20),
            "last_executed_step": cls._normalize_last_executed_step(
                graph_state_from_metadata.get("last_executed_step")
            ),
            "step_states": step_states,
            "pending_interrupt": cls._extract_pending_interrupt_from_metadata(graph_state_from_metadata),
            "tool_invocations": cls._normalize_tool_invocations(graph_state_from_metadata.get("tool_invocations")),
            "graph_metadata": (
                cls._normalize_dict_memory(graph_state_from_metadata.get("metadata"))
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
        """根据 emitted events 收敛 step/tool/audit 状态。"""
        events = list(state.get("emitted_events") or [])
        next_state: PlannerReActLangGraphState = dict(state)
        step_states = list(next_state.get("step_states") or [])
        tool_invocations = dict(next_state.get("tool_invocations") or {})
        graph_metadata = dict(next_state.get("graph_metadata") or {})
        artifact_refs = list(next_state.get("artifact_refs") or [])
        pending_interrupt = cls._normalize_pending_interrupt(next_state.get("pending_interrupt"))

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
                pending_interrupt = cls._normalize_pending_interrupt(event.payload)
                graph_metadata["latest_wait_event_id"] = str(event.id)
                graph_metadata["pending_interrupts"] = [
                    {
                        "interrupt_id": event.interrupt_id,
                        "payload": cls._to_json_safe(pending_interrupt),
                    }
                ]
                if event.interrupt_id:
                    graph_metadata["latest_interrupt_id"] = str(event.interrupt_id)
                continue

            if isinstance(event, ErrorEvent):
                graph_metadata["run_status"] = "failed"
                graph_metadata["last_error"] = event.error
                graph_metadata["last_error_event_id"] = str(event.id)
                continue

            if isinstance(event, DoneEvent):
                graph_metadata["run_status"] = "completed"
                graph_metadata["last_done_event_id"] = str(event.id)
                pending_interrupt = {}

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
        next_state["pending_interrupt"] = pending_interrupt
        next_state["tool_invocations"] = tool_invocations
        next_state["graph_metadata"] = graph_metadata
        next_state["artifact_refs"] = list(dict.fromkeys(artifact_refs))
        next_state["audit_events"] = audit_events
        return next_state

    @classmethod
    def build_runtime_metadata(cls, state: PlannerReActLangGraphState) -> Dict[str, Any]:
        """将 graph state 收敛为 WorkflowRun.runtime_metadata。"""
        events = list(state.get("emitted_events") or [])
        last_event = events[-1] if events else None
        pending_interrupt = cls._normalize_pending_interrupt(state.get("pending_interrupt"))
        graph_metadata = cls._normalize_dict_memory(state.get("graph_metadata"))

        if pending_interrupt:
            graph_metadata["pending_interrupts"] = [
                {
                    "interrupt_id": graph_metadata.get("latest_interrupt_id"),
                    "payload": cls._to_json_safe(pending_interrupt),
                }
            ]
        else:
            graph_metadata.pop("pending_interrupts", None)

        return {
            "memory": {
                "recall_count": len(state.get("retrieved_memories") or []),
                "recall_ids": cls._extract_memory_ids(list(state.get("retrieved_memories") or [])),
                "write_count": len(state.get("pending_memory_writes") or []),
                "write_ids": cls._extract_memory_ids(list(state.get("pending_memory_writes") or [])),
                "compacted": bool(graph_metadata.get("memory_compacted", False)),
                "last_compaction_at": graph_metadata.get("memory_last_compaction_at"),
                "summary_version": state.get("memory_context_version"),
                "persisted_write_count": int(graph_metadata.get("memory_write_count") or 0),
                "persisted_write_ids": cls._to_json_safe(graph_metadata.get("memory_write_ids") or []),
            },
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
                    "user_id": state.get("user_id"),
                    "run_id": state.get("run_id"),
                    "thread_id": state.get("thread_id"),
                    "checkpoint_ref_namespace": state.get("checkpoint_ref_namespace"),
                    "checkpoint_ref_id": state.get("checkpoint_ref_id"),
                    "input_parts": cls._to_json_safe(state.get("input_parts") or []),
                    "message_window": cls._to_json_safe(state.get("message_window") or []),
                    "conversation_summary": str(state.get("conversation_summary") or ""),
                    "working_memory": cls._to_json_safe(state.get("working_memory") or {}),
                    "retrieved_memories": cls._to_json_safe(state.get("retrieved_memories") or []),
                    "pending_memory_writes": cls._to_json_safe(state.get("pending_memory_writes") or []),
                    "planner_local_memory": cls._to_json_safe(state.get("planner_local_memory") or {}),
                    "step_local_memory": cls._to_json_safe(state.get("step_local_memory") or {}),
                    "summary_local_memory": cls._to_json_safe(state.get("summary_local_memory") or {}),
                    "memory_context_version": state.get("memory_context_version"),
                    "plan": cls._to_json_safe(state.get("plan")),
                    "current_step_id": state.get("current_step_id"),
                    "execution_count": int(state.get("execution_count") or 0),
                    "max_execution_steps": int(state.get("max_execution_steps") or 20),
                    "last_executed_step": cls._to_json_safe(state.get("last_executed_step")),
                    "step_states": cls._to_json_safe(state.get("step_states") or []),
                    "pending_interrupt": cls._to_json_safe(pending_interrupt),
                    "tool_invocations": cls._to_json_safe(state.get("tool_invocations") or {}),
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
