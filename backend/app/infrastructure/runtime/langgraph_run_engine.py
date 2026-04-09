#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : langgraph_run_engine.py
"""
import asyncio
import base64
import json
import logging
import mimetypes
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, Optional, List

from langgraph.types import Command

from app.domain.external import LLM, FileStorage
from app.domain.models import (
    BaseEvent,
    Message,
    File,
    WaitEvent,
    WorkflowRunSummary,
    SessionContextSnapshot,
    WorkflowRunStatus,
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime import RunEngine
from app.domain.services.runtime.langgraph_state import (
    GraphStateContractMapper,
    PlannerReActLangGraphState,
    get_graph_projection,
    replace_graph_projection,
)
from app.domain.services.runtime.normalizers import build_delivery_text, normalize_ref_list, normalize_text_list
from app.domain.services.runtime.stage_llm import ensure_required_stage_llms
from app.domain.services.tools import BaseTool
from app.infrastructure.runtime.checkpoint_store_adapter import CheckpointStoreAdapter
from app.infrastructure.runtime.langgraph_graphs import (
    bind_live_event_sink,
    build_planner_react_langgraph_graph,
    unbind_live_event_sink,
)
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.runtime_logging import (
    bind_trace_id,
    build_trace_id,
    describe_stage_llms,
    elapsed_ms,
    log_runtime,
    now_perf,
    reset_trace_id,
)
from app.infrastructure.runtime.langgraph_long_term_memory_repository import LangGraphLongTermMemoryRepository
from app.infrastructure.utils import BaseUtils

logger = logging.getLogger(__name__)

SESSION_CONTEXT_SUMMARY_LIMIT = 20


@dataclass(frozen=True)
class ResumeCheckpointInspection:
    """恢复前对 checkpoint 的只读检查结果。"""

    run_id: Optional[str]
    invoke_config: Dict[str, Dict[str, str]]
    checkpoint_state: Optional[PlannerReActLangGraphState]
    pending_interrupt: Dict[str, Any]

    @property
    def has_checkpoint(self) -> bool:
        return self.checkpoint_state is not None

    @property
    def is_resumable(self) -> bool:
        return bool(self.run_id) and self.has_checkpoint and len(self.pending_interrupt) > 0


class _EventDeduplicator:
    """基于 event_id + payload signature 的事件去重器。"""

    def __init__(self) -> None:
        self._event_ids: set[str] = set()
        self._event_signatures: set[str] = set()

    @staticmethod
    def _build_signature(event: BaseEvent) -> str:
        payload = event.model_dump(mode="json")
        payload.pop("id", None)
        return json.dumps(
            {
                "event_type": event.type,
                "payload": payload,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def should_emit(self, event: BaseEvent) -> bool:
        event_id = str(event.id or "")
        event_signature = self._build_signature(event)
        if event_id and event_id in self._event_ids:
            return False
        if event_signature in self._event_signatures:
            return False
        if event_id:
            self._event_ids.add(event_id)
        self._event_signatures.add(event_signature)
        return True


class LangGraphRunEngine(RunEngine):
    """基于 LangGraph 的运行时引擎。"""

    def __init__(
            self,
            session_id: str,
            stage_llms: Dict[str, LLM],
            file_storage: Optional[FileStorage] = None,
            user_id: Optional[str] = None,
            uow_factory: Optional[Callable[[], IUnitOfWork]] = None,
            runtime_tools: Optional[List[BaseTool]] = None,
            max_tool_iterations: Optional[int] = None,
            checkpointer: Any | None = None,
    ) -> None:
        self._session_id = session_id
        self._file_storage = file_storage
        self._user_id = user_id
        self._uow_factory = uow_factory
        self._checkpointer = checkpointer
        normalized_stage_llms = ensure_required_stage_llms(stage_llms)
        self._long_term_memory_repository = (
            LangGraphLongTermMemoryRepository(uow_factory=uow_factory)
            if uow_factory is not None
            else None
        )
        self._graph = self._build_graph(
            stage_llms=normalized_stage_llms,
            runtime_tools=runtime_tools,
            max_tool_iterations=max_tool_iterations,
            checkpointer=self._checkpointer,
            long_term_memory_repository=self._long_term_memory_repository,
        )
        self._checkpoint_adapter = (
            CheckpointStoreAdapter(session_id=session_id, uow_factory=uow_factory)
            if uow_factory is not None
            else None
        )

    @staticmethod
    def _build_graph(
            *,
            stage_llms: Dict[str, LLM],
            runtime_tools: Optional[List[BaseTool]],
            max_tool_iterations: Optional[int],
            checkpointer: Any,
            long_term_memory_repository: Any,
    ) -> Any:
        graph_kwargs: Dict[str, Any] = {
            "stage_llms": stage_llms,
            "checkpointer": checkpointer,
            "long_term_memory_repository": long_term_memory_repository,
        }
        if runtime_tools is not None or max_tool_iterations is not None:
            graph_kwargs["runtime_tools"] = runtime_tools
            graph_kwargs["max_tool_iterations"] = max_tool_iterations or 5

        log_runtime(
            logger,
            logging.INFO,
            "初始化运行流程图",
            runtime_tool_count=len(list(runtime_tools or [])),
            max_tool_iterations=max_tool_iterations or 5,
            stage_llm_count=len(stage_llms),
            stage_llm_models=describe_stage_llms(stage_llms),
            has_checkpointer=checkpointer is not None,
            has_repository=long_term_memory_repository is not None,
        )
        return build_planner_react_langgraph_graph(**graph_kwargs)

    async def _build_input_parts(
            self,
            message: Message,
            uow: Optional[IUnitOfWork] = None,
    ) -> List[Dict[str, Any]]:
        """根据 message.attachments 构建输入片段。"""
        parts: List[Dict[str, Any]] = []
        attachment_paths = message.attachments or []

        for filepath in attachment_paths:
            part_type = BaseUtils.resolve_part_type_by_filepath(filepath=filepath)
            file_record: Optional[File] = await uow.session.get_file_by_path(
                session_id=self._session_id,
                filepath=filepath,
            )
            if file_record is None:
                continue

            guessed_mime_type = str(mimetypes.guess_type(filepath)[0] or "").strip()
            mime_type = file_record.mime_type or guessed_mime_type or "application/octet-stream"

            file_stream, _ = await self._file_storage.download_file(
                file_id=file_record.id,
                user_id=self._user_id,
            )

            bytes_raw, _ = BaseUtils.read_limited_bytes(stream=file_stream)

            part: Dict[str, Any] = {
                "type": part_type,
                "base64_payload": base64.b64encode(bytes_raw).decode('utf-8'),
                "mime_type": mime_type,
                "file_url": self._file_storage.get_file_url(file=file_record),
                "sandbox_filepath": filepath,
            }
            parts.append(part)
        return parts

    async def _build_graph_input_state(
            self,
            message: Message,
            run_id: Optional[str],
            invoke_config: Dict[str, Dict[str, str]],
    ) -> PlannerReActLangGraphState:
        """构建 BE-LG-04 契约化 graph input state。"""

        configurable = invoke_config.get("configurable") or {}
        thread_id = str(configurable.get("thread_id") or self._session_id)
        input_parts: List[Dict[str, Any]] = []

        try:
            async with self._uow_factory() as uow:
                session = await uow.session.get_by_id(session_id=self._session_id)
                if session is None:
                    raise ValueError(f"会话[{self._session_id}]不存在，无法构建Graph初始状态")

                resolved_run_id = run_id or session.current_run_id
                run = (
                    await uow.workflow_run.get_by_id(resolved_run_id)
                    if resolved_run_id
                    else None
                )
                completed_run_summaries = await uow.workflow_run_summary.list_by_session_id(
                    session_id=session.id,
                    limit=SESSION_CONTEXT_SUMMARY_LIMIT,
                    statuses=[WorkflowRunStatus.COMPLETED],
                )
                recent_attempt_summaries = await uow.workflow_run_summary.list_by_session_id(
                    session_id=session.id,
                    limit=SESSION_CONTEXT_SUMMARY_LIMIT,
                    statuses=[WorkflowRunStatus.FAILED, WorkflowRunStatus.CANCELLED],
                )
                session_context_snapshot = await uow.session_context_snapshot.get_by_session_id(
                    session_id=session.id,
                )
                input_parts = await self._build_input_parts(message=message, uow=uow)

                return GraphStateContractMapper.build_initial_state(
                    session=session,
                    run=run,
                    completed_run_summaries=completed_run_summaries,
                    recent_attempt_summaries=recent_attempt_summaries,
                    session_context_snapshot=session_context_snapshot,
                    user_message=message.message,
                    input_parts=input_parts,
                    continue_cancelled_task=(
                            message.command is not None
                            and message.command.type == "continue_cancelled_task"
                    ),
                    thread_id=thread_id,
                )
        except Exception as e:
            # 状态构建失败时必须降级，不能阻断主链路执行。
            log_runtime(
                logger,
                logging.WARNING,
                "构建初始状态失败，回退最小输入",
                session_id=self._session_id,
                run_id=run_id,
                thread_id=thread_id,
                error=str(e),
            )
            return {
                "session_id": self._session_id,
                "user_id": self._user_id,
                "run_id": run_id,
                "thread_id": thread_id,
                "user_message": message.message,
                "input_parts": input_parts,
                "message_window": [],
                "conversation_summary": "",
                "working_memory": {},
                "retrieved_memories": [],
                "pending_memory_writes": [],
                "recent_run_briefs": [],
                "recent_attempt_briefs": [],
                "session_open_questions": [],
                "session_blockers": [],
                "selected_artifacts": [],
                "historical_artifact_refs": [],
                "plan": None,
                "current_step_id": None,
                "execution_count": 0,
                "max_execution_steps": 20,
                "last_executed_step": None,
                "step_states": [],
                "pending_interrupt": {},
                "graph_metadata": {},
                "artifact_refs": [],
                "final_message": "",
                "emitted_events": [],
            }

    @staticmethod
    def _build_step_ledger(state: PlannerReActLangGraphState) -> List[Dict[str, Any]]:
        return [
            dict(item)
            for item in list(state.get("step_states") or [])
            if isinstance(item, dict)
        ]

    @staticmethod
    def _build_context_brief(summary: WorkflowRunSummary) -> Dict[str, Any]:
        return {
            "run_id": summary.run_id,
            "title": summary.title,
            "goal": summary.goal,
            "status": summary.status.value,
            "final_answer_summary": str(summary.final_answer_summary or "").strip()[:200],
        }

    @classmethod
    def _resolve_run_status_for_projection(
            cls,
            *,
            run: Any,
            state: PlannerReActLangGraphState,
    ) -> WorkflowRunStatus:
        pending_interrupt = GraphStateContractMapper.get_pending_interrupt(state)
        if pending_interrupt:
            return WorkflowRunStatus.WAITING

        raw_status = str(get_graph_projection(state.get("graph_metadata")).get("run_status") or "").strip()
        if raw_status:
            try:
                return WorkflowRunStatus(raw_status)
            except ValueError:
                log_runtime(
                    logger,
                    logging.WARNING,
                    "运行状态无效，回退数据库状态",
                    run_id=getattr(run, "id", ""),
                    status=raw_status,
                )

        return run.status

    @classmethod
    def _build_run_summary_projection(
            cls,
            *,
            run: Any,
            state: PlannerReActLangGraphState,
    ) -> WorkflowRunSummary:
        # 提取计划对象和步骤记录
        plan = state.get("plan")
        step_ledger = cls._build_step_ledger(state)

        # 统计已完成的步骤数量
        completed_steps = sum(
            1 for item in step_ledger if str(item.get("status") or "") == "completed"
        )

        # 初始化收集列表
        blockers: List[str] = []
        facts_learned: List[str] = []
        open_questions: List[str] = []

        # 从步骤记录的结果中提取阻碍项、习得事实和待解决问题
        for item in step_ledger:
            outcome = item.get("outcome")
            if not isinstance(outcome, dict):
                continue
            blockers.extend([str(value) for value in list(outcome.get("blockers") or [])])
            facts_learned.extend([str(value) for value in list(outcome.get("facts_learned") or [])])
            open_questions.extend([str(value) for value in list(outcome.get("open_questions") or [])])

        # 补充工作记忆中的会话事实和待解决问题
        facts_learned.extend(
            [str(item) for item in list((state.get("working_memory") or {}).get("facts_in_session") or [])])
        open_questions.extend(
            [str(item) for item in list((state.get("working_memory") or {}).get("open_questions") or [])])

        # final_message 是轻量摘要；重交付正文若存在，则单独来自 final_delivery_payload。
        final_message = str(state.get("final_message") or "").strip()
        final_delivery_text = build_delivery_text((state.get("working_memory") or {}).get("final_delivery_payload"))

        # 运行摘要状态必须从当前 graph state 真实语义推导，不能依赖外部事件是否已先落库。
        run_status = cls._resolve_run_status_for_projection(run=run, state=state)

        return WorkflowRunSummary(
            run_id=run.id,
            session_id=run.session_id,
            user_id=run.user_id,
            thread_id=run.thread_id,
            # 目标：优先取自计划对象，其次取自工作记忆
            goal=str(getattr(plan, "goal", "") or (state.get("working_memory") or {}).get("goal") or ""),
            # 标题：优先取自计划对象，其次取自最终消息前 80 字符，最后兜底
            title=str(getattr(plan, "title", "") or final_message[:80] or "未命名运行"),
            final_answer_summary=final_message[:500],
            final_answer_text=final_delivery_text or final_message,
            status=run_status,
            completed_steps=completed_steps,
            total_steps=len(getattr(plan, "steps", []) or []),
            step_ledger=step_ledger,
            # 仅保留当前 run 明确选择/确认过的产物，避免把事件层噪音引用投影到历史上下文。
            artifacts=normalize_ref_list(
                [str(item) for item in list(state.get("selected_artifacts") or [])]
            ),
            open_questions=normalize_text_list(open_questions),
            blockers=normalize_text_list(blockers),
            facts_learned=normalize_text_list(facts_learned),
        )

    @classmethod
    def _build_session_context_snapshot_projection(
            cls,
            *,
            session_id: str,
            user_id: Optional[str],
            summaries: List[WorkflowRunSummary],
    ) -> SessionContextSnapshot:
        recent_summaries = list(summaries or [])[:SESSION_CONTEXT_SUMMARY_LIMIT]

        # 提取非空的摘要文本或标题，用于构建简短的上下文概览
        summary_text_parts = [
            str(item.final_answer_summary or item.title or "").strip()
            for item in recent_summaries
            if str(item.final_answer_summary or item.title or "").strip()
        ]

        return SessionContextSnapshot(
            session_id=session_id,
            user_id=user_id,
            # 设置最近一次运行的 ID，若无运行记录则为 None
            last_run_id=recent_summaries[0].run_id if recent_summaries else None,
            # 将前三个摘要片段拼接成简要文本
            summary_text=" | ".join(summary_text_parts[:3]),
            # 构建详细的近期运行简报列表
            recent_run_briefs=[
                cls._build_context_brief(item)
                for item in recent_summaries
            ],
            # 收集并去重所有运行中的待解决问题
            open_questions=normalize_text_list(
                [question for item in recent_summaries for question in list(item.open_questions or [])]
            ),
            # 收集并去重所有运行中产生的工件引用
            artifact_refs=normalize_ref_list(
                [artifact for item in recent_summaries for artifact in list(item.artifacts or [])]
            ),
        )

    async def _sync_context_projections(
            self,
            *,
            run: Any,
            state: PlannerReActLangGraphState,
            uow: IUnitOfWork,
    ) -> None:
        run_status = self._resolve_run_status_for_projection(run=run, state=state)
        if run_status in {
            WorkflowRunStatus.PENDING,
            WorkflowRunStatus.RUNNING,
            WorkflowRunStatus.WAITING,
        }:
            return

        # 构建当前运行的摘要投影并持久化
        summary = self._build_run_summary_projection(run=run, state=state)
        await uow.workflow_run_summary.upsert(summary)

        # 获取会话最近的运行摘要列表，用于构建上下文快照
        recent_summaries = await uow.workflow_run_summary.list_by_session_id(
            session_id=run.session_id,
            limit=SESSION_CONTEXT_SUMMARY_LIMIT,
            statuses=[WorkflowRunStatus.COMPLETED],
        )

        # 基于最新摘要列表构建会话上下文快照并持久化
        snapshot = self._build_session_context_snapshot_projection(
            session_id=run.session_id,
            user_id=run.user_id,
            summaries=recent_summaries,
        )
        await uow.session_context_snapshot.upsert(snapshot)

    async def _sync_graph_state_contract(
            self,
            run_id: Optional[str],
            state: Optional[PlannerReActLangGraphState],
    ) -> None:
        """回写 BE-LG-04 graph state contract 到 runtime_metadata。"""
        if self._uow_factory is None or state is None:
            return

        resolved_run_id = run_id or state.get("run_id")
        if not resolved_run_id:
            return

        runtime_metadata = GraphStateContractMapper.build_runtime_metadata(state)
        try:
            async with self._uow_factory() as uow:
                run = await uow.workflow_run.get_by_id(resolved_run_id)
                if run is None:
                    log_runtime(
                        logger,
                        logging.WARNING,
                        "运行不存在，跳过状态回写",
                        session_id=self._session_id,
                        run_id=resolved_run_id,
                    )
                    return
                await uow.workflow_run.update_runtime_metadata(
                    run_id=resolved_run_id,
                    runtime_metadata=runtime_metadata,
                    current_step_id=state.get("current_step_id"),
                )
                await self._sync_context_projections(
                    run=run,
                    state=state,
                    uow=uow,
                )
                log_runtime(
                    logger,
                    logging.INFO,
                    "状态合同回写完成",
                    state=state,
                    run_id=resolved_run_id,
                )
        except Exception as e:
            log_runtime(
                logger,
                logging.WARNING,
                "状态合同回写失败",
                session_id=self._session_id,
                run_id=resolved_run_id,
                error=str(e),
            )

    async def _resolve_invoke_context(self) -> tuple[Dict[str, Dict[str, str]], Optional[str]]:
        invoke_config = {"configurable": {"thread_id": self._session_id}}
        run_id = None
        if self._checkpoint_adapter is not None:
            try:
                invoke_config, run_id = await self._checkpoint_adapter.resolve_invoke_config()
            except Exception as e:
                # 配置解析失败不阻断主流程，降级到 session 级 thread。
                log_runtime(
                    logger,
                    logging.WARNING,
                    "解析检查点配置失败，回退默认线程配置",
                    session_id=self._session_id,
                    error=str(e),
                )
        return invoke_config, run_id

    async def _load_checkpoint_state(
            self,
            invoke_config: Dict[str, Dict[str, str]],
    ) -> Optional[PlannerReActLangGraphState]:
        aget_state = getattr(self._graph, "aget_state", None)
        if callable(aget_state):
            snapshot = await aget_state(invoke_config)
        else:
            get_state = getattr(self._graph, "get_state", None)
            snapshot = get_state(invoke_config) if callable(get_state) else None

        if snapshot is None:
            return None
        values = getattr(snapshot, "values", None)
        if not isinstance(values, dict):
            return None
        return dict(values)

    async def inspect_resume_checkpoint(self) -> ResumeCheckpointInspection:
        """读取当前恢复点的 checkpoint 状态，不推进图执行。"""
        invoke_config, run_id = await self._resolve_invoke_context()
        checkpoint_state = await self._load_checkpoint_state(invoke_config=invoke_config)
        pending_interrupt = GraphStateContractMapper.get_pending_interrupt(checkpoint_state)
        return ResumeCheckpointInspection(
            run_id=run_id,
            invoke_config=invoke_config,
            checkpoint_state=checkpoint_state,
            pending_interrupt=pending_interrupt,
        )

    @staticmethod
    def _build_wait_events(raw_interrupts: Any) -> List[WaitEvent]:
        wait_events: List[WaitEvent] = []
        for item in list(raw_interrupts or []):
            payload = getattr(item, "value", None)
            if not isinstance(payload, dict):
                payload = {"value": payload}
            wait_events.append(
                WaitEvent.from_interrupt(
                    interrupt_id=getattr(item, "id", None),
                    payload=payload,
                )
            )
        return wait_events

    @staticmethod
    def _inject_pending_interrupts(
            state: PlannerReActLangGraphState,
            wait_events: List[WaitEvent],
    ) -> PlannerReActLangGraphState:
        graph_metadata = state.get("graph_metadata")
        if len(wait_events) > 0:
            graph_metadata = replace_graph_projection(
                graph_metadata,
                {"run_status": WorkflowRunStatus.WAITING.value},
            )
        return {
            **state,
            "pending_interrupt": (
                dict(wait_events[-1].payload or {})
                if len(wait_events) > 0
                else {}
            ),
            "graph_metadata": graph_metadata,
        }

    @staticmethod
    def _same_event(left: BaseEvent, right: BaseEvent) -> bool:
        return (
                str(left.id or "") == str(right.id or "")
                and _EventDeduplicator._build_signature(left) == _EventDeduplicator._build_signature(right)
        )

    @classmethod
    def _resolve_output_events(
            cls,
            state: Optional[PlannerReActLangGraphState],
            baseline_state: Optional[PlannerReActLangGraphState],
    ) -> List[BaseEvent]:
        # 获取当前状态和基准状态中已发射的事件列表
        current_events = list((state or {}).get("emitted_events") or [])
        baseline_events = list((baseline_state or {}).get("emitted_events") or [])

        # 如果基准事件为空，说明没有需要过滤的历史事件，直接返回当前所有事件
        if not baseline_events:
            return current_events

        # 计算前后缀匹配长度，找出从开头开始连续相同的事件数量
        max_prefix = min(len(current_events), len(baseline_events))
        prefix_len = 0
        while prefix_len < max_prefix and cls._same_event(current_events[prefix_len], baseline_events[prefix_len]):
            prefix_len += 1

        # 如果基准事件完全是当前事件的前缀，则返回当前事件中超出基准的部分（即新增事件）
        if prefix_len == len(baseline_events):
            return current_events[prefix_len:]

        # 构建基准事件 ID 集合，用于快速查找
        baseline_event_ids = {str(event.id or "") for event in baseline_events if str(event.id or "").strip()}

        # 如果基准事件没有有效 ID，则无法通过 ID 去重，保守返回所有当前事件
        if not baseline_event_ids:
            return current_events

        # 返回当前事件中那些不在基准事件 ID 集合中的事件（处理乱序或非前缀重复场景）
        return [event for event in current_events if str(event.id or "") not in baseline_event_ids]

    async def _run_graph(
            self,
            *,
            graph_input: Any,
            invoke_config: Dict[str, Dict[str, str]],
            run_id: Optional[str],
            fallback_state: Optional[PlannerReActLangGraphState],
    ) -> AsyncGenerator[BaseEvent, None]:
        started_at = now_perf()
        state: Optional[PlannerReActLangGraphState] = None
        deduplicator = _EventDeduplicator()
        live_event_queue: "asyncio.Queue[BaseEvent]" = asyncio.Queue()
        input_state = graph_input if isinstance(graph_input, dict) else fallback_state
        log_runtime(
            logger,
            logging.INFO,
            "开始运行流程",
            state=input_state,
            run_id=run_id,
            graph_input_type=type(graph_input).__name__,
        )

        async def _enqueue_live_event(base_event: BaseEvent) -> None:
            await live_event_queue.put(base_event)

        sink_token = bind_live_event_sink(_enqueue_live_event)
        graph_task: asyncio.Task | None = None
        try:
            graph_task = asyncio.create_task(
                self._graph.ainvoke(
                    graph_input,
                    config=invoke_config,
                )
            )
            async for live_event in self._forward_live_events(
                    graph_task=graph_task,
                    live_event_queue=live_event_queue,
                    deduplicator=deduplicator,
            ):
                yield live_event

            raw_result = await graph_task
            if isinstance(raw_result, dict) and "__interrupt__" in raw_result:
                wait_events = self._build_wait_events(raw_result.get("__interrupt__"))
                checkpoint_state = await self._load_checkpoint_state(invoke_config=invoke_config)
                state = checkpoint_state or fallback_state
                if state is not None:
                    state = self._inject_pending_interrupts(state=state, wait_events=wait_events)
                log_runtime(
                    logger,
                    logging.INFO,
                    "流程进入等待",
                    state=state or fallback_state,
                    wait_event_count=len(wait_events),
                )
            else:
                state = raw_result
                state = GraphStateContractMapper.apply_emitted_events(state=state)
                log_runtime(
                    logger,
                    logging.INFO,
                    "流程执行完成",
                    state=state,
                    emitted_event_count=len(list((state or {}).get("emitted_events") or [])),
                )
        finally:
            unbind_live_event_sink(sink_token)
            if graph_task is not None and not graph_task.done():
                graph_task.cancel()
                with suppress(Exception):
                    await graph_task

            if self._checkpoint_adapter is not None:
                # 无论图执行成功或失败，都尝试同步最新 checkpoint 引用，保证恢复点尽量前移。
                await self._checkpoint_adapter.sync_latest_checkpoint_ref(
                    run_id=run_id,
                    checkpointer=getattr(self._graph, "checkpointer", None),
                    invoke_config=invoke_config,
                )
            # graph 执行失败或中断时，回写当前已知状态。
            await self._sync_graph_state_contract(
                run_id=run_id,
                state=state or fallback_state,
            )
            log_runtime(
                logger,
                logging.INFO,
                "流程收尾同步完成",
                state=state or fallback_state,
                run_id=run_id,
                elapsed_ms=elapsed_ms(started_at),
            )

        for event in self._resolve_output_events(state=state, baseline_state=fallback_state):
            if deduplicator.should_emit(event):
                yield event.model_copy(deep=True)
        for event in wait_events:
            if deduplicator.should_emit(event):
                yield event

    async def invoke(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        started_at = now_perf()
        invoke_config, run_id = await self._resolve_invoke_context()
        trace_token = bind_trace_id(build_trace_id(self._session_id, run_id))
        try:
            log_runtime(
                logger,
                logging.INFO,
                "收到新消息并开始处理",
                session_id=self._session_id,
                run_id=run_id,
                message_length=len(str(message.message or "")),
                attachment_count=len(list(message.attachments or [])),
                command_type=message.command.type if message.command is not None else "",
            )

            graph_input_state = await self._build_graph_input_state(
                message=message,
                run_id=run_id,
                invoke_config=invoke_config,
            )
            async for event in self._run_graph(
                    graph_input=graph_input_state,
                    invoke_config=invoke_config,
                    run_id=run_id,
                    fallback_state=graph_input_state,
            ):
                yield event
            log_runtime(
                logger,
                logging.INFO,
                "本轮消息处理完成",
                session_id=self._session_id,
                run_id=run_id,
                elapsed_ms=elapsed_ms(started_at),
            )
        finally:
            reset_trace_id(trace_token)

    async def resume(self, value: Any) -> AsyncGenerator[BaseEvent, None]:
        started_at = now_perf()
        invoke_config, run_id = await self._resolve_invoke_context()
        checkpoint_state = await self._load_checkpoint_state(invoke_config=invoke_config)
        trace_token = bind_trace_id(build_trace_id(self._session_id, run_id))
        try:
            log_runtime(
                logger,
                logging.INFO,
                "收到恢复输入并继续处理",
                state=checkpoint_state,
                run_id=run_id,
                resume_value_type=type(value).__name__,
            )
            async for event in self._run_graph(
                    graph_input=Command(resume=value),
                    invoke_config=invoke_config,
                    run_id=run_id,
                    fallback_state=checkpoint_state,
            ):
                yield event
            log_runtime(
                logger,
                logging.INFO,
                "本轮恢复处理完成",
                state=checkpoint_state,
                run_id=run_id,
                elapsed_ms=elapsed_ms(started_at),
            )
        finally:
            reset_trace_id(trace_token)

    async def _forward_live_events(
            self,
            *,
            graph_task: "asyncio.Task[Any]",
            live_event_queue: "asyncio.Queue[BaseEvent]",
            deduplicator: _EventDeduplicator,
    ) -> AsyncGenerator[BaseEvent, None]:
        """边执行边转发节点实时事件。"""
        while True:
            if graph_task.done() and live_event_queue.empty():
                break
            try:
                event = await asyncio.wait_for(live_event_queue.get(), timeout=0.1)
            except TimeoutError:
                continue

            if deduplicator.should_emit(event):
                # 对外输出副本，避免下游写入 event.id 反向污染 graph state 中的同一对象。
                yield event.model_copy(deep=True)
