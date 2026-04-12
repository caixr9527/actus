#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 上下文工程服务。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.domain.models import (
    BrowserToolContent,
    ExecutionStatus,
    FetchPageToolContent,
    SearchToolContent,
    ShellToolContent,
    Step,
    StepDeliveryContextState,
    StepDeliveryRole,
    StepOutputMode,
    StepTaskModeHint,
    ToolEvent,
    normalize_wait_payload,
)
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.runtime.normalizers import (
    normalize_controlled_value,
    normalize_delivery_payload,
    normalize_file_path_list,
    normalize_message_window_entry,
    normalize_step_result_text,
    normalize_text_list,
    truncate_text,
)

from .contracts import PendingConfirmationPacket, PromptContextPacket, PromptStage
from .policies import ContextPolicy, get_context_policy

# P2 上下文裁剪常量集中在这里，避免 domain 反向依赖 infrastructure settings。
_ARTIFACT_LIMIT = 6
_BRIEF_LIMIT = 3
_COMPLETED_STEP_LIMIT = 5
_MESSAGE_LIMIT = 6
_OPEN_QUESTION_LIMIT = 8
_SUMMARY_MAX_CHARS = 400
_MESSAGE_MAX_CHARS = 200
_MEMORY_SUMMARY_MAX_CHARS = 160
_MEMORY_CONTENT_MAX_CHARS = 240
_STEP_SUMMARY_MAX_CHARS = 160
_STEP_DELIVERY_MAX_CHARS = 200
_FINAL_MESSAGE_MAX_CHARS = 240
_DIGEST_TEXT_MAX_CHARS = 240


class RuntimeContextService:
    """按 stage + task_mode 构造 PromptContextPacket。"""

    def build_packet(
            self,
            *,
            stage: PromptStage,
            state: PlannerReActLangGraphState,
            step: Optional[Step] = None,
            task_mode: str = "",
    ) -> PromptContextPacket:
        """统一生成结构化 Prompt 上下文数据包。"""
        resolved_task_mode = self._resolve_task_mode(stage=stage, state=state, step=step, task_mode=task_mode)
        policy = get_context_policy(stage, resolved_task_mode)
        # 先根据 stage + task_mode 决定字段是否可见，不允许先构造一整包再事后裁剪。
        packet: PromptContextPacket = {
            "stage": stage,
            "task_mode": resolved_task_mode,
            "user_goal": self._build_user_goal(state=state),
        }
        if policy.include_current_step:
            packet["current_step"] = self._build_current_step(stage=stage, state=state, step=step)
        if policy.include_open_questions:
            packet["open_questions"] = self._build_open_questions(state=state, step=step)
        if policy.include_pending_confirmation:
            packet["pending_confirmation"] = self._build_pending_confirmation(state=state)
        if policy.include_environment_digest:
            packet["environment_digest"] = self._build_environment_digest(
                state=state,
                stage=stage,
                task_mode=resolved_task_mode,
            )
        if policy.include_observation_digest:
            packet["observation_digest"] = self._build_observation_digest(
                state=state,
                stage=stage,
                task_mode=resolved_task_mode,
                step=step,
            )
        if policy.include_recent_action_digest:
            packet["recent_action_digest"] = self._build_recent_action_digest(
                state=state,
                task_mode=resolved_task_mode,
            )
        if policy.include_working_memory_digest:
            packet["working_memory_digest"] = self._build_working_memory_digest(
                state=state,
                stage=stage,
            )
        if policy.include_retrieved_memory_digest:
            packet["retrieved_memory_digest"] = self._build_retrieved_memory_digest(
                state=state,
                stage=stage,
                task_mode=resolved_task_mode,
                policy=policy,
            )
        if policy.include_stable_background:
            packet["stable_background"] = self._build_stable_background(
                state=state,
                policy=policy,
            )
        if policy.include_audit_refs:
            packet["audit_refs"] = self._build_audit_refs(state=state)
        packet["prompt_visible_fields"] = [
            field_name
            for field_name, value in packet.items()
            if field_name != "prompt_visible_fields" and self._has_visible_value(value)
        ]
        return packet

    def extract_state_updates(self, packet: PromptContextPacket) -> Dict[str, Any]:
        """从 packet 中提取需要回写到 graph state 的 digest 字段。"""
        task_mode = str(packet.get("task_mode") or "")
        return {
            "task_mode": task_mode,
            "environment_digest": self._wrap_state_digest(task_mode=task_mode, payload=packet.get("environment_digest")),
            "observation_digest": self._wrap_state_digest(task_mode=task_mode, payload=packet.get("observation_digest")),
            "recent_action_digest": self._wrap_state_digest(task_mode=task_mode, payload=packet.get("recent_action_digest")),
        }

    def normalize_runtime_recent_action(self, raw: Any) -> Dict[str, Any]:
        """统一接收执行阶段产出的 recent_action 结构，避免节点自己做字段筛选。"""
        if not isinstance(raw, dict):
            return {}

        normalized: Dict[str, Any] = {}
        last_failed_action = raw.get("last_failed_action")
        if isinstance(last_failed_action, dict) and last_failed_action:
            normalized["last_failed_action"] = dict(last_failed_action)
        last_blocked_tool_call = raw.get("last_blocked_tool_call")
        if isinstance(last_blocked_tool_call, dict) and last_blocked_tool_call:
            normalized["last_blocked_tool_call"] = dict(last_blocked_tool_call)
        last_no_progress_reason = str(raw.get("last_no_progress_reason") or "").strip()
        if last_no_progress_reason:
            normalized["last_no_progress_reason"] = last_no_progress_reason
        return normalized

    def merge_runtime_recent_action(
            self,
            *,
            state_updates: Dict[str, Any],
            task_mode: str,
            runtime_recent_action: Any,
    ) -> Dict[str, Any]:
        """把当前步骤 recent_action 合并回同模式 digest，节点层不再关心 envelope 细节。"""
        normalized_recent_action = self.normalize_runtime_recent_action(runtime_recent_action)
        if not normalized_recent_action:
            return state_updates

        wrapped_digest = state_updates.get("recent_action_digest")
        existing_payload: Dict[str, Any] = {}
        if isinstance(wrapped_digest, dict) and str(wrapped_digest.get("task_mode") or "").strip() == str(task_mode or "").strip():
            payload = wrapped_digest.get("payload")
            if isinstance(payload, dict):
                existing_payload = dict(payload)
        return {
            **state_updates,
            "recent_action_digest": self._wrap_state_digest(
                task_mode=task_mode,
                payload={
                    **existing_payload,
                    **normalized_recent_action,
                },
            ),
        }

    @staticmethod
    def _has_visible_value(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            return bool(value.strip())
        if isinstance(value, dict):
            return len(value) > 0
        if isinstance(value, list):
            return len(value) > 0
        return True

    @staticmethod
    def _truncate_text(value: Any, *, max_chars: int) -> str:
        return truncate_text(value, max_chars=max_chars)

    def _resolve_task_mode(
            self,
            *,
            stage: PromptStage,
            state: PlannerReActLangGraphState,
            step: Optional[Step],
            task_mode: str,
    ) -> str:
        normalized_task_mode = normalize_controlled_value(task_mode, StepTaskModeHint)
        if normalized_task_mode:
            return normalized_task_mode
        # planner 必须从通用模式起步，避免沿用上一轮执行态。
        if stage == "planner":
            return StepTaskModeHint.GENERAL.value
        if stage == "summary":
            state_task_mode = normalize_controlled_value(state.get("task_mode"), StepTaskModeHint)
            return state_task_mode or StepTaskModeHint.GENERAL.value
        if step is not None:
            structured_hint = normalize_controlled_value(getattr(step, "task_mode_hint", None), StepTaskModeHint)
            if structured_hint:
                return structured_hint
        state_task_mode = normalize_controlled_value(state.get("task_mode"), StepTaskModeHint)
        return state_task_mode or StepTaskModeHint.GENERAL.value

    def _build_user_goal(self, *, state: PlannerReActLangGraphState) -> str:
        working_memory = dict(state.get("working_memory") or {})
        goal = str(working_memory.get("goal") or "").strip()
        if goal:
            return goal
        plan = state.get("plan")
        if plan is not None and str(plan.goal or "").strip():
            return str(plan.goal or "").strip()
        return str(state.get("user_message") or "").strip()

    def _build_open_questions(
            self,
            *,
            state: PlannerReActLangGraphState,
            step: Optional[Step],
    ) -> List[str]:
        working_memory = dict(state.get("working_memory") or {})
        open_questions = normalize_text_list(state.get("session_open_questions"))
        open_questions.extend(normalize_text_list(working_memory.get("open_questions")))
        last_step = state.get("last_executed_step")
        if last_step is not None and last_step.outcome is not None:
            open_questions.extend(normalize_text_list(last_step.outcome.open_questions))
        if step is not None and step.outcome is not None:
            open_questions.extend(normalize_text_list(step.outcome.open_questions))
        deduped_questions: List[str] = []
        for item in open_questions:
            if item in deduped_questions:
                continue
            deduped_questions.append(item)
        return [
            self._truncate_text(item, max_chars=120)
            for item in deduped_questions[:_OPEN_QUESTION_LIMIT]
            if str(item).strip()
        ]

    def _build_pending_confirmation(self, *, state: PlannerReActLangGraphState) -> PendingConfirmationPacket:
        payload = normalize_wait_payload(state.get("pending_interrupt"))
        if not payload:
            return {}

        packet: PendingConfirmationPacket = {
            "kind": str(payload.get("kind") or "").strip(),
            "title": str(payload.get("title") or "").strip(),
            "prompt": str(payload.get("prompt") or "").strip(),
            "details": self._truncate_text(payload.get("details"), max_chars=240),
            "suggest_user_takeover": str(payload.get("suggest_user_takeover") or "none").strip(),
            "attachments": normalize_file_path_list(payload.get("attachments"))[:_ARTIFACT_LIMIT],
        }
        if packet["kind"] == "confirm":
            packet["confirm_label"] = str(payload.get("confirm_label") or "").strip()
            packet["cancel_label"] = str(payload.get("cancel_label") or "").strip()
        elif packet["kind"] == "input_text":
            packet["submit_label"] = str(payload.get("submit_label") or "").strip()
            packet["response_key"] = str(payload.get("response_key") or "").strip()
        elif packet["kind"] == "select":
            packet["options"] = [
                {
                    "label": str(item.get("label") or "").strip(),
                    "description": self._truncate_text(item.get("description"), max_chars=120),
                    "resume_value": item.get("resume_value"),
                }
                for item in list(payload.get("options") or [])[:_OPEN_QUESTION_LIMIT]
                if isinstance(item, dict) and str(item.get("label") or "").strip()
            ]
        return packet

    def _build_environment_digest(
            self,
            *,
            state: PlannerReActLangGraphState,
            stage: PromptStage,
            task_mode: str,
    ) -> Dict[str, Any]:
        # environment 只保留“当前模式还能依赖什么”，不负责回放完整轨迹。
        if task_mode in {StepTaskModeHint.RESEARCH.value, StepTaskModeHint.WEB_READING.value}:
            digest = {
                "recent_search_queries": self._collect_recent_search_queries(
                    state=state,
                    task_mode=task_mode,
                ),
                "candidate_links": self._collect_candidate_links(state=state),
                "read_page_summaries": self._collect_fetch_page_summaries(state=state),
                "current_page": self._collect_browser_page_summary(state=state),
            }
            return self._clean_dict(digest)
        if task_mode == StepTaskModeHint.BROWSER_INTERACTION.value:
            current_page = self._collect_browser_page_summary(state=state)
            digest = {
                "current_page": current_page,
                "actionable_elements": list(current_page.get("actionable_elements") or [])[:_OPEN_QUESTION_LIMIT],
            }
            return self._clean_dict(digest)
        if task_mode in {StepTaskModeHint.CODING.value, StepTaskModeHint.FILE_PROCESSING.value}:
            digest = {
                "cwd": self._extract_latest_shell_cwd(state=state),
                "recent_changed_files": self._collect_recent_changed_files(state=state),
                "file_tree_summary": self._collect_file_tree_summary(state=state),
                "available_artifacts": self._collect_available_artifacts(state=state),
                "shell_session_status": self._extract_shell_session_status(state=state),
            }
            return self._clean_dict(digest)
        if task_mode == StepTaskModeHint.HUMAN_WAIT.value:
            pending_confirmation = self._build_pending_confirmation(state=state)
            digest = {
                "wait_kind": str(pending_confirmation.get("kind") or "").strip(),
                "wait_prompt": str(pending_confirmation.get("prompt") or "").strip(),
                "wait_options": list(pending_confirmation.get("options") or []),
            }
            return self._clean_dict(digest)

        digest = {
            "available_artifacts": self._collect_available_artifacts(state=state),
            "stage": stage,
        }
        return self._clean_dict(digest)

    def _build_observation_digest(
            self,
            *,
            state: PlannerReActLangGraphState,
            stage: PromptStage,
            task_mode: str,
            step: Optional[Step],
    ) -> Dict[str, Any]:
        # observation 先收敛最近步骤结果，再补当前模式最关键的现场观察。
        last_step = step or state.get("last_executed_step")
        digest: Dict[str, Any] = {}
        if last_step is not None:
            digest["last_step_result"] = normalize_step_result_text(
                last_step.outcome.summary if last_step.outcome is not None else ""
            )
            if last_step.outcome is not None:
                digest["last_step_delivery_preview"] = self._truncate_text(
                    last_step.outcome.delivery_text,
                    max_chars=_STEP_DELIVERY_MAX_CHARS,
                )
                digest["last_step_blockers"] = [
                    self._truncate_text(item, max_chars=120)
                    for item in normalize_text_list(last_step.outcome.blockers)[:_OPEN_QUESTION_LIMIT]
                ]
        if task_mode in {StepTaskModeHint.RESEARCH.value, StepTaskModeHint.WEB_READING.value}:
            latest_fetch_summary = self._collect_fetch_page_summaries(state=state)
            if latest_fetch_summary:
                digest["latest_fetch_page"] = latest_fetch_summary[-1]
            current_page = self._collect_browser_page_summary(state=state)
            if current_page:
                digest["latest_browser_observation"] = current_page
        elif task_mode == StepTaskModeHint.BROWSER_INTERACTION.value:
            current_page = self._collect_browser_page_summary(state=state)
            if current_page:
                digest["latest_browser_observation"] = current_page
        elif task_mode in {StepTaskModeHint.CODING.value, StepTaskModeHint.FILE_PROCESSING.value}:
            latest_shell_output = self._extract_latest_shell_result(state=state)
            if latest_shell_output:
                digest["latest_shell_result"] = latest_shell_output
        digest["stage"] = stage
        digest["task_mode"] = task_mode
        return self._clean_dict(digest)

    def _build_recent_action_digest(
            self,
            *,
            state: PlannerReActLangGraphState,
            task_mode: str,
    ) -> Dict[str, Any]:
        # recent_action 只读取同模式沉淀结果，不能在这里跨模式扫描历史事件。
        same_mode_digest = self._read_state_digest(
            state=state,
            field_name="recent_action_digest",
            expected_task_mode=task_mode,
        )
        digest: Dict[str, Any] = {}
        if isinstance(same_mode_digest.get("last_failed_action"), dict):
            digest["last_failed_action"] = dict(same_mode_digest.get("last_failed_action") or {})
        if isinstance(same_mode_digest.get("last_blocked_tool_call"), dict):
            digest["last_blocked_tool_call"] = dict(same_mode_digest.get("last_blocked_tool_call") or {})
        if task_mode in {StepTaskModeHint.RESEARCH.value, StepTaskModeHint.WEB_READING.value}:
            digest["recent_search_queries"] = self._collect_recent_search_queries(
                state=state,
                task_mode=task_mode,
            )
        if task_mode == StepTaskModeHint.HUMAN_WAIT.value:
            digest["last_user_wait_reason"] = str(self._build_pending_confirmation(state=state).get("prompt") or "").strip()
        if same_mode_digest.get("last_no_progress_reason"):
            digest["last_no_progress_reason"] = str(same_mode_digest.get("last_no_progress_reason") or "").strip()
        return self._clean_dict(digest)

    def _build_working_memory_digest(
            self,
            *,
            state: PlannerReActLangGraphState,
            stage: PromptStage,
    ) -> Dict[str, Any]:
        working_memory = dict(state.get("working_memory") or {})
        digest: Dict[str, Any] = {
            "goal": self._build_user_goal(state=state),
            "decisions": [
                self._truncate_text(item, max_chars=120)
                for item in normalize_text_list(working_memory.get("decisions"))[:_OPEN_QUESTION_LIMIT]
            ],
            "facts_in_session": [
                self._truncate_text(item, max_chars=120)
                for item in normalize_text_list(working_memory.get("facts_in_session"))[:_OPEN_QUESTION_LIMIT]
            ],
            "user_preferences": dict(working_memory.get("user_preferences") or {}),
        }
        if stage == "summary":
            digest["final_delivery_payload"] = self._build_summary_delivery_payload(state=state)
            digest["execution_count"] = int(state.get("execution_count") or 0)
        return {
            key: value
            for key, value in digest.items()
            if self._has_visible_value(value)
        }

    def _build_retrieved_memory_digest(
            self,
            *,
            state: PlannerReActLangGraphState,
            stage: PromptStage,
            task_mode: str,
            policy: ContextPolicy,
    ) -> List[Dict[str, Any]]:
        # 召回记忆必须按 stage + task_mode 的策略收敛类型和数量，不能退回统一全量裁剪。
        normalized_stage = str(stage).strip().lower()
        normalized_task_mode = str(task_mode or "").strip().lower()
        if not normalized_stage or not normalized_task_mode:
            return []

        allowed_types = {
            str(item).strip().lower()
            for item in set(policy.retrieved_memory_allowed_types or set())
            if str(item).strip()
        }
        max_items = int(policy.retrieved_memory_max_items or 0)
        if max_items <= 0:
            return []

        sanitized_memories: List[Dict[str, Any]] = []
        for item in list(state.get("retrieved_memories") or []):
            if not isinstance(item, dict):
                continue
            memory_type = str(item.get("memory_type") or "").strip().lower()
            if allowed_types and memory_type not in allowed_types:
                continue
            content = item.get("content") if isinstance(item.get("content"), dict) else {}
            content_fragments: List[str] = []
            for key in list(content.keys()):
                normalized_key = str(key or "").strip().lower()
                if normalized_key in {"embedding", "vector", "source", "score", "similarity", "distance"}:
                    continue
                value = content.get(key)
                if isinstance(value, (str, int, float, bool)):
                    text = str(value).strip()
                    if text:
                        content_fragments.append(
                            self._truncate_text(
                                text if normalized_key in {"text", "content"} else f"{key}: {text}",
                                max_chars=80,
                            )
                        )
                elif isinstance(value, list):
                    items = [str(entry).strip() for entry in value if str(entry).strip()]
                    if items:
                        content_fragments.append(
                            self._truncate_text(
                                f"{key}: {', '.join(items[:4])}",
                                max_chars=80,
                            )
                        )
                if len(content_fragments) >= 3:
                    break
            sanitized_memories.append(
                {
                    "id": str(item.get("id") or "").strip(),
                    "memory_type": memory_type,
                    "summary": self._truncate_text(
                        item.get("summary"),
                        max_chars=_MEMORY_SUMMARY_MAX_CHARS,
                    ),
                    "content_preview": self._truncate_text(
                        " | ".join(content_fragments),
                        max_chars=_MEMORY_CONTENT_MAX_CHARS,
                    ),
                    "tags": [
                        self._truncate_text(tag, max_chars=40)
                        for tag in list(item.get("tags") or [])[:6]
                        if str(tag or "").strip()
                    ],
                }
            )
            if len(sanitized_memories) >= max_items:
                break
        return sanitized_memories

    def _build_stable_background(
            self,
            *,
            state: PlannerReActLangGraphState,
            policy: ContextPolicy,
    ) -> Dict[str, Any]:
        # 稳定背景承载跨阶段共用信息，不混入模式瞬时状态。
        background: Dict[str, Any] = {}
        if policy.stable_background.include_conversation_summary:
            background["conversation_summary"] = self._truncate_text(
                state.get("conversation_summary"),
                max_chars=_SUMMARY_MAX_CHARS,
            )
        if policy.stable_background.include_recent_messages:
            background["recent_messages"] = self._build_recent_messages(state=state)
        if policy.stable_background.include_completed_steps:
            background["completed_steps"] = self._build_completed_steps(state=state)
        if policy.stable_background.include_recent_run_briefs:
            background["recent_run_briefs"] = self._build_recent_briefs(
                state=state,
                field_name="recent_run_briefs",
            )
        if policy.stable_background.include_recent_attempt_briefs:
            background["recent_attempt_briefs"] = self._build_recent_briefs(
                state=state,
                field_name="recent_attempt_briefs",
            )
        if policy.stable_background.include_plan_snapshot:
            background["plan_snapshot"] = self._build_plan_snapshot(state=state)
        if policy.stable_background.include_summary_focus:
            background["summary_focus"] = {
                "final_message": self._truncate_text(
                    state.get("final_message"),
                    max_chars=_FINAL_MESSAGE_MAX_CHARS,
                ),
                "selected_artifacts": normalize_file_path_list(state.get("selected_artifacts"))[:_ARTIFACT_LIMIT],
            }
        return {
            key: value
            for key, value in background.items()
            if self._has_visible_value(value)
        }

    def _read_state_digest(
            self,
            *,
            state: PlannerReActLangGraphState,
            field_name: str,
            expected_task_mode: str,
    ) -> Dict[str, Any]:
        raw_digest = state.get(field_name)
        if not isinstance(raw_digest, dict):
            return {}
        task_mode = str(raw_digest.get("task_mode") or "").strip()
        if task_mode != expected_task_mode:
            return {}
        return self._read_state_digest_payload(state=state, field_name=field_name)

    def _read_state_digest_payload(
            self,
            *,
            state: PlannerReActLangGraphState,
            field_name: str,
    ) -> Dict[str, Any]:
        """统一读取 digest.payload，确保采集逻辑与 state 包装结构一致。"""
        raw_digest = state.get(field_name)
        if not isinstance(raw_digest, dict):
            return {}
        payload = raw_digest.get("payload")
        if not isinstance(payload, dict):
            return {}
        return payload

    @staticmethod
    def _wrap_state_digest(*, task_mode: str, payload: Any) -> Dict[str, Any]:
        normalized_payload = payload if isinstance(payload, dict) else {}
        return {
            "task_mode": str(task_mode or "").strip(),
            "payload": normalized_payload,
        }

    def _clean_dict(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return {
            key: item
            for key, item in value.items()
            if self._has_visible_value(item)
        }

    def _build_current_step(
            self,
            *,
            stage: PromptStage,
            state: PlannerReActLangGraphState,
            step: Optional[Step],
    ) -> Dict[str, Any]:
        target_step = step
        # replan 默认围绕最近执行完成的步骤组织当前步上下文。
        if target_step is None and stage == "replan":
            target_step = state.get("last_executed_step")
        if target_step is None:
            return {}

        current_step = {
            "step_id": str(target_step.id or "").strip(),
            "title": self._truncate_text(target_step.title, max_chars=80),
            "description": self._truncate_text(target_step.description, max_chars=160),
            "task_mode_hint": normalize_controlled_value(getattr(target_step, "task_mode_hint", None), StepTaskModeHint),
            "output_mode": normalize_controlled_value(getattr(target_step, "output_mode", None), StepOutputMode),
            "delivery_role": normalize_controlled_value(
                getattr(target_step, "delivery_role", None),
                StepDeliveryRole,
            ),
            "delivery_context_state": normalize_controlled_value(
                getattr(target_step, "delivery_context_state", None),
                StepDeliveryContextState,
            ),
            "status": str(getattr(target_step, "status", "") or "").strip(),
        }
        if target_step.outcome is not None:
            current_step["result"] = {
                "summary": normalize_step_result_text(target_step.outcome.summary),
                "delivery_preview": self._truncate_text(
                    target_step.outcome.delivery_text,
                    max_chars=_STEP_DELIVERY_MAX_CHARS,
                ),
                "blockers": [
                    self._truncate_text(item, max_chars=120)
                    for item in normalize_text_list(target_step.outcome.blockers)[:_OPEN_QUESTION_LIMIT]
                ],
                "open_questions": [
                    self._truncate_text(item, max_chars=120)
                    for item in normalize_text_list(target_step.outcome.open_questions)[:_OPEN_QUESTION_LIMIT]
                ],
            }
        return {
            key: value
            for key, value in current_step.items()
            if self._has_visible_value(value)
        }

    def _build_audit_refs(self, *, state: PlannerReActLangGraphState) -> Dict[str, Any]:
        return {
            "run_id": str(state.get("run_id") or "").strip(),
            "thread_id": str(state.get("thread_id") or "").strip(),
            "current_step_id": str(state.get("current_step_id") or "").strip(),
        }

    def _build_recent_messages(self, *, state: PlannerReActLangGraphState) -> List[Dict[str, Any]]:
        recent_messages: List[Dict[str, Any]] = []
        for item in list(state.get("message_window") or [])[-_MESSAGE_LIMIT:]:
            normalized_item = normalize_message_window_entry(
                item,
                default_role="assistant",
            )
            if normalized_item is None:
                continue
            recent_messages.append(
                {
                    **normalized_item,
                    "message": self._truncate_text(
                        normalized_item.get("message"),
                        max_chars=_MESSAGE_MAX_CHARS,
                    ),
                }
            )
        return recent_messages

    def _build_completed_steps(self, *, state: PlannerReActLangGraphState) -> List[Dict[str, Any]]:
        step_states = [dict(item) for item in list(state.get("step_states") or []) if isinstance(item, dict)]
        completed_steps = [
            item for item in step_states
            if str(item.get("status") or "") == ExecutionStatus.COMPLETED.value
        ][-_COMPLETED_STEP_LIMIT:]
        sanitized_completed_steps: List[Dict[str, Any]] = []
        for item in completed_steps:
            outcome = item.get("outcome") if isinstance(item.get("outcome"), dict) else {}
            sanitized_completed_steps.append(
                {
                    "step_id": str(item.get("step_id") or "").strip(),
                    "title": self._truncate_text(item.get("title"), max_chars=80),
                    "description": self._truncate_text(item.get("description"), max_chars=120),
                    "status": str(item.get("status") or "").strip(),
                    "summary": self._truncate_text(outcome.get("summary"), max_chars=_STEP_SUMMARY_MAX_CHARS),
                    "delivery_preview": self._truncate_text(
                        outcome.get("delivery_text"),
                        max_chars=_STEP_DELIVERY_MAX_CHARS,
                    ),
                }
            )
        return sanitized_completed_steps

    def _build_recent_briefs(
            self,
            *,
            state: PlannerReActLangGraphState,
            field_name: str,
    ) -> List[Dict[str, Any]]:
        briefs: List[Dict[str, Any]] = []
        for item in list(state.get(field_name) or [])[:_BRIEF_LIMIT]:
            if not isinstance(item, dict) or not str(item.get("run_id") or "").strip():
                continue
            briefs.append(
                {
                    "run_id": str(item.get("run_id") or "").strip(),
                    "title": str(item.get("title") or "").strip(),
                    "goal": str(item.get("goal") or "").strip(),
                    "status": str(item.get("status") or "").strip(),
                    "final_answer_summary": self._truncate_text(
                        item.get("final_answer_summary"),
                        max_chars=120,
                    ),
                    "final_answer_text_excerpt": self._truncate_text(
                        item.get("final_answer_text_excerpt"),
                        max_chars=160,
                    ),
                }
            )
        return briefs

    def _build_plan_snapshot(self, *, state: PlannerReActLangGraphState) -> Dict[str, Any]:
        plan = state.get("plan")
        if plan is None:
            return {}
        completed_step_summaries: List[str] = []
        failed_step_summaries: List[str] = []
        pending_step_titles: List[str] = []
        for step in list(plan.steps or []):
            if step.status == ExecutionStatus.COMPLETED:
                completed_step_summaries.append(
                    normalize_step_result_text(step.outcome.summary if step.outcome is not None else step.title)
                )
                continue
            if step.status == ExecutionStatus.FAILED:
                failed_step_summaries.append(
                    normalize_step_result_text(step.outcome.summary if step.outcome is not None else step.title)
                )
                continue
            pending_title = normalize_step_result_text(step.title, fallback=step.description)
            if pending_title:
                pending_step_titles.append(pending_title)
        return {
            "title": str(plan.title or "").strip(),
            "goal": str(plan.goal or "").strip(),
            "status": str(plan.status.value if hasattr(plan.status, "value") else plan.status or "").strip(),
            "step_count": len(list(plan.steps or [])),
            "completed_step_summaries": completed_step_summaries[:5],
            "failed_step_summaries": failed_step_summaries[:3],
            "pending_step_titles": pending_step_titles[:5],
        }

    def _build_summary_delivery_payload(self, *, state: PlannerReActLangGraphState) -> Dict[str, Any]:
        working_memory = dict(state.get("working_memory") or {})
        delivery_payload = normalize_delivery_payload(working_memory.get("final_delivery_payload"))
        return {
            "text": self._truncate_text(delivery_payload.get("text"), max_chars=1600),
            "sections": [
                {
                    "title": self._truncate_text(item.get("title"), max_chars=80),
                    "content": self._truncate_text(item.get("content"), max_chars=600),
                }
                for item in list(delivery_payload.get("sections") or [])[:6]
            ],
            "source_refs": normalize_file_path_list(delivery_payload.get("source_refs"))[:_ARTIFACT_LIMIT],
        }

    def _collect_recent_search_queries(
            self,
            *,
            state: PlannerReActLangGraphState,
            task_mode: str,
    ) -> List[str]:
        # 查询历史优先读同模式 digest，再把本轮新增的 search_web 调用补进来。
        stored_digest = self._read_state_digest(
            state=state,
            field_name="recent_action_digest",
            expected_task_mode=task_mode,
        )
        queries = normalize_text_list(stored_digest.get("recent_search_queries"))
        for event in list(state.get("emitted_events") or []):
            if not isinstance(event, ToolEvent):
                continue
            if str(event.function_name or "").strip().lower() != "search_web":
                continue
            query = str((event.function_args or {}).get("query") or (event.function_args or {}).get("q") or "").strip()
            if query:
                queries.append(query)
        deduped_queries: List[str] = []
        for item in queries:
            if item in deduped_queries:
                continue
            deduped_queries.append(item)
        return [self._truncate_text(item, max_chars=120) for item in deduped_queries[-_OPEN_QUESTION_LIMIT:]]

    def _collect_candidate_links(self, *, state: PlannerReActLangGraphState) -> List[Dict[str, str]]:
        candidate_links: List[Dict[str, str]] = []
        for event in list(state.get("emitted_events") or []):
            if not isinstance(event, ToolEvent) or not isinstance(event.tool_content, SearchToolContent):
                continue
            for item in list(event.tool_content.results or [])[:_OPEN_QUESTION_LIMIT]:
                candidate_links.append(
                    {
                        "title": self._truncate_text(getattr(item, "title", ""), max_chars=80),
                        "url": self._truncate_text(getattr(item, "url", ""), max_chars=200),
                        "snippet": self._truncate_text(getattr(item, "description", ""), max_chars=120),
                    }
                )
            if candidate_links:
                break
        # 当前轮没有新的搜索结果时，才回退到已持久化摘要。
        stored_digest = self._read_state_digest_payload(state=state, field_name="environment_digest")
        stored_links = list(stored_digest.get("candidate_links") or [])
        if candidate_links:
            return candidate_links[:_OPEN_QUESTION_LIMIT]
        return [item for item in stored_links if isinstance(item, dict)][: _OPEN_QUESTION_LIMIT]

    def _collect_fetch_page_summaries(self, *, state: PlannerReActLangGraphState) -> List[Dict[str, Any]]:
        page_summaries: List[Dict[str, Any]] = []
        for event in list(state.get("emitted_events") or []):
            if not isinstance(event, ToolEvent) or not isinstance(event.tool_content, FetchPageToolContent):
                continue
            page_summaries.append(
                {
                    "title": self._truncate_text(event.tool_content.title, max_chars=80),
                    "url": self._truncate_text(event.tool_content.url, max_chars=200),
                    "excerpt": self._truncate_text(
                        event.tool_content.excerpt or event.tool_content.content,
                        max_chars=160,
                    ),
                }
            )
        if page_summaries:
            return page_summaries[-_OPEN_QUESTION_LIMIT:]
        stored_digest = self._read_state_digest_payload(state=state, field_name="environment_digest")
        stored_pages = list(stored_digest.get("read_page_summaries") or [])
        return [item for item in stored_pages if isinstance(item, dict)][: _OPEN_QUESTION_LIMIT]

    def _collect_browser_page_summary(self, *, state: PlannerReActLangGraphState) -> Dict[str, Any]:
        browser_events = [
            event
            for event in list(state.get("emitted_events") or [])
            if isinstance(event, ToolEvent) and isinstance(event.tool_content, BrowserToolContent)
        ]
        if browser_events:
            latest = browser_events[-1].tool_content
            actionable_elements = [
                {
                    "text": self._truncate_text(getattr(item, "text", ""), max_chars=60),
                    "selector": self._truncate_text(getattr(item, "selector", ""), max_chars=120),
                    "type": self._truncate_text(getattr(item, "type", ""), max_chars=40),
                }
                for item in list(latest.actionable_elements or [])[:_OPEN_QUESTION_LIMIT]
            ]
            return {
                "url": self._truncate_text(latest.url, max_chars=200),
                "title": self._truncate_text(latest.title, max_chars=80),
                "page_type": self._truncate_text(latest.page_type, max_chars=40),
                "main_content_summary": self._truncate_text(
                    getattr(latest.main_content, "summary", "") or getattr(latest.main_content, "content", ""),
                    max_chars=200,
                ),
                "actionable_elements": actionable_elements,
                "degrade_reason": self._truncate_text(latest.degrade_reason, max_chars=120),
            }
        # 没有新的浏览器观察时，再回退到 environment_digest 的页面摘要。
        stored_digest = self._read_state_digest_payload(state=state, field_name="environment_digest")
        stored_page = dict(stored_digest.get("current_page") or {})
        return {
            key: value
            for key, value in stored_page.items()
            if key in {"url", "title", "page_type", "main_content_summary", "actionable_elements", "degrade_reason"}
        }

    def _extract_latest_shell_cwd(self, *, state: PlannerReActLangGraphState) -> str:
        for event in reversed(list(state.get("emitted_events") or [])):
            if not isinstance(event, ToolEvent):
                continue
            if not str(event.function_name or "").strip().lower().startswith("shell_"):
                continue
            cwd = str((event.function_args or {}).get("cwd") or (event.function_args or {}).get("workdir") or "").strip()
            if cwd:
                return cwd
        stored_digest = self._read_state_digest_payload(state=state, field_name="environment_digest")
        return str(stored_digest.get("cwd") or "").strip()

    def _collect_recent_changed_files(self, *, state: PlannerReActLangGraphState) -> List[str]:
        last_step = state.get("last_executed_step")
        recent_changed_files = normalize_file_path_list(
            last_step.outcome.produced_artifacts if last_step is not None and last_step.outcome is not None else []
        )
        if recent_changed_files:
            return recent_changed_files[:_ARTIFACT_LIMIT]
        stored_digest = self._read_state_digest_payload(state=state, field_name="environment_digest")
        stored_files = normalize_file_path_list(stored_digest.get("recent_changed_files"))
        return stored_files[:_ARTIFACT_LIMIT]

    def _collect_file_tree_summary(self, *, state: PlannerReActLangGraphState) -> List[str]:
        file_summaries: List[str] = []
        for event in list(state.get("emitted_events") or []):
            if not isinstance(event, ToolEvent):
                continue
            function_name = str(event.function_name or "").strip().lower()
            if function_name not in {"list_files", "find_files"}:
                continue
            result_message = ""
            if event.function_result is not None:
                result_message = self._truncate_text(event.function_result.message, max_chars=_DIGEST_TEXT_MAX_CHARS)
            if result_message:
                file_summaries.append(result_message)
        if file_summaries:
            return file_summaries[-_OPEN_QUESTION_LIMIT:]
        stored_digest = self._read_state_digest_payload(state=state, field_name="environment_digest")
        stored_summaries = normalize_text_list(stored_digest.get("file_tree_summary"))
        return stored_summaries[:_OPEN_QUESTION_LIMIT]

    def _collect_available_artifacts(self, *, state: PlannerReActLangGraphState) -> List[str]:
        artifact_groups = [
            normalize_file_path_list(state.get("selected_artifacts")),
            normalize_file_path_list(
                self._read_state_digest_payload(state=state, field_name="environment_digest").get("available_artifacts")
            ),
        ]
        last_step = state.get("last_executed_step")
        if last_step is not None and last_step.outcome is not None:
            artifact_groups.append(normalize_file_path_list(last_step.outcome.produced_artifacts))
        merged: List[str] = []
        for group in artifact_groups:
            for item in group:
                if item in merged:
                    continue
                merged.append(item)
        return merged[:_ARTIFACT_LIMIT]

    def _extract_shell_session_status(self, *, state: PlannerReActLangGraphState) -> str:
        stored_digest = self._read_state_digest_payload(state=state, field_name="environment_digest")
        stored_status = str(stored_digest.get("shell_session_status") or "").strip()
        latest_shell_result = self._extract_latest_shell_result(state=state)
        if latest_shell_result:
            return "active"
        return stored_status

    def _extract_latest_shell_result(self, *, state: PlannerReActLangGraphState) -> Dict[str, Any]:
        for event in reversed(list(state.get("emitted_events") or [])):
            if not isinstance(event, ToolEvent) or not isinstance(event.tool_content, ShellToolContent):
                continue
            console = event.tool_content.console
            console_text = self._truncate_text(console, max_chars=_DIGEST_TEXT_MAX_CHARS)
            return {
                "function_name": str(event.function_name or "").strip(),
                "message": self._truncate_text(
                    event.function_result.message if event.function_result is not None else "",
                    max_chars=120,
                ),
                "console": console_text,
            }
        stored_digest = self._read_state_digest_payload(state=state, field_name="observation_digest")
        stored_result = dict(stored_digest.get("latest_shell_result") or {})
        return stored_result
