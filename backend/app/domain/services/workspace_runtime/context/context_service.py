#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 上下文工程服务。"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.domain.models import (
    ExecutionStatus,
    Step,
    StepOutputMode,
    StepTaskModeHint,
    normalize_wait_payload,
)
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.runtime.normalizers import (
    normalize_controlled_value,
    normalize_file_path_list,
    normalize_message_window_entry,
    normalize_step_result_text,
    normalize_text_list,
    truncate_text,
)
from app.domain.services.workspace_runtime import WorkspaceEnvironmentSnapshot, WorkspaceRuntimeService
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
_FINAL_MESSAGE_MAX_CHARS = 240
_DIGEST_TEXT_MAX_CHARS = 240


class RuntimeContextService:
    """按 stage + task_mode 构造 PromptContextPacket。"""

    def __init__(
            self,
            workspace_runtime_service: Optional[WorkspaceRuntimeService] = None,
    ) -> None:
        self._workspace_runtime_service = workspace_runtime_service

    def build_packet(
            self,
            *,
            stage: PromptStage,
            state: PlannerReActLangGraphState,
            step: Optional[Step] = None,
            task_mode: str = "",
    ) -> PromptContextPacket:
        """统一生成结构化 Prompt 上下文数据包。"""
        return self._build_packet(
            stage=stage,
            state=state,
            step=step,
            task_mode=task_mode,
            workspace_snapshot=None,
        )

    async def build_packet_async(
            self,
            *,
            stage: PromptStage,
            state: PlannerReActLangGraphState,
            step: Optional[Step] = None,
            task_mode: str = "",
    ) -> PromptContextPacket:
        """基于 workspace 快照构造 Prompt 上下文数据包。"""
        workspace_snapshot = None
        if self._workspace_runtime_service is not None:
            workspace_snapshot = await self._workspace_runtime_service.build_environment_snapshot()
        return self._build_packet(
            stage=stage,
            state=state,
            step=step,
            task_mode=task_mode,
            workspace_snapshot=workspace_snapshot,
        )

    def _build_packet(
            self,
            *,
            stage: PromptStage,
            state: PlannerReActLangGraphState,
            step: Optional[Step],
            task_mode: str,
            workspace_snapshot: Optional[WorkspaceEnvironmentSnapshot],
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
                workspace_snapshot=workspace_snapshot,
            )
        if policy.include_observation_digest:
            packet["observation_digest"] = self._build_observation_digest(
                state=state,
                stage=stage,
                task_mode=resolved_task_mode,
                step=step,
                workspace_snapshot=workspace_snapshot,
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
            "environment_digest": self._wrap_state_digest(task_mode=task_mode,
                                                          payload=packet.get("environment_digest")),
            "observation_digest": self._wrap_state_digest(task_mode=task_mode,
                                                          payload=packet.get("observation_digest")),
            "recent_action_digest": self._wrap_state_digest(task_mode=task_mode,
                                                            payload=packet.get("recent_action_digest")),
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
        recent_search_queries = normalize_text_list(raw.get("recent_search_queries"))
        if recent_search_queries:
            normalized["recent_search_queries"] = recent_search_queries
        search_evidence_summaries = self._normalize_search_evidence_items(raw.get("search_evidence_summaries"))
        if search_evidence_summaries:
            normalized["search_evidence_summaries"] = search_evidence_summaries
        web_reading_evidence_summaries = self._normalize_web_reading_evidence_items(
            raw.get("web_reading_evidence_summaries")
        )
        if web_reading_evidence_summaries:
            normalized["web_reading_evidence_summaries"] = web_reading_evidence_summaries
        explicit_url_read_state = raw.get("explicit_url_read_state")
        if isinstance(explicit_url_read_state, dict) and explicit_url_read_state:
            normalized["explicit_url_read_state"] = dict(explicit_url_read_state)
        research_convergence = raw.get("research_convergence")
        if isinstance(research_convergence, dict) and research_convergence:
            normalized["research_convergence"] = dict(research_convergence)
        web_reading_convergence = raw.get("web_reading_convergence")
        if isinstance(web_reading_convergence, dict) and web_reading_convergence:
            normalized["web_reading_convergence"] = dict(web_reading_convergence)
        research_progress = raw.get("research_progress")
        if isinstance(research_progress, dict) and research_progress:
            # 只保留可读、低体量字段，供后续步骤知道已有 snippet 证据，避免再次强制抓页。
            normalized["research_progress"] = {
                "candidate_url_count": int(research_progress.get("candidate_url_count") or 0),
                "fetched_url_count": int(research_progress.get("fetched_url_count") or 0),
                "fetch_success_count": int(research_progress.get("fetch_success_count") or 0),
                "latest_query": str(research_progress.get("latest_query") or "").strip(),
                "search_evidence_summaries": self._normalize_search_evidence_items(
                    research_progress.get("search_evidence_summaries")
                ),
                "web_reading_evidence_summaries": self._normalize_web_reading_evidence_items(
                    research_progress.get("web_reading_evidence_summaries")
                ),
            }
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
        if isinstance(wrapped_digest, dict) and str(wrapped_digest.get("task_mode") or "").strip() == str(
                task_mode or "").strip():
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
            workspace_snapshot: Optional[WorkspaceEnvironmentSnapshot] = None,
    ) -> Dict[str, Any]:
        # environment 只保留“当前模式还能依赖什么”，不负责回放完整轨迹。
        if task_mode in {StepTaskModeHint.RESEARCH.value, StepTaskModeHint.WEB_READING.value}:
            digest = {
                "recent_search_queries": self._collect_recent_search_queries(
                    state=state,
                    task_mode=task_mode,
                ),
                "candidate_links": self._collect_candidate_links(
                    workspace_snapshot=workspace_snapshot,
                ),
                "search_evidence_summaries": self._collect_search_evidence_summaries(
                    workspace_snapshot=workspace_snapshot,
                ),
                "read_page_summaries": self._collect_fetch_page_summaries(
                    workspace_snapshot=workspace_snapshot,
                ),
                "current_page": self._collect_browser_page_summary(
                    workspace_snapshot=workspace_snapshot,
                ),
            }
            return self._clean_dict(digest)
        if task_mode == StepTaskModeHint.BROWSER_INTERACTION.value:
            current_page = self._collect_browser_page_summary(
                workspace_snapshot=workspace_snapshot,
            )
            digest = {
                "current_page": current_page,
                "actionable_elements": list(current_page.get("actionable_elements") or [])[:_OPEN_QUESTION_LIMIT],
            }
            return self._clean_dict(digest)
        if task_mode in {StepTaskModeHint.CODING.value, StepTaskModeHint.FILE_PROCESSING.value}:
            digest = {
                "cwd": self._extract_latest_shell_cwd(
                    workspace_snapshot=workspace_snapshot,
                ),
                "recent_changed_files": self._collect_recent_changed_files(
                    workspace_snapshot=workspace_snapshot,
                ),
                "file_tree_summary": self._collect_file_tree_summary(
                    workspace_snapshot=workspace_snapshot,
                ),
                "available_artifacts": self._collect_available_artifacts(
                    workspace_snapshot=workspace_snapshot,
                ),
                "shell_session_status": self._extract_shell_session_status(
                    workspace_snapshot=workspace_snapshot,
                ),
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
            "available_artifacts": self._collect_available_artifacts(
                workspace_snapshot=workspace_snapshot,
            ),
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
            workspace_snapshot: Optional[WorkspaceEnvironmentSnapshot] = None,
    ) -> Dict[str, Any]:
        # observation 先收敛最近步骤结果，再补当前模式最关键的现场观察。
        last_step = step or state.get("last_executed_step")
        digest: Dict[str, Any] = {}
        if last_step is not None:
            digest["last_step_result"] = normalize_step_result_text(
                last_step.outcome.summary if last_step.outcome is not None else ""
            )
            if last_step.outcome is not None:
                digest["last_step_blockers"] = [
                    self._truncate_text(item, max_chars=120)
                    for item in normalize_text_list(last_step.outcome.blockers)[:_OPEN_QUESTION_LIMIT]
                ]
                # facts_learned 作为步骤级证据文本，需要把完整内容传给后续 step，而不是再次截断成摘要。
                digest["last_step_facts"] = [
                    str(item)
                    for item in normalize_text_list(last_step.outcome.facts_learned)[:_OPEN_QUESTION_LIMIT]
                ]
        if task_mode in {StepTaskModeHint.RESEARCH.value, StepTaskModeHint.WEB_READING.value}:
            latest_fetch_summary = self._collect_fetch_page_summaries(
                workspace_snapshot=workspace_snapshot,
            )
            if latest_fetch_summary:
                digest["latest_fetch_page"] = latest_fetch_summary[-1]
            current_page = self._collect_browser_page_summary(
                workspace_snapshot=workspace_snapshot,
            )
            if current_page:
                digest["latest_browser_observation"] = current_page
        elif task_mode == StepTaskModeHint.BROWSER_INTERACTION.value:
            current_page = self._collect_browser_page_summary(
                workspace_snapshot=workspace_snapshot,
            )
            if current_page:
                digest["latest_browser_observation"] = current_page
        elif task_mode in {StepTaskModeHint.CODING.value, StepTaskModeHint.FILE_PROCESSING.value}:
            latest_shell_output = self._extract_latest_shell_result(
                workspace_snapshot=workspace_snapshot,
            )
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
        # recent_action 的失败/阻断只读同模式，research/web_reading 证据允许向后投影给下一阶段消费。
        same_mode_digest = self._read_state_digest(
            state=state,
            field_name="recent_action_digest",
            expected_task_mode=task_mode,
        )
        cross_mode_digest = self._read_state_digest_payload(
            state=state,
            field_name="recent_action_digest",
        )
        digest: Dict[str, Any] = {}
        if isinstance(same_mode_digest.get("last_failed_action"), dict):
            digest["last_failed_action"] = dict(same_mode_digest.get("last_failed_action") or {})
        if isinstance(same_mode_digest.get("last_blocked_tool_call"), dict):
            digest["last_blocked_tool_call"] = dict(same_mode_digest.get("last_blocked_tool_call") or {})
        if task_mode in {StepTaskModeHint.RESEARCH.value, StepTaskModeHint.WEB_READING.value}:
            digest["recent_search_queries"] = normalize_text_list(same_mode_digest.get("recent_search_queries"))
        digest.update(
            self._project_cross_step_evidence(
                task_mode=task_mode,
                source_digest=cross_mode_digest,
            )
        )
        if task_mode == StepTaskModeHint.HUMAN_WAIT.value:
            digest["last_user_wait_reason"] = str(
                self._build_pending_confirmation(state=state).get("prompt") or "").strip()
        if same_mode_digest.get("last_no_progress_reason"):
            digest["last_no_progress_reason"] = str(same_mode_digest.get("last_no_progress_reason") or "").strip()
        return self._clean_dict(digest)

    def _project_cross_step_evidence(
            self,
            *,
            task_mode: str,
            source_digest: Dict[str, Any],
    ) -> Dict[str, Any]:
        """把上一阶段可复用证据投影给下一阶段，不投影失败/阻断等反馈语义。"""
        if not isinstance(source_digest, dict) or not source_digest:
            return {}
        normalized_task_mode = str(task_mode or "").strip().lower()
        projected: Dict[str, Any] = {}
        if normalized_task_mode in {StepTaskModeHint.WEB_READING.value, StepTaskModeHint.GENERAL.value}:
            research_progress = dict(source_digest.get("research_progress") or {})
            search_evidence = self._normalize_search_evidence_items(source_digest.get("search_evidence_summaries"))
            search_evidence.extend(
                self._normalize_search_evidence_items(research_progress.get("search_evidence_summaries"))
            )
            search_evidence = self._dedupe_evidence_items(search_evidence, key_names=("url", "snippet"))[:5]
            if search_evidence:
                projected["search_evidence_summaries"] = search_evidence
                projected["research_progress"] = {
                    "candidate_url_count": int(research_progress.get("candidate_url_count") or len(search_evidence)),
                    "fetched_url_count": int(research_progress.get("fetched_url_count") or 0),
                    "fetch_success_count": int(research_progress.get("fetch_success_count") or 0),
                    "latest_query": str(research_progress.get("latest_query") or "").strip(),
                    "search_evidence_summaries": search_evidence,
                }
            web_evidence = self._normalize_web_reading_evidence_items(
                source_digest.get("web_reading_evidence_summaries")
            )
            web_evidence.extend(
                self._normalize_web_reading_evidence_items(research_progress.get("web_reading_evidence_summaries"))
            )
            web_evidence = self._dedupe_evidence_items(web_evidence, key_names=("url", "summary"))[:6]
            if web_evidence:
                projected["web_reading_evidence_summaries"] = web_evidence
                projected.setdefault("research_progress", {})["web_reading_evidence_summaries"] = web_evidence
            explicit_url_state = source_digest.get("explicit_url_read_state") or research_progress.get(
                "explicit_url_read_state"
            )
            if isinstance(explicit_url_state, dict) and explicit_url_state:
                projected["explicit_url_read_state"] = dict(explicit_url_state)
        return projected

    def _normalize_search_evidence_items(self, raw_items: Any) -> List[Dict[str, str]]:
        """归一化低体量 search snippet 证据，供 Prompt 和执行初始态复用。"""
        items: List[Dict[str, str]] = []
        for item in list(raw_items or []):
            if not isinstance(item, dict):
                continue
            title = self._truncate_text(item.get("title"), max_chars=100)
            url = self._truncate_text(item.get("url"), max_chars=240)
            snippet = self._truncate_text(item.get("snippet"), max_chars=320)
            if not url and not snippet:
                continue
            items.append({"title": title, "url": url, "snippet": snippet})
        return items[:5]

    def _normalize_web_reading_evidence_items(self, raw_items: Any) -> List[Dict[str, Any]]:
        """归一化页面证据，保留完成合同需要的 url/summary/quality/link_count。"""
        items: List[Dict[str, Any]] = []
        for item in list(raw_items or []):
            if not isinstance(item, dict):
                continue
            summary = self._truncate_text(item.get("summary"), max_chars=420)
            url = self._truncate_text(item.get("url"), max_chars=240)
            if not summary and not url:
                continue
            items.append(
                {
                    "function_name": str(item.get("function_name") or "").strip(),
                    "url": url,
                    "title": self._truncate_text(item.get("title"), max_chars=100),
                    "summary": summary,
                    "content_length": int(item.get("content_length") or 0),
                    "link_count": int(item.get("link_count") or 0),
                    "quality": str(item.get("quality") or "").strip(),
                }
            )
        return items[:6]

    @staticmethod
    def _dedupe_evidence_items(items: List[Dict[str, Any]], *, key_names: tuple[str, ...]) -> List[Dict[str, Any]]:
        deduped: List[Dict[str, Any]] = []
        seen_keys: set[str] = set()
        for item in items:
            key = "|".join([str(item.get(key_name) or "").strip() for key_name in key_names]).strip("|")
            if not key:
                key = str(item).strip()
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped.append(item)
        return deduped

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
            # facts_in_session 沉淀的是可复用证据，不应在上下文服务中再次压缩。
            "facts_in_session": [
                str(item)
                for item in normalize_text_list(working_memory.get("facts_in_session"))[:_OPEN_QUESTION_LIMIT]
            ],
            "user_preferences": dict(working_memory.get("user_preferences") or {}),
        }
        if stage == "summary":
            # Summary 只消费轻量工作记忆与结构化上下文，最终正文由 summary_node 统一组织。
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
                "final_answer_text": self._truncate_text(
                    state.get("final_answer_text"),
                    max_chars=_FINAL_MESSAGE_MAX_CHARS,
                ),
                "selected_artifacts": normalize_file_path_list(state.get("selected_artifacts"))[:_ARTIFACT_LIMIT],
            }
        if policy.stable_background.include_topic_anchor:
            topic_anchor = self._build_topic_anchor(state=state)
            if topic_anchor:
                background["topic_anchor"] = topic_anchor
        return {
            key: value
            for key, value in background.items()
            if self._has_visible_value(value)
        }

    def _build_topic_anchor(self, *, state: PlannerReActLangGraphState) -> Dict[str, Any]:
        """为 direct_answer 生成显式主题锚点，避免追问时主题漂移。"""
        conversation_summary = self._truncate_text(
            state.get("conversation_summary"),
            max_chars=_SUMMARY_MAX_CHARS,
        )
        if conversation_summary:
            return {
                "source": "conversation_summary",
                "text": conversation_summary,
            }

        run_brief_anchor = self._build_topic_anchor_from_recent_run_brief(state=state)
        if run_brief_anchor:
            return run_brief_anchor

        message_window_anchor = self._build_topic_anchor_from_recent_messages(state=state)
        if message_window_anchor:
            return message_window_anchor

        previous_final_message = self._truncate_text(
            state.get("previous_final_message"),
            max_chars=_FINAL_MESSAGE_MAX_CHARS,
        )
        if previous_final_message:
            return {
                "source": "previous_final_message",
                "text": previous_final_message,
            }

        final_message = self._truncate_text(
            state.get("final_answer_text") or state.get("final_message"),
            max_chars=_FINAL_MESSAGE_MAX_CHARS,
        )
        if final_message:
            return {
                "source": "final_answer_text" if state.get("final_answer_text") else "final_message",
                "text": final_message,
            }
        return {}

    def _build_topic_anchor_from_recent_run_brief(
            self,
            *,
            state: PlannerReActLangGraphState,
    ) -> Dict[str, Any]:
        for item in list(state.get("recent_run_briefs") or []):
            if not isinstance(item, dict):
                continue
            title = self._truncate_text(item.get("title"), max_chars=80)
            goal = self._truncate_text(item.get("goal"), max_chars=120)
            summary = self._truncate_text(
                item.get("final_answer_summary") or item.get("final_answer_text_excerpt"),
                max_chars=200,
            )
            parts = [value for value in (title, goal, summary) if value]
            if not parts:
                continue
            return {
                "source": "recent_run_brief",
                "text": " | ".join(parts[:3]),
            }
        return {}

    def _build_topic_anchor_from_recent_messages(
            self,
            *,
            state: PlannerReActLangGraphState,
    ) -> Dict[str, Any]:
        normalized_messages = self._build_recent_messages(state=state)
        if not normalized_messages:
            return {}

        last_assistant_index = -1
        for index in range(len(normalized_messages) - 1, -1, -1):
            if str(normalized_messages[index].get("role") or "").strip() == "assistant":
                last_assistant_index = index
                break
        if last_assistant_index < 0:
            return {}

        assistant_message = self._truncate_text(
            normalized_messages[last_assistant_index].get("message"),
            max_chars=160,
        )
        user_message = ""
        for index in range(last_assistant_index - 1, -1, -1):
            if str(normalized_messages[index].get("role") or "").strip() == "user":
                user_message = self._truncate_text(
                    normalized_messages[index].get("message"),
                    max_chars=120,
                )
                break

        parts = []
        if user_message:
            parts.append(f"用户主题: {user_message}")
        if assistant_message:
            parts.append(f"最近回复: {assistant_message}")
        if not parts:
            return {}
        return {
            "source": "message_window",
            "text": " | ".join(parts),
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

    @staticmethod
    def _get_workspace_environment_summary(
            workspace_snapshot: Optional[WorkspaceEnvironmentSnapshot],
    ) -> Dict[str, Any]:
        if workspace_snapshot is None:
            return {}
        return dict(workspace_snapshot.workspace.environment_summary or {})

    def _get_workspace_artifact_paths(
            self,
            *,
            workspace_snapshot: Optional[WorkspaceEnvironmentSnapshot],
    ) -> List[str]:
        if workspace_snapshot is None:
            return []
        paths: List[str] = []
        for artifact in workspace_snapshot.artifacts:
            path = str(artifact.path or "").strip()
            if not path or path in paths:
                continue
            paths.append(path)
        return paths[:_ARTIFACT_LIMIT]

    async def list_workspace_artifact_paths(self) -> List[str]:
        """返回当前 workspace 已索引的产物路径。"""
        if self._workspace_runtime_service is None:
            return []
        workspace_snapshot = await self._workspace_runtime_service.build_environment_snapshot()
        return self._get_workspace_artifact_paths(workspace_snapshot=workspace_snapshot)

    def _collect_recent_search_queries(
            self,
            *,
            state: PlannerReActLangGraphState,
            task_mode: str,
    ) -> List[str]:
        stored_digest = self._read_state_digest(
            state=state,
            field_name="recent_action_digest",
            expected_task_mode=task_mode,
        )
        return normalize_text_list(stored_digest.get("recent_search_queries"))[:_OPEN_QUESTION_LIMIT]

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
            "task_mode_hint": normalize_controlled_value(getattr(target_step, "task_mode_hint", None),
                                                         StepTaskModeHint),
            "output_mode": normalize_controlled_value(getattr(target_step, "output_mode", None), StepOutputMode),
            "status": str(getattr(target_step, "status", "") or "").strip(),
        }
        if target_step.outcome is not None:
            current_step["result"] = {
                "summary": normalize_step_result_text(target_step.outcome.summary),
                "blockers": [
                    self._truncate_text(item, max_chars=120)
                    for item in normalize_text_list(target_step.outcome.blockers)[:_OPEN_QUESTION_LIMIT]
                ],
                # 当前步骤执行结果里的证据文本必须完整暴露给后续 prompt，上层模型才能直接复用。
                "facts_learned": [
                    str(item)
                    for item in normalize_text_list(target_step.outcome.facts_learned)[:_OPEN_QUESTION_LIMIT]
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
                    # 已完成步骤摘要中的 facts_learned 同样保留全文，避免下游只能看到残缺片段。
                    "facts_learned": [
                        str(item)
                        for item in normalize_text_list(outcome.get("facts_learned"))[:_OPEN_QUESTION_LIMIT]
                    ],
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

    def _collect_candidate_links(
            self,
            *,
            workspace_snapshot: Optional[WorkspaceEnvironmentSnapshot] = None,
    ) -> List[Dict[str, str]]:
        workspace_links = list(self._get_workspace_environment_summary(workspace_snapshot).get("candidate_links") or [])
        return [item for item in workspace_links if isinstance(item, dict)][: _OPEN_QUESTION_LIMIT]

    def _collect_search_evidence_summaries(
            self,
            *,
            workspace_snapshot: Optional[WorkspaceEnvironmentSnapshot] = None,
    ) -> List[Dict[str, str]]:
        return [
            {
                "title": self._truncate_text(item.get("title"), max_chars=80),
                "snippet": self._truncate_text(item.get("snippet"), max_chars=220),
                "url": self._truncate_text(item.get("url"), max_chars=200),
            }
            for item in self._collect_candidate_links(workspace_snapshot=workspace_snapshot)
            if isinstance(item, dict)
        ]

    def _collect_fetch_page_summaries(
            self,
            *,
            workspace_snapshot: Optional[WorkspaceEnvironmentSnapshot] = None,
    ) -> List[Dict[str, Any]]:
        workspace_pages = list(
            self._get_workspace_environment_summary(workspace_snapshot).get("read_page_summaries") or [])
        return [item for item in workspace_pages if isinstance(item, dict)][: _OPEN_QUESTION_LIMIT]

    def _collect_browser_page_summary(
            self,
            *,
            workspace_snapshot: Optional[WorkspaceEnvironmentSnapshot] = None,
    ) -> Dict[str, Any]:
        stored_page = dict(
            workspace_snapshot.workspace.browser_snapshot or {}) if workspace_snapshot is not None else {}
        return {
            key: value
            for key, value in stored_page.items()
            if key in {"url", "title", "page_type", "main_content_summary", "actionable_elements", "degrade_reason"}
        }

    def _extract_latest_shell_cwd(
            self,
            *,
            workspace_snapshot: Optional[WorkspaceEnvironmentSnapshot] = None,
    ) -> str:
        if workspace_snapshot is not None:
            return str(workspace_snapshot.workspace.cwd or "").strip()
        return ""

    def _collect_recent_changed_files(
            self,
            *,
            workspace_snapshot: Optional[WorkspaceEnvironmentSnapshot] = None,
    ) -> List[str]:
        workspace_files = normalize_file_path_list(
            self._get_workspace_environment_summary(workspace_snapshot).get("recent_changed_files")
        )
        return workspace_files[:_ARTIFACT_LIMIT]

    def _collect_file_tree_summary(
            self,
            *,
            workspace_snapshot: Optional[WorkspaceEnvironmentSnapshot] = None,
    ) -> List[str]:
        workspace_summaries = normalize_text_list(
            self._get_workspace_environment_summary(workspace_snapshot).get("file_tree_summary")
        )
        return workspace_summaries[:_OPEN_QUESTION_LIMIT]

    def _collect_available_artifacts(
            self,
            *,
            workspace_snapshot: Optional[WorkspaceEnvironmentSnapshot] = None,
    ) -> List[str]:
        return self._get_workspace_artifact_paths(workspace_snapshot=workspace_snapshot)

    def _extract_shell_session_status(
            self,
            *,
            workspace_snapshot: Optional[WorkspaceEnvironmentSnapshot] = None,
    ) -> str:
        return str(
            self._get_workspace_environment_summary(workspace_snapshot).get("shell_session_status") or ""
        ).strip()

    def _extract_latest_shell_result(
            self,
            *,
            workspace_snapshot: Optional[WorkspaceEnvironmentSnapshot] = None,
    ) -> Dict[str, Any]:
        workspace_result = self._get_workspace_environment_summary(workspace_snapshot).get("latest_shell_result")
        if isinstance(workspace_result, dict) and workspace_result:
            return dict(workspace_result)
        return {}
