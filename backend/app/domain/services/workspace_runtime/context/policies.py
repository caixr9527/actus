#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 上下文可见性策略。"""

from dataclasses import dataclass
from typing import FrozenSet

from app.domain.models import StepTaskModeHint

from .contracts import PromptStage


@dataclass(frozen=True)
class StableBackgroundPolicy:
    """稳定背景字段的可见性策略。"""

    include_conversation_summary: bool = True
    include_recent_messages: bool = False
    include_recent_run_briefs: bool = False
    include_recent_attempt_briefs: bool = False
    include_completed_steps: bool = False
    include_plan_snapshot: bool = False
    include_summary_focus: bool = False


@dataclass(frozen=True)
class ContextPolicy:
    """定义某个 stage + task_mode 下 Prompt 可读取的上下文范围。"""

    include_current_step: bool = False
    include_open_questions: bool = False
    include_pending_confirmation: bool = False
    include_environment_digest: bool = False
    include_observation_digest: bool = False
    include_recent_action_digest: bool = False
    include_working_memory_digest: bool = False
    include_retrieved_memory_digest: bool = False
    # 按 stage + task_mode 控制召回记忆的类型和数量，避免统一上限导致噪音扩散。
    retrieved_memory_allowed_types: FrozenSet[str] = frozenset()
    retrieved_memory_max_items: int = 0
    include_stable_background: bool = False
    include_audit_refs: bool = False
    stable_background: StableBackgroundPolicy = StableBackgroundPolicy()


_DEFAULT_POLICY = ContextPolicy()

_EXECUTE_BASE_POLICY = ContextPolicy(
    include_current_step=True,
    include_open_questions=True,
    include_pending_confirmation=True,
    include_environment_digest=True,
    include_observation_digest=True,
    include_recent_action_digest=True,
    include_working_memory_digest=True,
    include_retrieved_memory_digest=True,
    retrieved_memory_allowed_types=frozenset({"profile", "fact", "instruction"}),
    retrieved_memory_max_items=4,
    include_stable_background=True,
    include_audit_refs=True,
)


def _build_execute_policy(
        *,
        stable_background: StableBackgroundPolicy,
        include_observation_digest: bool = True,
        include_retrieved_memory_digest: bool = True,
        retrieved_memory_allowed_types: FrozenSet[str] = _EXECUTE_BASE_POLICY.retrieved_memory_allowed_types,
        retrieved_memory_max_items: int = _EXECUTE_BASE_POLICY.retrieved_memory_max_items,
) -> ContextPolicy:
    """execute 阶段默认字段高度一致，只让各模式声明差异项。"""
    return ContextPolicy(
        include_current_step=_EXECUTE_BASE_POLICY.include_current_step,
        include_open_questions=_EXECUTE_BASE_POLICY.include_open_questions,
        include_pending_confirmation=_EXECUTE_BASE_POLICY.include_pending_confirmation,
        include_environment_digest=_EXECUTE_BASE_POLICY.include_environment_digest,
        include_observation_digest=include_observation_digest,
        include_recent_action_digest=_EXECUTE_BASE_POLICY.include_recent_action_digest,
        include_working_memory_digest=_EXECUTE_BASE_POLICY.include_working_memory_digest,
        include_retrieved_memory_digest=include_retrieved_memory_digest,
        retrieved_memory_allowed_types=retrieved_memory_allowed_types,
        retrieved_memory_max_items=retrieved_memory_max_items,
        include_stable_background=_EXECUTE_BASE_POLICY.include_stable_background,
        include_audit_refs=_EXECUTE_BASE_POLICY.include_audit_refs,
        stable_background=stable_background,
    )

_POLICIES: dict[tuple[PromptStage, str], ContextPolicy] = {
    (
        "planner",
        StepTaskModeHint.GENERAL.value,
    ): ContextPolicy(
        include_open_questions=True,
        include_working_memory_digest=True,
        include_retrieved_memory_digest=True,
        # P3-一次性收口：planner 不消费 profile 记忆，避免把历史偏好误当本轮已确认事实。
        retrieved_memory_allowed_types=frozenset({"fact", "instruction"}),
        retrieved_memory_max_items=5,
        include_stable_background=True,
        stable_background=StableBackgroundPolicy(
            include_conversation_summary=True,
            include_recent_messages=True,
            include_recent_run_briefs=True,
            include_recent_attempt_briefs=True,
        ),
    ),
    (
        "execute",
        StepTaskModeHint.RESEARCH.value,
    ): _build_execute_policy(
        retrieved_memory_allowed_types=frozenset({"profile", "fact", "instruction"}),
        retrieved_memory_max_items=4,
        stable_background=StableBackgroundPolicy(
            include_conversation_summary=True,
            include_recent_run_briefs=True,
            include_recent_attempt_briefs=True,
            include_completed_steps=True,
        ),
    ),
    (
        "execute",
        StepTaskModeHint.WEB_READING.value,
    ): _build_execute_policy(
        retrieved_memory_allowed_types=frozenset({"profile", "fact", "instruction"}),
        retrieved_memory_max_items=4,
        stable_background=StableBackgroundPolicy(
            include_conversation_summary=True,
            include_recent_run_briefs=True,
            include_recent_attempt_briefs=True,
            include_completed_steps=True,
        ),
    ),
    (
        "execute",
        StepTaskModeHint.BROWSER_INTERACTION.value,
    ): _build_execute_policy(
        retrieved_memory_allowed_types=frozenset({"profile", "instruction"}),
        retrieved_memory_max_items=3,
        stable_background=StableBackgroundPolicy(
            include_conversation_summary=True,
            include_recent_attempt_briefs=True,
            include_completed_steps=True,
        ),
    ),
    (
        "execute",
        StepTaskModeHint.CODING.value,
    ): _build_execute_policy(
        retrieved_memory_allowed_types=frozenset({"profile", "fact", "instruction"}),
        retrieved_memory_max_items=4,
        stable_background=StableBackgroundPolicy(
            include_conversation_summary=True,
            include_completed_steps=True,
        ),
    ),
    (
        "execute",
        StepTaskModeHint.FILE_PROCESSING.value,
    ): _build_execute_policy(
        retrieved_memory_allowed_types=frozenset({"profile", "fact", "instruction"}),
        retrieved_memory_max_items=4,
        stable_background=StableBackgroundPolicy(
            include_conversation_summary=True,
            include_completed_steps=True,
        ),
    ),
    (
        "execute",
        StepTaskModeHint.HUMAN_WAIT.value,
    ): _build_execute_policy(
        include_observation_digest=False,
        include_retrieved_memory_digest=False,
        retrieved_memory_allowed_types=frozenset(),
        retrieved_memory_max_items=0,
        stable_background=StableBackgroundPolicy(
            include_conversation_summary=True,
        ),
    ),
    (
        "execute",
        StepTaskModeHint.GENERAL.value,
    ): _build_execute_policy(
        retrieved_memory_allowed_types=frozenset({"profile", "fact", "instruction"}),
        retrieved_memory_max_items=4,
        stable_background=StableBackgroundPolicy(
            include_conversation_summary=True,
            include_completed_steps=True,
        ),
    ),
    (
        "replan",
        StepTaskModeHint.GENERAL.value,
    ): ContextPolicy(
        include_current_step=True,
        include_open_questions=True,
        include_observation_digest=True,
        include_recent_action_digest=True,
        include_working_memory_digest=True,
        include_retrieved_memory_digest=True,
        retrieved_memory_allowed_types=frozenset({"profile", "fact", "instruction"}),
        retrieved_memory_max_items=4,
        include_stable_background=True,
        include_audit_refs=True,
        stable_background=StableBackgroundPolicy(
            include_conversation_summary=True,
            include_recent_run_briefs=True,
            include_recent_attempt_briefs=True,
            include_completed_steps=True,
            include_plan_snapshot=True,
        ),
    ),
    (
        "summary",
        StepTaskModeHint.GENERAL.value,
    ): ContextPolicy(
        include_working_memory_digest=True,
        include_stable_background=True,
        stable_background=StableBackgroundPolicy(
            include_conversation_summary=True,
            include_recent_run_briefs=True,
            include_recent_attempt_briefs=True,
            include_completed_steps=True,
            include_plan_snapshot=True,
            include_summary_focus=True,
        ),
    ),
}


def get_context_policy(stage: PromptStage, task_mode: str) -> ContextPolicy:
    """按 stage + task_mode 返回上下文可见性策略。"""
    normalized_task_mode = str(task_mode or "").strip().lower() or StepTaskModeHint.GENERAL.value
    return _POLICIES.get((stage, normalized_task_mode)) or _POLICIES.get(
        (stage, StepTaskModeHint.GENERAL.value),
        _DEFAULT_POLICY,
    )
