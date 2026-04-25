#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 上下文可见性策略。"""

from dataclasses import dataclass
from typing import FrozenSet

from app.domain.models import StepTaskModeHint

from .contracts import PromptStage


@dataclass(frozen=True)
class StableBackgroundPolicy:
    """稳定背景字段的可见性策略。

    这里控制的是“跨阶段相对稳定、不会随着本轮单步执行频繁抖动”的背景信息。
    这些字段主要用于给 planner / replan / summary / direct_answer 提供长期上下文锚点，
    避免模型只看到当前步骤局部信息而丢失会话主线。
    """

    # 会话级主题摘要，用于告诉模型“这轮会话整体在讨论什么”。
    include_conversation_summary: bool = True
    # 最近对话消息窗口，主要用于保持追问、澄清、直接回复时的上下文连续性。
    include_recent_messages: bool = False
    # 最近几轮运行的简报，用于让模型知道最近已经做过哪些事、产出过什么结果。
    include_recent_run_briefs: bool = False
    # 最近失败或未完成尝试的简报，用于提醒模型避免重复走回头路。
    include_recent_attempt_briefs: bool = False
    # 已完成步骤摘要，用于让后续步骤继承前面已经拿到的阶段性结论与事实。
    include_completed_steps: bool = False
    # 当前计划快照，用于 replan / summary 识别原计划结构、执行进度和剩余步骤。
    include_plan_snapshot: bool = False
    # 最终收口焦点，承载 final_message / final_answer_text / selected_artifacts 等交付锚点。
    include_summary_focus: bool = False
    # direct_answer 需要显式主题锚点，避免模型只看到弱历史时自行漂移主题。
    include_topic_anchor: bool = False


@dataclass(frozen=True)
class ContextPolicy:
    """定义某个 stage + task_mode 下 Prompt 可读取的上下文范围。

    `ContextPolicy` 的职责不是“控制模型能力”，而是声明：
    - 当前阶段真正需要看到哪些上下文字段；
    - 哪些字段应被隐藏以减少噪音、避免误导；
    - 召回记忆允许进入什么类型、最多多少条。

    业务上可理解为 Runtime 的“上下文配方”：
    planner、execute、replan、summary、direct_answer 各自看到的材料不同，
    由这里统一定义，避免不同节点各自拼 Prompt 导致语义漂移。
    """

    # 当前正在处理的步骤定义。
    # 业务含义：告诉模型“此刻这一阶段到底要完成什么 step”，是 execute / replan 的主任务输入。
    include_current_step: bool = False
    # 当前仍未解决的问题列表。
    # 业务含义：提醒模型还有哪些缺口、待确认项、待补证点，避免遗漏关键约束。
    include_open_questions: bool = False
    # 等待用户确认/补充的信息。
    # 业务含义：让模型知道当前是否存在 human_wait 语义、应该向用户确认什么。
    include_pending_confirmation: bool = False
    # 环境摘要，如工作区、文件、工具环境等稳定执行条件。
    # 业务含义：让模型了解“当前可用的执行环境是什么”，便于决定是否写文件、读文件、继续操作。
    include_environment_digest: bool = False
    # 观察摘要，如本轮已看到的页面、文件、命令结果等观测事实。
    # 业务含义：为模型提供“已经观察到什么”的最新外部事实，避免重复读取。
    include_observation_digest: bool = False
    # 最近一次动作摘要，如最近搜索、抓页、阻断、失败原因、研究进度等。
    # 业务含义：承担 step 内以及 step 间的短期执行记忆，帮助后续步骤复用近端证据。
    include_recent_action_digest: bool = False
    # 工作记忆摘要，汇总当前运行过程中已形成的中间事实、结论、产物线索。
    # 业务含义：给 planner / execute / summary 提供“当前运行已经积累了什么”。
    include_working_memory_digest: bool = False
    # 检索得到的长期记忆摘要。
    # 业务含义：把历史沉淀的用户事实、偏好、指令约束按需注入本轮推理。
    include_retrieved_memory_digest: bool = False
    # 按 stage + task_mode 控制召回记忆的类型和数量，避免统一上限导致噪音扩散。
    # 业务含义：限制当前阶段允许使用哪些记忆类型，避免把不相关历史带进来。
    retrieved_memory_allowed_types: FrozenSet[str] = frozenset()
    # 业务含义：限制记忆注入条数，避免 Prompt 被历史记忆淹没。
    retrieved_memory_max_items: int = 0
    # 是否注入稳定背景区块。
    # 业务含义：在需要跨轮/跨阶段承接主题时，给模型提供会话主线和历史锚点。
    include_stable_background: bool = False
    # 是否注入审计引用，如 run_id / step_id / trace 引用信息。
    # 业务含义：主要服务于可追溯执行与节点级诊断，不直接增加业务事实，但便于链路对齐。
    include_audit_refs: bool = False
    # 稳定背景子策略。
    # 业务含义：只有在 include_stable_background=True 时生效，用来细分背景区块里具体放哪些字段。
    stable_background: StableBackgroundPolicy = StableBackgroundPolicy()


_DEFAULT_POLICY = ContextPolicy()

# execute 阶段的共性基线：
# - 默认需要当前 step、本轮环境、最近动作、工作记忆与少量长期记忆；
# - 各 task_mode 只声明差异，不再重复维护整套字段。
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
        "direct_answer",
        StepTaskModeHint.GENERAL.value,
    ): ContextPolicy(
        # direct_answer 不执行工具，也不关心当前 step，
        # 核心是基于已有工作记忆和稳定背景直接回答用户追问。
        include_working_memory_digest=True,
        include_stable_background=True,
        stable_background=StableBackgroundPolicy(
            include_conversation_summary=True,
            include_recent_messages=True,
            include_recent_run_briefs=True,
            include_summary_focus=True,
            include_topic_anchor=True,
        ),
    ),
    (
        "planner",
        StepTaskModeHint.GENERAL.value,
    ): ContextPolicy(
        # planner 要重新理解用户目标、约束、未解决问题并生成计划，
        # 因此重点看 open_questions、工作记忆和稳定历史背景。
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
        # research 需要继承最近运行简报和已完成步骤，
        # 这样后续检索可以围绕已有证据继续扩展，而不是从零开始搜。
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
        # web_reading 依赖 research 交过来的候选链接、历史失败尝试和已完成步骤，
        # 因此背景口径与 research 基本一致。
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
        # browser_interaction 更依赖当前页面状态和操作约束，
        # 不需要太多历史事实，主要保留 profile/instruction 以及最近失败尝试。
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
        # coding 要承接当前会话目标和已完成步骤的产物/结论，
        # 但通常不需要过多历史运行简报，避免代码上下文被历史任务噪音干扰。
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
        # file_processing 与 coding 类似，重点是当前任务、环境和已完成步骤的文件结论承接。
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
        # human_wait 的职责只是等待用户补充/确认，不需要观察摘要或长期记忆，
        # 否则容易把等待步骤误引向分析或执行。
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
        # general 是通用执行模式，默认继承当前会话摘要和已完成步骤，
        # 让模型能基于已有中间结果继续整理、总结或产出。
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
        # replan 既要看当前 step 的失败/阻断上下文，也要看原计划和已完成步骤，
        # 才能决定是改写当前步骤、补插步骤，还是整体重排计划。
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
        # summary 不负责继续执行，而是基于工作记忆、计划快照和交付焦点做最终收口，
        # 因此需要完整的稳定背景，但不再需要环境/观察/最近动作这些执行态细节。
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
