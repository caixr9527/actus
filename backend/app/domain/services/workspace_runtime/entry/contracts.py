#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 入口合同模型。

EntryContract 是入口层唯一真相源：它描述当前用户请求应走哪条执行通道、需要多少上下文、
允许多少工具预算，以及后续是否可以从原子动作升级到 Planner。
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field

from app.domain.models import StepTaskModeHint


class EntryRoute(str, Enum):
    """入口执行通道。"""

    ANSWER = "answer"
    WAIT = "wait"
    ATOMIC_ACTION = "atomic_action"
    PLANNED_TASK = "planned_task"
    RESUME_PLAN = "resume_plan"


class EntryContextProfile(str, Enum):
    """入口阶段上下文读取档位。"""

    NONE = "none"
    MINIMAL_HISTORY = "minimal_history"
    WORKSPACE = "workspace"
    FULL = "full"


class EntryToolBudget(str, Enum):
    """入口阶段工具预算建议。"""

    NONE = "none"
    SINGLE_CALL = "single_call"
    SMALL_LOOP = "small_loop"
    PLANNER_CONTROLLED = "planner_controlled"


class EntryRiskLevel(str, Enum):
    """入口任务风险等级。"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class EntrySourceSnapshot(BaseModel):
    """入口判断时看到的原始输入快照。"""

    user_message: str = ""
    has_input_parts: bool = False
    has_active_plan: bool = False
    contextual_followup_anchor: bool = False


class EntryUpgradePolicy(BaseModel):
    """原子动作运行中升级策略。"""

    allow_upgrade: bool = False
    max_tool_calls_before_upgrade: int = 0
    upgrade_on_second_tool_family: bool = False
    upgrade_on_user_confirmation_required: bool = False
    upgrade_on_file_output_required: bool = False
    upgrade_on_open_questions: bool = False


class EntryContract(BaseModel):
    """入口编译输出合同。

    业务含义：
    - route 决定图入口分支。
    - context_profile 决定入口阶段读取多少历史与工作区上下文。
    - tool_budget 是执行器的预算建议，不替代 task mode 工具白名单。
    - source 保存原始请求，供 wait 确认恢复后继续执行同一任务。
    """

    route: EntryRoute
    task_mode: StepTaskModeHint = StepTaskModeHint.GENERAL
    context_profile: EntryContextProfile = EntryContextProfile.FULL
    tool_budget: EntryToolBudget = EntryToolBudget.PLANNER_CONTROLLED
    needs_summary: bool = True
    plan_only: bool = False
    risk_level: EntryRiskLevel = EntryRiskLevel.LOW
    complexity_score: int = Field(default=0, ge=0)
    tool_need_score: int = Field(default=0, ge=0)
    freshness_score: int = Field(default=0, ge=0)
    context_need_score: int = Field(default=0, ge=0)
    reason_codes: List[str] = Field(default_factory=list)
    upgrade_policy: EntryUpgradePolicy = Field(default_factory=EntryUpgradePolicy)
    source: EntrySourceSnapshot = Field(default_factory=EntrySourceSnapshot)
