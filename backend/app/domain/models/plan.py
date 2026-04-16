#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/19 18:50
@Author : caixiaorong01@outlook.com
@File   : plan.py
"""
import uuid
import hashlib
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


def build_step_objective_source(title: str, description: str) -> str:
    """构建步骤 objective_key 的稳定语义源文本。"""
    normalized_parts: List[str] = []
    for raw_item in [title, description]:
        normalized_item = str(raw_item or "").strip()
        if normalized_item and normalized_item not in normalized_parts:
            normalized_parts.append(normalized_item)
    if not normalized_parts:
        return "empty-step"
    return " | ".join(normalized_parts)


def build_step_objective_key(title: str, description: str) -> str:
    """根据稳定语义源文本生成 objective_key。"""
    source_text = build_step_objective_source(title=title, description=description)
    return hashlib.md5(source_text.encode("utf-8")).hexdigest()[:16]


class StepTaskModeHint(str, Enum):
    """步骤结构化任务模式。"""

    GENERAL = "general"
    RESEARCH = "research"
    WEB_READING = "web_reading"
    BROWSER_INTERACTION = "browser_interaction"
    FILE_PROCESSING = "file_processing"
    CODING = "coding"
    HUMAN_WAIT = "human_wait"


class StepOutputMode(str, Enum):
    """步骤结构化输出模式。"""

    NONE = "none"
    INLINE = "inline"
    FILE = "file"


class StepArtifactPolicy(str, Enum):
    """步骤结构化产物策略。"""

    DEFAULT = "default"
    FORBID_FILE_OUTPUT = "forbid_file_output"
    ALLOW_FILE_OUTPUT = "allow_file_output"
    REQUIRE_FILE_OUTPUT = "require_file_output"


class StepDeliveryRole(str, Enum):
    """步骤结构化交付角色。"""

    NONE = "none"
    INTERMEDIATE = "intermediate"
    FINAL = "final"


class StepDeliveryContextState(str, Enum):
    """最终交付上下文准备状态。"""

    NONE = "none"
    NEEDS_PREPARATION = "needs_preparation"
    READY = "ready"


class ExecutionStatus(str, Enum):
    """执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepOutcome(BaseModel):
    """步骤执行结果。"""

    done: bool = False
    summary: str = ""
    # 重交付正文只在明确需要时写入，避免和轻量 summary 混用。
    delivery_text: str = ""
    produced_artifacts: List[str] = Field(default_factory=list)
    blockers: List[str] = Field(default_factory=list)
    facts_learned: List[str] = Field(default_factory=list)
    open_questions: List[str] = Field(default_factory=list)
    # 结构化附件交付偏好：False 表示本步骤明确禁止最终附件交付；None 表示未显式声明。
    deliver_result_as_attachment: Optional[bool] = None
    next_hint: Optional[str] = None
    reused_from_run_id: Optional[str] = None
    reused_from_step_id: Optional[str] = None


class Step(BaseModel):
    """步骤/子任务。"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    description: str = ""
    # 仅供执行链路使用的内部模板，不用于前端展示。
    execution_template: str = ""
    # 执行模板中依赖的槽位键；用于 wait->resume 后参数注入。
    required_slots: List[str] = Field(default_factory=list)
    # 当前步骤已绑定的槽位值；执行前用于渲染执行文本。
    execution_slots: Dict[str, Any] = Field(default_factory=dict)
    # 结构化步骤模式优先，文本规则只做兜底。
    task_mode_hint: Optional[StepTaskModeHint] = None
    # 结构化输出模式，表示该步骤默认产出形态。
    output_mode: Optional[StepOutputMode] = None
    # 结构化产物策略，决定当前步骤是否允许或要求文件产出。
    artifact_policy: Optional[StepArtifactPolicy] = None
    # 结构化交付角色，显式标记该步骤是否承担最终重交付正文。
    delivery_role: Optional[StepDeliveryRole] = None
    # 结构化交付上下文状态，仅在 final 步骤下有意义。
    # needs_preparation 表示本步骤仍需先检索/读取/操作，再输出最终正文；
    # ready 表示前序上下文已准备好，本步骤应直接组织最终正文。
    delivery_context_state: Optional[StepDeliveryContextState] = None
    objective_key: str = ""
    success_criteria: List[str] = Field(default_factory=list)
    status: ExecutionStatus = ExecutionStatus.PENDING
    outcome: Optional[StepOutcome] = None
    error: Optional[str] = None

    @property
    def done(self) -> bool:
        """判断任务是否完成。"""
        return self.status in [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        ]


class Plan(BaseModel):
    """计划模型，拆分出来的子任务/子步骤。"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = ""
    goal: str = ""
    language: str = ""
    steps: List[Step] = Field(default_factory=list)
    message: str = ""
    status: ExecutionStatus = ExecutionStatus.PENDING
    error: Optional[str] = None

    @property
    def done(self) -> bool:
        """判断任务是否完成。"""
        return self.status in [
            ExecutionStatus.COMPLETED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        ]

    def get_next_step(self) -> Optional[Step]:
        """获取下一个未完成的步骤。"""
        return next((step for step in self.steps if not step.done), None)
