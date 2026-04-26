#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""执行约束层契约定义。"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from app.domain.models import Step
from app.domain.services.tools import BaseTool
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.execution_context import (
    ExecutionContext,
)
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.execution_state import (
    ExecutionState,
)


@dataclass(slots=True)
class ConstraintInput:
    """执行前约束评估输入。

    业务含义：
    - 描述“当前这一轮准备调用哪个工具、处于什么任务模式、当前执行态是什么”；
    - 由 `ToolPolicyEngine` 在真正执行工具前组装，只读传入 `ConstraintEngine`；
    - policy 只能基于该快照做 `allow/block/rewrite` 决策，禁止直接修改执行状态。
    """

    step: Step
    task_mode: str
    # 模型当前轮选中的原始函数名，尚未经过 rewrite 收口。
    function_name: str
    # 标准化函数名，供 policy 做稳定判断，不依赖大小写或格式差异。
    normalized_function_name: str
    # 当前轮候选工具参数；若发生 rewrite，会在引擎内部替换为 rewrite 后参数再二次评估。
    function_args: Dict[str, Any]
    # 当前函数名对应的工具实现；仅用于判断工具是否存在，不允许 policy 直接调用。
    matched_tool: Optional[BaseTool]
    # 当前轮已被上游明确拉黑的函数集合，主要用于 human_wait / task_mode 等强约束。
    iteration_blocked_function_names: Set[str]
    # 当前步骤级上下文快照，包含任务模式、工具白名单、显式 URL、文件上下文等只读事实。
    execution_context: ExecutionContext
    # 当前步骤执行态，包含研究进展、浏览器进展、重复调用计数等运行中状态。
    execution_state: ExecutionState
    # 来自反馈层的只读外部信号快照；执行约束层只读消费，不负责写回。
    external_signals_snapshot: Dict[str, Any] = field(default_factory=dict)
    # 当前轮可用工具集合；仅供 rewrite 后重新解析 matched_tool，禁止 policy 直接调用。
    runtime_tools: List[BaseTool] = field(default_factory=list)


@dataclass(slots=True)
class ConstraintToolResultPayload:
    """约束层输出给工具循环消费的失败结果契约。

    业务含义：
    - 仅在 `block` 场景下使用；
    - 让工具循环把“约束拦截”当作一次标准失败结果写回模型与事件链路；
    - 不允许塞入真实工具执行产物，避免把 guard 与 executor 语义混淆。
    """

    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConstraintDecision:
    """执行约束层决策契约。

    业务含义：
    - `action` 是约束层对本轮工具调用的最终判断：`allow / block / rewrite`；
    - `reason_code` 是唯一可追溯原因码，供日志、effects、收敛层统一消费；
    - `block_mode` 与 `loop_break_reason` 共同决定是否立即收敛当前步骤；
    - `rewrite_target` 仅描述最终要执行的函数名与参数，不允许改 step 语义。
    """

    action: str
    reason_code: str
    block_mode: str = ""
    loop_break_reason: str = ""
    tool_result_payload: Optional[ConstraintToolResultPayload] = None
    message_for_model: str = ""
    rewrite_target: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ConstraintPolicyTraceEntry:
    """单条策略评估追踪记录。

    业务含义：
    - 记录每个 policy 在当前轮的判断结果；
    - 用于最终日志回放与问题定位，帮助确认“是谁命中了 block/rewrite”。
    """

    policy_name: str
    action: str
    reason_code: str


@dataclass(slots=True)
class ConstraintEngineResult:
    """执行约束引擎最终输出。

    业务含义：
    - `constraint_decision` 是最终执行前约束结论；
    - `final_function_*` 是经过 rewrite 收口后的真实执行目标；
    - `policy_trace / winning_policy / tool_call_fingerprint` 组成可追溯诊断信息；
    - 由 `ToolPolicyEngine` 统一消费，驱动后续 executor / effects / tool events。
    """

    constraint_decision: ConstraintDecision
    final_function_name: str
    final_normalized_function_name: str
    final_function_args: Dict[str, Any]
    policy_trace: List[ConstraintPolicyTraceEntry] = field(default_factory=list)
    winning_policy: str = ""
    tool_call_fingerprint: str = ""
    rewrite_applied: bool = False
    rewrite_reason: str = ""
    rewrite_metadata: Dict[str, Any] = field(default_factory=dict)


def build_default_external_signals_snapshot(
        snapshot: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """约束层默认快照，避免调用点反复判空。"""
    normalized = dict(snapshot or {})
    normalized.setdefault("blocked_fingerprints", [])
    normalized.setdefault("avoid_url_keys", [])
    normalized.setdefault("avoid_domains", [])
    normalized.setdefault("avoid_query_patterns", [])
    normalized.setdefault("recent_failure_reasons", [])
    return normalized
