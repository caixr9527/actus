#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层 graph control 元数据 helper。

这些函数只负责读取、写回和清理 graph control 字段，不承载业务流程决策。
独立出来后，节点实现可以继续使用同一组控制态语义，避免在不同节点里重复散落字段名。
"""

from typing import Any, Dict

from app.domain.models import Step
from app.domain.services.runtime.langgraph_state import (
    PlannerReActLangGraphState,
    get_graph_control,
    replace_graph_control,
)


def get_control_metadata(state: PlannerReActLangGraphState) -> Dict[str, Any]:
    return get_graph_control(state.get("graph_metadata"))


def replace_control_metadata(state: PlannerReActLangGraphState, control: Dict[str, Any]) -> Dict[str, Any]:
    return replace_graph_control(state.get("graph_metadata"), control)


def is_direct_wait_execute_step(step: Step, control: Dict[str, Any]) -> bool:
    """识别 direct_wait 确认后的真实执行步骤。"""
    return (
            str(control.get("entry_strategy") or "").strip() == "direct_wait"
            and str(step.id or "").strip() == "direct-wait-execute"
    )


def clear_direct_wait_control_state(control: Dict[str, Any]) -> None:
    """清理 direct_wait 专属控制字段，避免取消后误伤后续重规划链路。"""
    if str(control.get("entry_strategy") or "").strip() == "direct_wait":
        control.pop("entry_strategy", None)
    control.pop("skip_replan_when_plan_finished", None)
    control.pop("direct_wait_original_message", None)
    control.pop("direct_wait_execute_task_mode", None)
    control.pop("direct_wait_original_task_executed", None)


def clear_plan_only_control_state(control: Dict[str, Any]) -> None:
    """清理仅规划模式的控制字段，避免跨 run 残留。"""
    control.pop("plan_only", None)
