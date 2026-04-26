#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层 graph control 元数据 helper。

这些函数只负责读取、写回和清理 graph control 字段，不承载入口决策。
入口业务语义统一来自 EntryContract，运行中升级统一来自 entry_upgrade。
"""

from typing import Any, Dict

from app.domain.models import Step
from app.domain.services.runtime.langgraph_state import (
    PlannerReActLangGraphState,
    get_graph_control,
    replace_graph_control,
)
from app.domain.services.workspace_runtime.entry import EntryContract, EntryRoute


def get_control_metadata(state: PlannerReActLangGraphState) -> Dict[str, Any]:
    return get_graph_control(state.get("graph_metadata"))


def replace_control_metadata(state: PlannerReActLangGraphState, control: Dict[str, Any]) -> Dict[str, Any]:
    return replace_graph_control(state.get("graph_metadata"), control)


def get_entry_contract_payload(control: Dict[str, Any]) -> Dict[str, Any]:
    """读取入口合同原始 payload；缺失表示入口节点未正确执行。"""
    payload = control.get("entry_contract")
    return dict(payload) if isinstance(payload, dict) else {}


def get_entry_contract(control: Dict[str, Any]) -> EntryContract:
    """解析入口合同，节点内部必须显式消费合同字段。"""
    payload = get_entry_contract_payload(control)
    if not payload:
        raise ValueError("缺少入口合同 entry_contract")
    return EntryContract.model_validate(payload)


def is_entry_wait_execute_step(step: Step, control: Dict[str, Any]) -> bool:
    """识别入口等待确认后的真实执行步骤。"""
    payload = get_entry_contract_payload(control)
    return (
            str(payload.get("route") or "").strip() == EntryRoute.WAIT.value
            and str(step.id or "").strip() == "direct-wait-execute"
    )


def clear_wait_entry_runtime_state(control: Dict[str, Any]) -> None:
    """等待入口被取消后清理运行态信号；入口合同本身保留用于审计。"""
    control.pop("entry_upgrade", None)
