#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层 Prompt Context helper。

本模块只负责从 RuntimeContextService 构造上下文包、提取状态更新并拼接到 prompt，
不决定 planner/execute/replan/summary 的节点流转。
"""

import json
from typing import Any, Dict, Optional

from app.domain.models import Step
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.workspace_runtime.context import RuntimeContextService


async def _build_prompt_context_packet_async(
        *,
        stage: str,
        state: PlannerReActLangGraphState,
        runtime_context_service: RuntimeContextService,
        step: Optional[Step] = None,
        task_mode: str = "",
) -> Dict[str, Any]:
    """统一从上下文服务异步构造 Prompt 数据包。"""
    return await runtime_context_service.build_packet_async(
        stage=stage,  # type: ignore[arg-type]
        state=state,
        step=step,
        task_mode=task_mode,
    )

def _extract_prompt_context_state_updates(
        *,
        runtime_context_service: RuntimeContextService,
        context_packet: Dict[str, Any],
) -> Dict[str, Any]:
    """只回写 digest 与 task_mode，避免节点直接操心字段细节。"""
    return runtime_context_service.extract_state_updates(context_packet)

def _append_prompt_context_to_prompt(prompt: str, context_packet: Dict[str, Any]) -> str:
    """将结构化 context packet 追加到 Prompt，避免节点手写上下文拼装。"""
    context_json = json.dumps(context_packet, ensure_ascii=False, indent=2)
    return f"{prompt}\n\n已知上下文:\n```json\n{context_json}\n```"
