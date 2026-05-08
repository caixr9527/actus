#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层 Prompt Context helper。

本模块只负责从 RuntimeContextService 构造上下文包、提取状态更新并拼接到 prompt，
不决定 planner/execute/replan/summary 的节点流转。
"""

import json
from typing import Any, Dict, List, Optional

from app.application.service.document_input_service import DocumentInputService
from app.domain.models import Step
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState
from app.domain.services.workspace_runtime.context import RuntimeContextService

_DOCUMENT_INPUT_SERVICE = DocumentInputService()
_DOCUMENT_CONTEXT_STAGES = {"planner", "execute"}


async def _build_prompt_context_packet_async(
        *,
        stage: str,
        state: PlannerReActLangGraphState,
        runtime_context_service: RuntimeContextService,
        step: Optional[Step] = None,
        task_mode: str = "",
        include_sandbox_profile_for_summary: bool = False,
) -> Dict[str, Any]:
    """统一从上下文服务异步构造 Prompt 数据包。"""
    packet = await runtime_context_service.build_packet_async(
        stage=stage,  # type: ignore[arg-type]
        state=state,
        step=step,
        task_mode=task_mode,
        include_sandbox_profile_for_summary=include_sandbox_profile_for_summary,
    )
    if str(stage) in _DOCUMENT_CONTEXT_STAGES:
        _append_document_context(packet=packet, state=state)
    return packet

def _extract_prompt_context_state_updates(
        *,
        runtime_context_service: RuntimeContextService,
        context_packet: Dict[str, Any],
) -> Dict[str, Any]:
    """只回写 digest 与 task_mode，避免节点直接操心字段细节。"""
    return runtime_context_service.extract_state_updates(context_packet)

def _append_prompt_context_to_prompt(prompt: str, context_packet: Dict[str, Any]) -> str:
    """将结构化 context packet 追加到 Prompt，避免节点手写上下文拼装。"""
    context_json = json.dumps(_build_prompt_safe_context_packet(context_packet), ensure_ascii=False, indent=2)
    return f"{prompt}\n\n已知上下文:\n```json\n{context_json}\n```"


def _build_prompt_safe_context_packet(context_packet: Dict[str, Any]) -> Dict[str, Any]:
    """裁剪 prompt 可见上下文，结构化 evidence context 只给 runtime 消费。"""
    visible_fields = [
        str(field_name)
        for field_name in list(context_packet.get("prompt_visible_fields") or [])
        if str(field_name) != "evidence_context"
    ]
    safe_packet = {
        field_name: context_packet[field_name]
        for field_name in visible_fields
        if field_name in context_packet
    }
    if "prompt_visible_fields" in context_packet:
        safe_packet["prompt_visible_fields"] = visible_fields
    return safe_packet


def extract_document_attachment_paths(input_parts: List[Dict[str, Any]]) -> List[str]:
    """统一从 document input part source 中提取 sandbox 文件路径。"""
    paths: List[str] = []
    for part in list(input_parts or []):
        if not isinstance(part, dict):
            continue
        source = part.get("source")
        if not isinstance(source, dict):
            continue
        sandbox_filepath = str(source.get("sandbox_filepath") or "").strip()
        if sandbox_filepath and sandbox_filepath not in paths:
            paths.append(sandbox_filepath)
    return paths


def _append_document_context(
        *,
        packet: Dict[str, Any],
        state: PlannerReActLangGraphState,
) -> None:
    input_parts = [part for part in list(state.get("input_parts") or []) if isinstance(part, dict)]
    if not input_parts:
        return
    context_result = _DOCUMENT_INPUT_SERVICE.build_prompt_context(parts=input_parts)
    context_payload = context_result.model_dump(mode="json")
    packet["document_context"] = context_payload
    visible_fields = list(packet.get("prompt_visible_fields") or [])
    if "document_context" not in visible_fields:
        visible_fields.append("document_context")
    packet["prompt_visible_fields"] = visible_fields
