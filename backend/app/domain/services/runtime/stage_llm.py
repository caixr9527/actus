#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""运行时阶段模型约束。"""

from typing import Dict, Mapping

from app.domain.external import LLM

REQUIRED_STAGE_LLM_NAMES: tuple[str, ...] = (
    "router",
    "planner",
    "executor",
    "replan",
    "summary",
)


def build_uniform_stage_llms(llm: LLM) -> Dict[str, LLM]:
    """用同一个 LLM 填满所有运行时阶段。"""
    return {
        stage_name: llm
        for stage_name in REQUIRED_STAGE_LLM_NAMES
    }


def ensure_required_stage_llms(stage_llms: Mapping[str, LLM]) -> Dict[str, LLM]:
    """校验并规整运行时阶段模型映射。"""
    normalized_stage_llms = {
        str(stage_name).strip(): llm
        for stage_name, llm in dict(stage_llms or {}).items()
        if str(stage_name).strip() and llm is not None
    }
    missing_stage_names = [
        stage_name
        for stage_name in REQUIRED_STAGE_LLM_NAMES
        if stage_name not in normalized_stage_llms
    ]
    if missing_stage_names:
        raise ValueError(
            "LangGraph Runtime 缺少必要阶段模型配置: "
            + ", ".join(missing_stage_names)
        )
    return {
        stage_name: normalized_stage_llms[stage_name]
        for stage_name in REQUIRED_STAGE_LLM_NAMES
    }
