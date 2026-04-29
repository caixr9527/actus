#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 服务包导出。

该包会被纯领域契约间接导入，导出运行引擎时必须惰性加载，避免领域模型初始化
过程中反向导入 runtime engine 造成循环依赖。
"""

from typing import Any

__all__ = [
    "RunEngine",
    "GraphRuntime",
    "DefaultGraphRuntime",
    "GRAPH_STATE_CONTRACT_SCHEMA_VERSION",
    "GraphStateContractMapper",
    "PlannerReActLangGraphState",
    "SkillRuntimePolicy",
    "SkillDefinition",
    "SkillGraphRegistry",
    "SkillGraphRuntime",
]


def __getattr__(name: str) -> Any:
    if name == "RunEngine":
        from .run_engine import RunEngine

        return RunEngine
    if name in {"GraphRuntime", "DefaultGraphRuntime"}:
        from .graph_runtime import DefaultGraphRuntime, GraphRuntime

        return {
            "GraphRuntime": GraphRuntime,
            "DefaultGraphRuntime": DefaultGraphRuntime,
        }[name]
    if name in {
        "GRAPH_STATE_CONTRACT_SCHEMA_VERSION",
        "GraphStateContractMapper",
        "PlannerReActLangGraphState",
    }:
        from .langgraph_state import (
            GRAPH_STATE_CONTRACT_SCHEMA_VERSION,
            GraphStateContractMapper,
            PlannerReActLangGraphState,
        )

        return {
            "GRAPH_STATE_CONTRACT_SCHEMA_VERSION": GRAPH_STATE_CONTRACT_SCHEMA_VERSION,
            "GraphStateContractMapper": GraphStateContractMapper,
            "PlannerReActLangGraphState": PlannerReActLangGraphState,
        }[name]
    if name in {
        "SkillRuntimePolicy",
        "SkillDefinition",
        "SkillGraphRegistry",
        "SkillGraphRuntime",
    }:
        from .skill_graph_registry import (
            SkillDefinition,
            SkillGraphRegistry,
            SkillGraphRuntime,
            SkillRuntimePolicy,
        )

        return {
            "SkillRuntimePolicy": SkillRuntimePolicy,
            "SkillDefinition": SkillDefinition,
            "SkillGraphRegistry": SkillGraphRegistry,
            "SkillGraphRuntime": SkillGraphRuntime,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
