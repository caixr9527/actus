#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""节点层 working memory helper。

本模块只负责补齐 working_memory 默认结构，不负责修改节点流转。
"""

from typing import Any, Dict

from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState


def _ensure_working_memory(state: PlannerReActLangGraphState) -> Dict[str, Any]:
    working_memory = dict(state.get("working_memory") or {})
    working_memory.setdefault("goal", "")
    working_memory.setdefault("constraints", [])
    working_memory.setdefault("decisions", [])
    working_memory.setdefault("open_questions", [])
    working_memory.setdefault("user_preferences", {})
    working_memory.setdefault("facts_in_session", [])
    # description-only 主干：human_wait 恢复后的已确认信息统一沉淀到 confirmed_facts。
    working_memory.setdefault("confirmed_facts", {})
    # 轻 summary 与重交付正文分轨：最终长正文只放在 working_memory，不进入 final_message 热路径。
    working_memory.setdefault(
        "final_delivery_payload",
        {
            "text": "",
            "sections": [],
            "source_refs": [],
        },
    )
    return working_memory
