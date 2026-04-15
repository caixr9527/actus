#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构：执行循环可变状态。"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass(slots=True)
class ExecutionState:
    llm_message: Dict[str, Any] = field(default_factory=dict)
    research_search_ready: bool = False
    research_fetch_completed: bool = False
    research_candidate_urls: List[str] = field(default_factory=list)
    browser_page_type: str = ""
    browser_structured_ready: bool = False
    browser_main_content_ready: bool = False
    browser_cards_ready: bool = False
    browser_link_match_ready: bool = False
    browser_actionables_ready: bool = False
    last_browser_route_url: str = ""
    last_browser_route_title: str = ""
    last_browser_route_selector: str = ""
    last_browser_route_index: Optional[int] = None
    last_tool_fingerprint: str = ""
    same_tool_repeat_count: int = 0
    search_repeat_counter: Dict[str, int] = field(default_factory=dict)
    fetch_repeat_counter: Dict[str, int] = field(default_factory=dict)
    last_browser_observation_fingerprint: str = ""
    browser_no_progress_count: int = 0
    failed_browser_high_level_keys: Set[str] = field(default_factory=set)
    consecutive_failure_count: int = 0
    runtime_recent_action: Dict[str, Any] = field(default_factory=dict)
    # P3-一次性收口：记录最近一次成功工具调用，供“重复调用但已有有效结果”收敛时兜底。
    last_successful_tool_call: Dict[str, Any] = field(default_factory=dict)
    last_successful_tool_fingerprint: str = ""
