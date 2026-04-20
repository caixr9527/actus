#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构：执行循环可变状态。"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass(slots=True)
class ExecutionState:
    """单步工具循环的运行时可变状态。

    业务含义：
    - 这是 `execute_step_with_prompt()` 单步执行期间的唯一可变状态容器；
    - `ConstraintEngine` 只读消费它，`effects / convergence` 负责在每轮后写回；
    - 字段按研究链路、浏览器链路、重复调用治理、最近动作沉淀四类组织。
    """

    # 最近一轮 LLM 响应原文，供 tool-call message 回放与 no-tool 分支收尾复用。
    llm_message: Dict[str, Any] = field(default_factory=dict)
    # 研究链路进展：标记“是否已有候选链接”“是否已完成正文抓取”。
    research_search_ready: bool = False
    research_fetch_completed: bool = False
    # 研究链路素材池：搜索产生的候选 URL / 查询历史 / 已抓取 URL 与域名覆盖情况。
    research_candidate_urls: List[str] = field(default_factory=list)
    research_search_evidence_items: List[Dict[str, Any]] = field(default_factory=list)
    research_query_history: List[str] = field(default_factory=list)
    research_fetched_urls: List[str] = field(default_factory=list)
    research_candidate_domains: List[str] = field(default_factory=list)
    research_fetched_domains: List[str] = field(default_factory=list)
    # 记录本步骤内抓取失败的 URL 去重键，避免反复命中同一失效页面。
    research_failed_fetch_url_keys: Set[str] = field(default_factory=set)
    # 记录本步骤内已执行过 explicit URL 改写的去重键，避免 search->fetch 对同一 explicit URL 反复改写。
    research_explicit_rewrite_url_keys: Set[str] = field(default_factory=set)
    # 研究链路 fetch_page 连续失败计数，仅用于 research 重写/guard 恢复策略。
    consecutive_fetch_failure_count: int = 0
    # 跨域覆盖约束使用的重复阻断次数，连续命中后会升级为 hard block。
    research_cross_domain_repeat_blocks: int = 0
    # 研究链路核心指标：成功抓取数、累计搜索结果数、综合覆盖评分。
    research_fetch_success_count: int = 0
    research_total_search_results: int = 0
    research_coverage_score: float = 0.0
    # 研究诊断辅助字段：区分“搜索已执行”和“抓取结果是否真的有正文价值”。
    search_invocation_count: int = 0
    research_snippet_sufficient: bool = False
    research_recommended_fetch_urls: List[str] = field(default_factory=list)
    last_search_evidence_quality: Dict[str, Any] = field(default_factory=dict)
    last_fetch_quality: Dict[str, Any] = field(default_factory=dict)
    # 浏览器链路进展：记录当前页面类型以及结构化/正文/cards/actionables 等能力是否已就绪。
    browser_page_type: str = ""
    browser_structured_ready: bool = False
    browser_main_content_ready: bool = False
    browser_cards_ready: bool = False
    browser_link_match_ready: bool = False
    browser_actionables_ready: bool = False
    # 最近一次浏览器路由定位到的核心页面信息。
    last_browser_route_url: str = ""
    last_browser_route_title: str = ""
    last_browser_route_selector: str = ""
    last_browser_route_index: Optional[int] = None
    # 重复调用治理：最近一次工具指纹、连续相同调用计数、按 search/fetch 聚合的去重计数。
    last_tool_fingerprint: str = ""
    same_tool_repeat_count: int = 0
    search_repeat_counter: Dict[str, int] = field(default_factory=dict)
    fetch_repeat_counter: Dict[str, int] = field(default_factory=dict)
    # 浏览器无进展治理：记录最近一次观察指纹与无进展次数，用于提前收敛。
    last_browser_observation_fingerprint: str = ""
    browser_no_progress_count: int = 0
    failed_browser_high_level_keys: Set[str] = field(default_factory=set)
    # 通用失败统计：记录连续失败次数，辅助失败上限收敛。
    consecutive_failure_count: int = 0
    # 最近动作沉淀：供下一轮模型与最终收敛链路感知“上一轮发生了什么”。
    runtime_recent_action: Dict[str, Any] = field(default_factory=dict)
    # P3-一次性收口：记录最近一次成功工具调用，供“重复调用但已有有效结果”收敛时兜底。
    last_successful_tool_call: Dict[str, Any] = field(default_factory=dict)
    last_successful_tool_fingerprint: str = ""
