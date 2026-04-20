#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研究链路查询规范化与主查询生成。"""

from __future__ import annotations

import re
from typing import Any

_CJK_CONNECTOR_PATTERN = re.compile(r"[，,、；;|/]+")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")
_URL_PATTERN = re.compile(r"https?://\S+", re.IGNORECASE)

# 只移除明确属于“请求动作”的引导短语，避免误删主题正文。
_QUESTION_PREFIXES = (
    "请帮我查一下",
    "帮我查一下",
    "请帮我查",
    "帮我查",
    "请帮我看看",
    "帮我看看",
    "请帮我搜一下",
    "帮我搜一下",
    "请搜索一下",
    "搜索一下",
    "请查询一下",
    "查询一下",
    "我想了解一下",
    "我想了解",
    "我想知道",
)

# 这些 token 更接近“附加限制条件”，而不是研究主题主体。
_TAIL_CONSTRAINT_PREFIXES = (
    "预算",
    "人均",
    "人数",
    "上限",
    "以内",
    "日期",
    "时间",
    "出发",
    "出发地",
    "往返",
    "酒店",
    "住宿",
    "门票",
    "交通",
    "价格",
    "费用",
    "车程",
    "时长",
)
_TAIL_CONSTRAINT_EXACTS = {
    "一天",
    "两天一夜",
    "三天两夜",
    "四天三夜",
    "五天四夜",
}
_TAIL_CONSTRAINT_TIME_PATTERN = re.compile(r"^\d+\s*(天|晚|小时|个小时)(内|以内)?$")
_TAIL_CONSTRAINT_CN_TIME_PATTERN = re.compile(r"^[一二两三四五六七八九十]+天[一二两三四五六七八九十]+夜$")


def build_research_query(raw_query: Any) -> str:
    """把 research 查询收口为单主题自然语言主查询。

    设计原则：
    - 只移除明显的请求前缀、URL 和尾部附加限制；
    - 不做关键词重写，不把查询改造成另一种语义；
    - 主题主体不足时宁可少删，避免把正文裁坏。
    """
    query = str(raw_query or "").strip()
    if not query:
        return ""
    query = _URL_PATTERN.sub("", query)
    query = _strip_question_prefix(query)
    query = _CJK_CONNECTOR_PATTERN.sub(" ", query)
    query = _MULTI_SPACE_PATTERN.sub(" ", query).strip()
    query = _strip_trailing_constraints(query)
    query = _MULTI_SPACE_PATTERN.sub(" ", query).strip(" ，,、；;。.!?？")
    return query


def _strip_trailing_constraints(query: str) -> str:
    """移除明显的尾部附加限制，优先保留主题主体。"""
    if not query:
        return ""
    tokens = [token for token in query.split(" ") if token]
    if len(tokens) <= 1:
        return query

    for index, token in enumerate(tokens):
        if not _looks_like_tail_constraint(token):
            continue
        prefix = " ".join(tokens[:index]).strip()
        if _is_meaningful_research_topic(prefix):
            return prefix
    return query


def _strip_question_prefix(query: str) -> str:
    """去掉明确的请求动作前缀，不改写主题正文。"""
    normalized = query.strip()
    for prefix in _QUESTION_PREFIXES:
        if not normalized.startswith(prefix):
            continue
        remainder = normalized[len(prefix):].strip()
        if remainder:
            return remainder
    return normalized


def _looks_like_tail_constraint(token: str) -> bool:
    """判断 token 是否更像尾部附加约束，而非主题正文。"""
    normalized = token.strip(" ，,、；;。.!?？")
    if not normalized:
        return False
    if normalized in _TAIL_CONSTRAINT_EXACTS:
        return True
    if _TAIL_CONSTRAINT_TIME_PATTERN.match(normalized):
        return True
    if _TAIL_CONSTRAINT_CN_TIME_PATTERN.match(normalized):
        return True
    return normalized.startswith(_TAIL_CONSTRAINT_PREFIXES)


def _is_meaningful_research_topic(text: str) -> bool:
    """主题主体至少要足够长，避免把查询截成残句。"""
    normalized = text.strip()
    if not normalized:
        return False
    compact = normalized.replace(" ", "")
    return len(compact) >= 8
