#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""SearXNG 搜索质控策略配置。"""

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Set, Tuple


@dataclass(frozen=True, slots=True)
class SearchQualityPolicy:
    query_token_pattern: re.Pattern[str]
    tracking_query_keys: Set[str]
    intermediate_hosts: Set[str]
    search_engine_host_suffixes: Tuple[str, ...]
    search_endpoint_path_prefixes: Tuple[str, ...]
    search_query_keys: Set[str]
    high_quality_host_suffixes: Tuple[str, ...]
    domain_diversity_top_window: int
    domain_cap_in_top_window: int
    token_recall_weight: float
    title_length_bonus: float
    content_length_bonus: float
    zero_token_penalty: float
    short_content_penalty: float
    high_quality_suffix_bonus: float
    engine_bonus_names: Set[str]
    engine_bonus_score: float
    origin_score_max_bonus: float
    origin_score_divisor: float


def _parse_env_set(name: str, default: Set[str]) -> Set[str]:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return set(default)
    return {
        str(item).strip().lower()
        for item in raw.split(",")
        if str(item).strip()
    }


def _parse_env_tuple(name: str, default: Tuple[str, ...]) -> Tuple[str, ...]:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return tuple(default)
    values = [
        str(item).strip().lower()
        for item in raw.split(",")
        if str(item).strip()
    ]
    return tuple(values) if values else tuple(default)


def _parse_env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _parse_env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, "")).strip()
    if not raw:
        return default
    try:
        return float(raw)
    except Exception:
        return default


@lru_cache(maxsize=1)
def get_search_quality_policy() -> SearchQualityPolicy:
    return SearchQualityPolicy(
        query_token_pattern=re.compile(r"[\s,，;；|/]+"),
        tracking_query_keys=_parse_env_set(
            "SEARXNG_SEARCH_TRACKING_QUERY_KEYS",
            {"from", "source", "ref", "referer", "spm", "fr", "ch", "wt", "eqid", "sessionid"},
        ),
        intermediate_hosts=_parse_env_set(
            "SEARXNG_SEARCH_INTERMEDIATE_HOSTS",
            {"mbd.baidu.com"},
        ),
        search_engine_host_suffixes=_parse_env_tuple(
            "SEARXNG_SEARCH_ENGINE_HOST_SUFFIXES",
            ("baidu.com", "bing.com", "sogou.com"),
        ),
        search_endpoint_path_prefixes=_parse_env_tuple(
            "SEARXNG_SEARCH_ENDPOINT_PATH_PREFIXES",
            ("/search", "/s", "/web"),
        ),
        search_query_keys=_parse_env_set(
            "SEARXNG_SEARCH_QUERY_KEYS",
            {"q", "query", "wd", "keyword", "word"},
        ),
        high_quality_host_suffixes=_parse_env_tuple(
            "SEARXNG_SEARCH_HIGH_QUALITY_HOST_SUFFIXES",
            (".gov.cn", ".edu.cn"),
        ),
        domain_diversity_top_window=max(1, _parse_env_int("SEARXNG_SEARCH_DOMAIN_DIVERSITY_TOP_WINDOW", 8)),
        domain_cap_in_top_window=max(1, _parse_env_int("SEARXNG_SEARCH_DOMAIN_CAP_IN_TOP_WINDOW", 2)),
        token_recall_weight=_parse_env_float("SEARXNG_SEARCH_TOKEN_RECALL_WEIGHT", 1.3),
        title_length_bonus=_parse_env_float("SEARXNG_SEARCH_TITLE_LENGTH_BONUS", 0.15),
        content_length_bonus=_parse_env_float("SEARXNG_SEARCH_CONTENT_LENGTH_BONUS", 0.1),
        zero_token_penalty=_parse_env_float("SEARXNG_SEARCH_ZERO_TOKEN_PENALTY", 0.25),
        short_content_penalty=_parse_env_float("SEARXNG_SEARCH_SHORT_CONTENT_PENALTY", 0.08),
        high_quality_suffix_bonus=_parse_env_float("SEARXNG_SEARCH_HIGH_QUALITY_SUFFIX_BONUS", 0.25),
        engine_bonus_names=_parse_env_set("SEARXNG_SEARCH_ENGINE_BONUS_NAMES", {"bing", "baidu"}),
        engine_bonus_score=_parse_env_float("SEARXNG_SEARCH_ENGINE_BONUS_SCORE", 0.05),
        origin_score_max_bonus=_parse_env_float("SEARXNG_SEARCH_ORIGIN_SCORE_MAX_BONUS", 0.2),
        origin_score_divisor=max(1.0, _parse_env_float("SEARXNG_SEARCH_ORIGIN_SCORE_DIVISOR", 50.0)),
    )
