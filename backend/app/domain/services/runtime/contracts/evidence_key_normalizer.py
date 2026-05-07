#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evidence action/subject key 归一纯函数。

统一 normalizer 是重复执行拦截的安全边界：EvidenceFactAssembler 与
EvidenceReusePolicy 必须得到同一组 action_key/subject_key，禁止各工具
自行拼接 key 或从 prompt/raw payload 临时推断。
"""

from __future__ import annotations

import hashlib
import posixpath
import re
from typing import Any, Mapping
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict

from app.domain.models.sandbox_fact import SandboxFactKind, SandboxFactRecord


class EvidenceActionSubjectKeyResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_key: str | None = None
    subject_key: str | None = None
    action_type: str | None = None
    function_name: str | None = None
    normalization_status: str
    reason_code: str | None = None


def build_evidence_action_subject_key_from_fact(fact: SandboxFactRecord) -> EvidenceActionSubjectKeyResult:
    """基于 P1-2 fact 生成 action/subject key，不读取 raw 工具结果。"""

    payload = dict(fact.payload or {})
    function_name = fact.source_ref.function_name
    if fact.fact_kind == SandboxFactKind.SEARCH_RESULT:
        query_hash = _hash_or_compute(payload.get("query_hash"), payload.get("query_excerpt"))
        return _ok(
            action_key=f"search:{query_hash}",
            subject_key=f"query:{query_hash}",
            action_type="search",
            function_name=function_name,
        )
    if fact.fact_kind == SandboxFactKind.FETCHED_PAGE:
        url_hash = _hash_or_compute(payload.get("fetched_url_hash"), payload.get("final_url_origin"))
        return _ok(
            action_key=f"fetch:{url_hash}",
            subject_key=f"page:{url_hash}",
            action_type="fetch",
            function_name=function_name,
        )
    if fact.fact_kind == SandboxFactKind.FILE_READ:
        normalized_path = normalize_path(str(payload.get("path") or ""))
        return _file_result("file_read", normalized_path, function_name)
    if fact.fact_kind in {SandboxFactKind.FILE_WRITE, SandboxFactKind.FILE_SNAPSHOT}:
        normalized_path = normalize_path(str(payload.get("path") or ""))
        after_hash = str(payload.get("after_content_sha256") or "").strip()
        suffix = after_hash or "hash_missing"
        return _ok(
            action_key=f"file_write:{normalized_path}:{suffix}",
            subject_key=f"file:{normalized_path}",
            action_type="file_write",
            function_name=function_name,
            reason_code=None if after_hash else "content_hash_missing",
        )
    if fact.fact_kind == SandboxFactKind.FILE_DELETE:
        normalized_path = normalize_path(str(payload.get("path") or ""))
        return _file_result("file_delete", normalized_path, function_name)
    if fact.fact_kind == SandboxFactKind.FILE_LIST:
        normalized_path = normalize_path(str(payload.get("dir_path") or ""))
        return _ok(
            action_key=f"file_list:{normalized_path}",
            subject_key=f"dir:{normalized_path}",
            action_type="file_list",
            function_name=function_name,
        )
    if fact.fact_kind == SandboxFactKind.FILE_SEARCH:
        normalized_path = normalize_path(str(payload.get("path") or ""))
        regex_hash = str(payload.get("regex_hash") or "").strip() or short_hash(str(payload.get("regex_excerpt") or ""))
        return _ok(
            action_key=f"file_search:{normalized_path}:{regex_hash}",
            subject_key=f"dir:{normalized_path}",
            action_type="file_search",
            function_name=function_name,
        )
    if fact.fact_kind == SandboxFactKind.BROWSER_SNAPSHOT:
        url_hash = str(payload.get("url_hash") or "").strip()
        if not url_hash:
            return _skipped(function_name=function_name, reason_code="url_hash_missing")
        return _ok(
            action_key=f"browser_snapshot:{url_hash}",
            subject_key=f"browser:{url_hash}",
            action_type="browser_snapshot",
            function_name=function_name,
        )
    if fact.fact_kind == SandboxFactKind.BROWSER_ACTION:
        action = str(payload.get("action") or "browser_action").strip()
        target_hash = short_hash(str(payload.get("target_summary") or ""))
        url_hash = str(payload.get("url_hash_after") or payload.get("url_hash_before") or "").strip()
        return _ok(
            action_key=f"browser_action:{action}:{target_hash}",
            subject_key=f"browser:{url_hash or 'unknown'}",
            action_type="browser_action",
            function_name=function_name,
        )
    if fact.fact_kind == SandboxFactKind.DOCUMENT_CONTEXT:
        file_id = str(payload.get("file_id") or "").strip()
        if not file_id:
            return _skipped(function_name=function_name, reason_code="document_file_id_missing")
        content_hash = str(payload.get("read_content_sha256") or payload.get("parse_status") or "unknown").strip()
        return _ok(
            action_key=f"document_context:{file_id}:{content_hash}",
            subject_key=f"document:{file_id}",
            action_type="document_context",
            function_name=function_name,
        )
    if fact.fact_kind == SandboxFactKind.TOOL_FAILURE:
        reason_code = str(payload.get("reason_code") or "tool_failed").strip()
        subject_key = fact.subject_ref.subject_key
        return _ok(
            action_key=f"tool_failure:{function_name or 'unknown'}:{reason_code}:{subject_key}",
            subject_key=subject_key,
            action_type="tool_failure",
            function_name=function_name,
        )
    if fact.fact_kind == SandboxFactKind.HUMAN_INTERACTION:
        interaction_id = fact.source_ref.source_event_id or fact.source_ref.tool_event_id or fact.id
        return _ok(
            action_key=f"human_confirm:{interaction_id}",
            subject_key=f"interaction:{interaction_id}",
            action_type="human_confirm",
            function_name=function_name,
        )
    return _skipped(function_name=function_name, reason_code="unsupported_fact_kind")


def build_evidence_action_subject_key_from_tool_call(
        function_name: str,
        function_args: Mapping[str, Any],
) -> EvidenceActionSubjectKeyResult:
    """工具执行前基于 tool call args 生成同一套 action/subject key。"""

    normalized_function = str(function_name or "").strip()
    args = dict(function_args or {})
    if normalized_function in {"search", "web_search", "research_search"}:
        query_hash = hash_query(str(args.get("query") or args.get("q") or ""))
        return _ok(
            action_key=f"search:{query_hash}",
            subject_key=f"query:{query_hash}",
            action_type="search",
            function_name=normalized_function,
        )
    if normalized_function in {"fetch", "fetch_page", "web_fetch", "read_url"}:
        url_hash = hash_url(str(args.get("url") or ""))
        return _ok(
            action_key=f"fetch:{url_hash}",
            subject_key=f"page:{url_hash}",
            action_type="fetch",
            function_name=normalized_function,
        )
    if normalized_function in {"read_file", "file_read"}:
        normalized_path = normalize_path(str(args.get("path") or ""))
        return _file_result("file_read", normalized_path, normalized_function)
    if normalized_function in {"write_file", "file_write"}:
        normalized_path = normalize_path(str(args.get("path") or ""))
        content_hash = str(args.get("after_content_sha256") or args.get("content_sha256") or "").strip()
        if not content_hash and "content" in args:
            content_hash = short_hash(str(args.get("content") or ""))
        return _ok(
            action_key=f"file_write:{normalized_path}:{content_hash or 'hash_missing'}",
            subject_key=f"file:{normalized_path}",
            action_type="file_write",
            function_name=normalized_function,
            reason_code=None if content_hash else "content_hash_missing",
        )
    if normalized_function in {"delete_file", "file_delete"}:
        normalized_path = normalize_path(str(args.get("path") or ""))
        return _file_result("file_delete", normalized_path, normalized_function)
    if normalized_function in {"list_files", "file_list", "ls"}:
        normalized_path = normalize_path(str(args.get("path") or args.get("dir_path") or ""))
        return _ok(
            action_key=f"file_list:{normalized_path}",
            subject_key=f"dir:{normalized_path}",
            action_type="file_list",
            function_name=normalized_function,
        )
    if normalized_function in {"search_files", "file_search", "grep"}:
        normalized_path = normalize_path(str(args.get("path") or "."))
        regex_hash = short_hash(str(args.get("regex") or args.get("pattern") or ""))
        return _ok(
            action_key=f"file_search:{normalized_path}:{regex_hash}",
            subject_key=f"dir:{normalized_path}",
            action_type="file_search",
            function_name=normalized_function,
        )
    return _skipped(function_name=normalized_function, reason_code="unsupported_tool_call")


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "").strip()).casefold()


def hash_query(query: str) -> str:
    return short_hash(normalize_query(query))


def normalize_url(url: str) -> str:
    parsed = urlsplit(str(url or "").strip())
    scheme = (parsed.scheme or "https").lower()
    host = parsed.netloc.lower()
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    normalized_path = posixpath.normpath(parsed.path or "/")
    if parsed.path.endswith("/") and not normalized_path.endswith("/"):
        normalized_path += "/"
    return urlunsplit((scheme, host, normalized_path, query, ""))


def hash_url(url: str) -> str:
    return short_hash(normalize_url(url))


def normalize_path(path: str) -> str:
    raw_path = str(path or "").strip() or "."
    normalized = posixpath.normpath(raw_path.replace("\\", "/"))
    if raw_path.startswith("/") and not normalized.startswith("/"):
        normalized = f"/{normalized}"
    return normalized


def short_hash(value: str) -> str:
    digest = hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _hash_or_compute(existing_hash: Any, fallback_value: Any) -> str:
    normalized = str(existing_hash or "").strip()
    if normalized:
        return normalized
    return short_hash(str(fallback_value or ""))


def _ok(
        *,
        action_key: str,
        subject_key: str,
        action_type: str,
        function_name: str | None,
        reason_code: str | None = None,
) -> EvidenceActionSubjectKeyResult:
    return EvidenceActionSubjectKeyResult(
        action_key=action_key,
        subject_key=subject_key,
        action_type=action_type,
        function_name=function_name,
        normalization_status="normalized",
        reason_code=reason_code,
    )


def _file_result(action_type: str, normalized_path: str, function_name: str | None) -> EvidenceActionSubjectKeyResult:
    if not normalized_path:
        return _skipped(function_name=function_name, reason_code="path_missing")
    return _ok(
        action_key=f"{action_type}:{normalized_path}",
        subject_key=f"file:{normalized_path}",
        action_type=action_type,
        function_name=function_name,
    )


def _skipped(*, function_name: str | None, reason_code: str) -> EvidenceActionSubjectKeyResult:
    return EvidenceActionSubjectKeyResult(
        action_key=None,
        subject_key=None,
        action_type=None,
        function_name=function_name,
        normalization_status="skipped",
        reason_code=reason_code,
    )
