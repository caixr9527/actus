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

BROWSER_SNAPSHOT_FUNCTIONS = {
    "browser_view",
    "browser_read_current_page_structured",
    "browser_extract_main_content",
    "browser_extract_cards",
    "browser_get_actionable_elements",
    "browser_find_actionable_elements",
}
BROWSER_ACTION_FUNCTIONS = {
    "browser_click",
    "browser_type",
    "browser_input",
    "browser_scroll_down",
    "browser_scroll_up",
    "browser_go_back",
    "browser_go_forward",
    "browser_navigate",
    "browser_open",
    "browser_restart",
    "browser_find_link_by_text",
    "browser_find_actionable_elements",
    "browser_move_mouse",
    "browser_press_key",
    "browser_select_option",
    "browser_console_exec",
    "browser_console_view",
}


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
    if fact.fact_kind == SandboxFactKind.COMMAND_EXECUTION:
        command_fingerprint = _hash_or_compute(payload.get("command_fingerprint"), payload.get("command_excerpt"))
        return _ok(
            action_key=f"command:{command_fingerprint}",
            subject_key=f"command:{command_fingerprint}",
            action_type="command_execution",
            function_name=function_name,
        )
    if fact.fact_kind == SandboxFactKind.SHELL_OUTPUT:
        session_ref = str(payload.get("session_ref") or fact.subject_ref.subject_key or "").strip()
        output_hash = short_hash(
            "|".join([
                session_ref,
                str(payload.get("process_status") or ""),
                str(payload.get("exit_code") or ""),
                str(payload.get("output_excerpt") or ""),
            ])
        )
        return _ok(
            action_key=f"shell_output:{output_hash}",
            subject_key=f"shell:{session_ref or output_hash}",
            action_type="shell_output",
            function_name=function_name,
        )
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
        mutation_intent_hash = str(payload.get("mutation_intent_hash") or "").strip()
        if not mutation_intent_hash:
            return _skipped(function_name=function_name, reason_code="mutation_intent_hash_missing")
        return _ok(
            action_key=f"file_write:{normalized_path}:{mutation_intent_hash}",
            subject_key=f"file:{normalized_path}",
            action_type="file_write",
            function_name=function_name,
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

    normalized_function = str(function_name or "").strip().lower()
    args = dict(function_args or {})
    if normalized_function in {"search", "web_search", "research_search", "search_web"}:
        query_hash = str(args.get("query_hash") or "").strip()
        if not query_hash:
            query_hash = hash_query(str(args.get("query") or args.get("q") or ""))
        return _ok(
            action_key=f"search:{query_hash}",
            subject_key=f"query:{query_hash}",
            action_type="search",
            function_name=normalized_function,
        )
    if normalized_function in {"fetch", "fetch_page", "web_fetch", "read_url"}:
        url_hash = str(args.get("fetched_url_hash") or args.get("url_hash") or "").strip()
        if not url_hash:
            url_hash = hash_url(str(args.get("url") or ""))
        return _ok(
            action_key=f"fetch:{url_hash}",
            subject_key=f"page:{url_hash}",
            action_type="fetch",
            function_name=normalized_function,
        )
    if normalized_function in {"read_file", "file_read"}:
        normalized_path = _tool_path(args)
        return _file_result("file_read", normalized_path, normalized_function)
    if normalized_function in {"write_file", "file_write", "replace_in_file"}:
        normalized_path = _tool_path(args)
        operation = "write"
        mutation_intent_hash = str(args.get("mutation_intent_hash") or "").strip()
        if not mutation_intent_hash:
            mutation_intent_hash = build_file_mutation_intent_hash(
                path=normalized_path,
                operation=operation,
                content=str(args.get("content") or ""),
                old_str=str(args.get("old_str") or args.get("old_string") or ""),
                new_str=str(args.get("new_str") or args.get("new_string") or ""),
                append=bool(args.get("append")),
                leading_newline=bool(args.get("leading_newline")),
                trailing_newline=bool(args.get("trailing_newline")),
            )
        return _ok(
            action_key=f"file_write:{normalized_path}:{mutation_intent_hash}",
            subject_key=f"file:{normalized_path}",
            action_type="file_write",
            function_name=normalized_function,
        )
    if normalized_function in {"delete_file", "file_delete"}:
        normalized_path = _tool_path(args)
        return _file_result("file_delete", normalized_path, normalized_function)
    if normalized_function in {"list_files", "find_files", "check_file_exists", "file_list", "ls"}:
        normalized_path = _tool_path(args, default_key="dir_path")
        return _ok(
            action_key=f"file_list:{normalized_path}",
            subject_key=f"dir:{normalized_path}",
            action_type="file_list",
            function_name=normalized_function,
        )
    if normalized_function in {"search_in_file", "search_files", "file_search", "grep"}:
        normalized_path = _tool_path(args)
        regex_hash = _hash_or_compute(args.get("regex_hash"), args.get("regex") or args.get("pattern"))
        return _ok(
            action_key=f"file_search:{normalized_path}:{regex_hash}",
            subject_key=f"dir:{normalized_path}",
            action_type="file_search",
            function_name=normalized_function,
        )
    if normalized_function in {"exec_command", "shell_execute"}:
        command_fingerprint = _hash_or_compute(
            args.get("command_fingerprint"),
            args.get("command") or args.get("input"),
        )
        return _ok(
            action_key=f"command:{command_fingerprint}",
            subject_key=f"command:{command_fingerprint}",
            action_type="command_execution",
            function_name=normalized_function,
        )
    if normalized_function in {"read_shell_output", "wait_process", "write_shell_input", "kill_process"}:
        session_ref = str(args.get("session_ref") or args.get("process_id") or args.get("tool_call_id") or "").strip()
        output_hash = str(args.get("output_hash") or "").strip()
        if not output_hash:
            output_hash = short_hash(
                "|".join([
                    session_ref,
                    str(args.get("process_status") or args.get("status") or ""),
                    str(args.get("exit_code") or ""),
                    str(args.get("output_excerpt") or args.get("output") or ""),
                ])
            )
        return _ok(
            action_key=f"shell_output:{output_hash}",
            subject_key=f"shell:{session_ref or output_hash}",
            action_type="shell_output",
            function_name=normalized_function,
        )
    if normalized_function in {"document_context", "read_document"}:
        file_id = str(args.get("file_id") or args.get("document_file_id") or "").strip()
        if not file_id:
            return _skipped(function_name=normalized_function, reason_code="document_file_id_missing")
        content_hash = str(
            args.get("read_content_sha256")
            or args.get("content_hash")
            or args.get("parse_status")
            or "unknown"
        ).strip()
        return _ok(
            action_key=f"document_context:{file_id}:{content_hash}",
            subject_key=f"document:{file_id}",
            action_type="document_context",
            function_name=normalized_function,
        )
    if normalized_function in {"message_ask_user", "ask_user"}:
        interaction_id = str(
            args.get("source_event_id")
            or args.get("tool_event_id")
            or args.get("interaction_id")
            or ""
        ).strip()
        if not interaction_id:
            return _skipped(function_name=normalized_function, reason_code="interaction_id_missing")
        return _ok(
            action_key=f"human_confirm:{interaction_id}",
            subject_key=f"interaction:{interaction_id}",
            action_type="human_confirm",
            function_name=normalized_function,
        )
    if normalized_function in BROWSER_SNAPSHOT_FUNCTIONS:
        url_hash = str(args.get("url_hash") or "").strip()
        if not url_hash and args.get("url"):
            url_hash = hash_url(str(args.get("url") or ""))
        if not url_hash:
            return _skipped(function_name=normalized_function, reason_code="url_hash_missing")
        return _ok(
            action_key=f"browser_snapshot:{url_hash}",
            subject_key=f"browser:{url_hash}",
            action_type="browser_snapshot",
            function_name=normalized_function,
        )
    if normalized_function in BROWSER_ACTION_FUNCTIONS or normalized_function.startswith("browser_"):
        action = normalized_function
        target_hash = short_hash(
            str(
                args.get("target_summary")
                or args.get("selector")
                or args.get("text")
                or args.get("url")
                or action
                or ""
            )
        )
        url_hash = str(args.get("url_hash_after") or args.get("url_hash_before") or args.get("url_hash") or "").strip()
        if not url_hash and args.get("url"):
            url_hash = hash_url(str(args.get("url") or ""))
        return _ok(
            action_key=f"browser_action:{action}:{target_hash}",
            subject_key=f"browser:{url_hash or 'unknown'}",
            action_type="browser_action",
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


def _tool_path(args: Mapping[str, Any], *, default_key: str = "path") -> str:
    value = (
        args.get(default_key)
        or args.get("path")
        or args.get("filepath")
        or args.get("file_path")
        or args.get("dir_path")
        or "."
    )
    return normalize_path(str(value or ""))


def build_file_mutation_intent_hash(
        *,
        path: str,
        operation: str,
        content: str,
        old_str: str,
        new_str: str,
        append: bool,
        leading_newline: bool,
        trailing_newline: bool,
) -> str:
    payload = "\n".join([
        normalize_path(path),
        str(operation or "").strip(),
        str(content or ""),
        str(old_str or ""),
        str(new_str or ""),
        str(bool(append)),
        str(bool(leading_newline)),
        str(bool(trailing_newline)),
    ])
    return short_hash(payload)


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
