#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sandbox Fact Ledger 应用服务。

PR2 只接收领域输入模型并写入 fact；工具事件适配进入 PR3 projector。
"""

from __future__ import annotations

import hashlib
import logging
import re
import uuid
from datetime import datetime
from typing import Any, Literal, Sequence
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field

from app.domain.models.sandbox_fact import (
    BrowserActionPayload,
    BrowserSnapshotPayload,
    CommandExecutionPayload,
    DocumentContextPayload,
    FetchedPagePayload,
    FileListPayload,
    FileMutationPayload,
    FileReadPayload,
    FileSearchPayload,
    HumanInteractionPayload,
    ProfileReferencePayload,
    SandboxFactKind,
    SandboxFactRecord,
    SandboxFactScope,
    SandboxFactSourceRef,
    SandboxFactSourceType,
    SandboxFactSubjectRef,
    SandboxFactVisibility,
    SearchResultPayload,
    ShellOutputPayload,
    ToolFailurePayload,
    build_sandbox_fact_idempotency_key,
    build_sandbox_fact_payload_hash,
    classify_sandbox_fact_data,
    validate_sandbox_fact_payload,
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.runtime.contracts.sandbox_fact_ports import (
    SandboxFactProjectionContext,
)

logger = logging.getLogger(__name__)


MAX_SUMMARY_CHARS = 500
REDACTED = "[REDACTED]"

_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(\b(?:access|refresh|id)?_?token\b\s*[:=]\s*)['\"]?[^'\"\s,}]{8,}"),
    re.compile(r"(?i)(\b(?:api[_-]?key|secret[_-]?key)\b\s*[:=]\s*)['\"]?[^'\"\s,}]{8,}"),
    re.compile(r"(?i)(\bpassword\b\s*[:=]\s*)['\"]?[^'\"\s,}]{4,}"),
    re.compile(r"(?i)(\bcookie\b\s*[:=]\s*)['\"]?[^'\"\n,}]{8,}"),
    re.compile(r"(?i)([\"'](?:access|refresh|id)?_?token[\"']\s*:\s*)[\"'][^\"']{8,}[\"']"),
    re.compile(r"(?i)([\"'](?:api[_-]?key|secret[_-]?key)[\"']\s*:\s*)[\"'][^\"']{8,}[\"']"),
    re.compile(r"(?i)([\"']password[\"']\s*:\s*)[\"'][^\"']{4,}[\"']"),
    re.compile(r"(?i)([\"']cookie[\"']\s*:\s*)[\"'][^\"']{8,}[\"']"),
    re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/-]{8,}"),
)


class SandboxFactLedgerError(RuntimeError):
    """Sandbox Fact Ledger 应用服务错误。"""


class SandboxFactScopeError(SandboxFactLedgerError):
    """fact scope 与 access scope 不一致。"""


class SandboxFactProfileMismatchError(SandboxFactLedgerError):
    """profile ref 与 context sandbox 绑定不一致。"""


class _FactInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fact_scope: SandboxFactScope = SandboxFactScope.STEP
    run_id: str | None = None
    step_id: str | None = None
    summary: str = ""
    visibility: SandboxFactVisibility = SandboxFactVisibility.EVIDENCE_INPUT
    source_type: SandboxFactSourceType = SandboxFactSourceType.SYSTEM_PROJECTION
    tool_event_id: str | None = None
    tool_call_id: str | None = None
    function_name: str | None = None
    artifact_id: str | None = None
    file_id: str | None = None
    document_source_id: str | None = None
    missing_fields: list[str] | None = None
    reason_code: str | None = None


class CommandExecutionFactInput(_FactInput):
    fact_kind: Literal[SandboxFactKind.COMMAND_EXECUTION] = SandboxFactKind.COMMAND_EXECUTION
    command: str
    cwd: str
    exit_code: int | None
    duration_ms: int | None
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    changed_paths: list[str] = Field(default_factory=list)
    timeout: bool = False


class ShellOutputFactInput(_FactInput):
    fact_kind: Literal[SandboxFactKind.SHELL_OUTPUT] = SandboxFactKind.SHELL_OUTPUT
    session_ref: str
    output: str
    output_truncated: bool = False
    console_record_count: int
    process_status: str
    exit_code: int | None = None
    duration_ms: int | None = None


class FileReadFactInput(_FactInput):
    fact_kind: Literal[SandboxFactKind.FILE_READ] = SandboxFactKind.FILE_READ
    path: str
    exists: bool
    size: int | None
    content_sha256: str | None
    content_sha256_kind: Literal["full_file_sha256", "read_content_sha256", "unknown"]
    mime_type: str
    line_range: dict[str, Any] | None = None
    content: str = ""
    is_truncated: bool = False
    mtime: datetime | None = None
    missing_fields: list[str] | None = None
    reason_code: str | None = None


class FileMutationFactInput(_FactInput):
    fact_kind: Literal[
        SandboxFactKind.FILE_WRITE,
        SandboxFactKind.FILE_DELETE,
        SandboxFactKind.FILE_SNAPSHOT,
    ]
    path: str
    operation: Literal["write", "delete", "snapshot"]
    mutation_intent_hash: str
    exists: bool
    before_content_sha256: str | None = None
    after_content_sha256: str | None = None
    content_sha256_kind: Literal["full_file_sha256", "read_content_sha256", "unknown"]
    size_after: int | None = None
    changed: bool
    missing_fields: list[str] | None = None
    reason_code: str | None = None


class FileSearchMatchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    line_number: int | None = None
    excerpt: str


class FileSearchFactInput(_FactInput):
    fact_kind: Literal[SandboxFactKind.FILE_SEARCH] = SandboxFactKind.FILE_SEARCH
    path: str
    regex: str
    match_count: int
    matches: list[FileSearchMatchInput] = Field(default_factory=list)
    is_truncated: bool = False


class FileListEntryFactInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: str = "file"
    size: int | None = None
    mtime: datetime | None = None


class FileListFactInput(_FactInput):
    fact_kind: Literal[SandboxFactKind.FILE_LIST] = SandboxFactKind.FILE_LIST
    dir_path: str
    entry_count: int
    entries: list[FileListEntryFactInput] = Field(default_factory=list)
    is_truncated: bool = False
    missing_fields: list[str] | None = None
    reason_code: str | None = None


class BrowserSnapshotFactInput(_FactInput):
    fact_kind: Literal[SandboxFactKind.BROWSER_SNAPSHOT] = SandboxFactKind.BROWSER_SNAPSHOT
    url: str
    title: str = ""
    screenshot_artifact_id: str | None = None
    screenshot_artifact_path: str | None = None
    screenshot_file_id: str | None = None
    screenshot_filename: str | None = None
    screenshot_filepath: str | None = None
    screenshot_key: str | None = None
    screenshot_mime_type: str | None = None
    screenshot_size: int | None = None
    structured_summary: str = ""
    actionable_element_count: int = 0
    degrade_reason: str | None = None


class BrowserActionFactInput(_FactInput):
    fact_kind: Literal[SandboxFactKind.BROWSER_ACTION] = SandboxFactKind.BROWSER_ACTION
    action: str
    target_summary: str
    url_before: str | None = None
    url_after: str | None = None
    success: bool
    degrade_reason: str | None = None


class SearchResultItemInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str
    url: str
    snippet: str = ""


class SearchResultFactInput(_FactInput):
    fact_kind: Literal[SandboxFactKind.SEARCH_RESULT] = SandboxFactKind.SEARCH_RESULT
    query: str
    result_count: int
    top_results: list[SearchResultItemInput] = Field(default_factory=list)
    is_truncated: bool = False


class FetchedPageFactInput(_FactInput):
    fact_kind: Literal[SandboxFactKind.FETCHED_PAGE] = SandboxFactKind.FETCHED_PAGE
    fetched_url: str
    final_url: str
    status_code: int | None
    content_type: str | None
    title: str = ""
    content: str = ""
    is_truncated: bool = False


class DocumentContextFactInput(_FactInput):
    fact_kind: Literal[SandboxFactKind.DOCUMENT_CONTEXT] = SandboxFactKind.DOCUMENT_CONTEXT
    source_type: SandboxFactSourceType = SandboxFactSourceType.DOCUMENT_INPUT
    file_id: str
    filename_extension: str
    mime_type: str
    parse_status: str
    reason_code: str | None
    full_file_sha256: str | None = None
    read_content_sha256: str | None = None
    is_truncated: bool = False
    excerpt_char_count: int


class ProfileReferenceFactInput(_FactInput):
    fact_kind: Literal[SandboxFactKind.PROFILE_REFERENCE] = SandboxFactKind.PROFILE_REFERENCE
    source_type: SandboxFactSourceType = SandboxFactSourceType.PROFILE
    reason_code: str | None = None


class ToolFailureFactInput(_FactInput):
    fact_kind: Literal[SandboxFactKind.TOOL_FAILURE] = SandboxFactKind.TOOL_FAILURE
    function_name: str
    reason_code: str
    message: str
    retry_count: int = 0
    timeout: bool = False
    diagnostic_type: str


class HumanInteractionFactInput(_FactInput):
    fact_kind: Literal[SandboxFactKind.HUMAN_INTERACTION] = SandboxFactKind.HUMAN_INTERACTION
    source_type: SandboxFactSourceType = SandboxFactSourceType.USER_CONFIRMATION
    interaction_type: str
    message: str
    confirmed: bool | None = None
    reason_code: str | None = None


SandboxFactInput = (
    CommandExecutionFactInput
    | ShellOutputFactInput
    | FileReadFactInput
    | FileMutationFactInput
    | FileListFactInput
    | FileSearchFactInput
    | BrowserSnapshotFactInput
    | BrowserActionFactInput
    | SearchResultFactInput
    | FetchedPageFactInput
    | DocumentContextFactInput
    | ProfileReferenceFactInput
    | ToolFailureFactInput
    | HumanInteractionFactInput
)


def sanitize_fact_text(text: str) -> tuple[str, int]:
    """对 fact 文本字段做保存前 secret 脱敏。"""

    redacted = str(text or "")
    redaction_count = 0
    for pattern in _SECRET_PATTERNS:
        redacted, count = pattern.subn(REDACTED, redacted)
        redaction_count += count
    return redacted, redaction_count


def redact_text(text: str) -> tuple[str, int]:
    return sanitize_fact_text(text)


def hash_text(value: str) -> str:
    digest = hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def hash_url(url: str | None) -> str | None:
    normalized = str(url or "").strip()
    if not normalized:
        return None
    split = urlsplit(normalized)
    stable = f"{split.scheme.lower()}://{split.netloc.lower()}{split.path}"
    return hash_text(stable)


def url_origin(url: str | None) -> str:
    normalized = str(url or "").strip()
    if not normalized:
        return ""
    split = urlsplit(normalized)
    if not split.scheme or not split.netloc:
        return ""
    return f"{split.scheme.lower()}://{split.netloc.lower()}"


def _sanitize_payload_text(text: str) -> tuple[str, int]:
    redacted, redaction_count = sanitize_fact_text(text)
    return redacted, redaction_count


def sanitize_summary(summary: str) -> str:
    sanitized, _ = _sanitize_payload_text(summary)
    return sanitized[:MAX_SUMMARY_CHARS]


def _merge_missing_fields(existing: list[str] | None, extra: list[str]) -> list[str] | None:
    values = [str(item) for item in (existing or []) if str(item or "").strip()]
    values.extend(str(item) for item in extra if str(item or "").strip())
    return values or None


def normalize_fact_input(fact_input: SandboxFactInput) -> tuple[dict[str, Any], SandboxFactSubjectRef, str]:
    """将 PR2 领域输入归一为 fact payload、subject ref 和默认摘要。"""

    if isinstance(fact_input, CommandExecutionFactInput):
        stdout, _ = _sanitize_payload_text(fact_input.stdout)
        stderr, _ = _sanitize_payload_text(fact_input.stderr)
        command_excerpt, _ = _sanitize_payload_text(fact_input.command)
        payload = CommandExecutionPayload(
            command_fingerprint=hash_text(command_excerpt),
            cwd=fact_input.cwd,
            exit_code=fact_input.exit_code,
            duration_ms=fact_input.duration_ms,
            stdout_excerpt=stdout,
            stderr_excerpt=stderr,
            stdout_truncated=fact_input.stdout_truncated,
            stderr_truncated=fact_input.stderr_truncated,
            changed_paths=fact_input.changed_paths,
            timeout=fact_input.timeout,
            command_excerpt=command_excerpt,
            missing_fields=fact_input.missing_fields,
            reason_code=fact_input.reason_code,
        )
        subject = SandboxFactSubjectRef(subject_type="command", subject_key=payload.command_fingerprint)
        return payload.model_dump(mode="json"), subject, sanitize_summary(fact_input.summary or "Command execution fact")

    if isinstance(fact_input, ShellOutputFactInput):
        output, _ = _sanitize_payload_text(fact_input.output)
        missing_fields = _merge_missing_fields(fact_input.missing_fields, ["exit_code"] if fact_input.exit_code is None else [])
        payload = ShellOutputPayload(
            session_ref=fact_input.session_ref,
            output_excerpt=output,
            output_truncated=fact_input.output_truncated,
            console_record_count=fact_input.console_record_count,
            process_status=fact_input.process_status,
            exit_code=fact_input.exit_code,
            duration_ms=fact_input.duration_ms,
            missing_fields=missing_fields,
            reason_code=fact_input.reason_code or ("exit_code_unavailable" if fact_input.exit_code is None else None),
        )
        subject = SandboxFactSubjectRef(subject_type="command", subject_key=f"shell:{fact_input.session_ref}")
        return payload.model_dump(mode="json"), subject, sanitize_summary(fact_input.summary or "Shell output fact")

    if isinstance(fact_input, FileReadFactInput):
        excerpt, _ = _sanitize_payload_text(fact_input.content)
        payload = FileReadPayload(
            path=fact_input.path,
            exists=fact_input.exists,
            size=fact_input.size,
            content_sha256=fact_input.content_sha256,
            content_sha256_kind=fact_input.content_sha256_kind,
            mime_type=fact_input.mime_type,
            line_range=fact_input.line_range,
            excerpt=excerpt,
            is_truncated=fact_input.is_truncated,
            mtime=fact_input.mtime,
            missing_fields=fact_input.missing_fields,
            reason_code=fact_input.reason_code,
        )
        subject = SandboxFactSubjectRef(subject_type="file", subject_key=fact_input.path, path=fact_input.path)
        return payload.model_dump(mode="json"), subject, sanitize_summary(fact_input.summary or "File read fact")

    if isinstance(fact_input, FileMutationFactInput):
        payload = FileMutationPayload(
            path=fact_input.path,
            operation=fact_input.operation,
            mutation_intent_hash=fact_input.mutation_intent_hash,
            exists=fact_input.exists,
            before_content_sha256=fact_input.before_content_sha256,
            after_content_sha256=fact_input.after_content_sha256,
            content_sha256_kind=fact_input.content_sha256_kind,
            size_after=fact_input.size_after,
            changed=fact_input.changed,
            missing_fields=fact_input.missing_fields,
            reason_code=fact_input.reason_code,
        )
        subject = SandboxFactSubjectRef(subject_type="file", subject_key=fact_input.path, path=fact_input.path)
        return payload.model_dump(mode="json"), subject, sanitize_summary(fact_input.summary or "File mutation fact")

    if isinstance(fact_input, FileSearchFactInput):
        matches = []
        for item in fact_input.matches:
            excerpt, _ = _sanitize_payload_text(item.excerpt)
            matches.append({"path": item.path, "line_number": item.line_number, "excerpt": excerpt})
        payload = FileSearchPayload(
            path=fact_input.path,
            regex_hash=hash_text(fact_input.regex),
            match_count=fact_input.match_count,
            matches=matches,
            is_truncated=fact_input.is_truncated,
            regex_excerpt=_sanitize_payload_text(fact_input.regex)[0],
            missing_fields=fact_input.missing_fields,
            reason_code=fact_input.reason_code,
        )
        subject = SandboxFactSubjectRef(subject_type="file", subject_key=f"{fact_input.path}:{payload.regex_hash}", path=fact_input.path)
        return payload.model_dump(mode="json"), subject, sanitize_summary(fact_input.summary or "File search fact")

    if isinstance(fact_input, FileListFactInput):
        payload = FileListPayload(
            dir_path=fact_input.dir_path,
            entry_count=fact_input.entry_count,
            entries=[item.model_dump(mode="json") for item in fact_input.entries],
            is_truncated=fact_input.is_truncated,
            missing_fields=fact_input.missing_fields,
            reason_code=fact_input.reason_code,
        )
        subject = SandboxFactSubjectRef(subject_type="file", subject_key=fact_input.dir_path, path=fact_input.dir_path)
        return payload.model_dump(mode="json"), subject, sanitize_summary(fact_input.summary or "File list fact")

    if isinstance(fact_input, BrowserSnapshotFactInput):
        url_hash = hash_url(fact_input.url) or hash_text("")
        payload = BrowserSnapshotPayload(
            url_hash=url_hash,
            url_origin=url_origin(fact_input.url),
            title=_sanitize_payload_text(fact_input.title)[0],
            screenshot_artifact_id=fact_input.screenshot_artifact_id,
            screenshot_artifact_path=fact_input.screenshot_artifact_path,
            structured_summary=_sanitize_payload_text(fact_input.structured_summary)[0],
            actionable_element_count=fact_input.actionable_element_count,
            degrade_reason=fact_input.degrade_reason,
            screenshot_file_id=fact_input.screenshot_file_id,
            screenshot_filename=fact_input.screenshot_filename,
            screenshot_filepath=fact_input.screenshot_filepath,
            screenshot_key=fact_input.screenshot_key,
            screenshot_mime_type=fact_input.screenshot_mime_type,
            screenshot_size=fact_input.screenshot_size,
            missing_fields=fact_input.missing_fields,
            reason_code=fact_input.reason_code,
        )
        subject = SandboxFactSubjectRef(subject_type="browser", subject_key=url_hash, url_hash=url_hash)
        return payload.model_dump(mode="json"), subject, sanitize_summary(fact_input.summary or "Browser snapshot fact")

    if isinstance(fact_input, BrowserActionFactInput):
        payload = BrowserActionPayload(
            action=fact_input.action,
            target_summary=_sanitize_payload_text(fact_input.target_summary)[0],
            url_hash_before=hash_url(fact_input.url_before),
            url_hash_after=hash_url(fact_input.url_after),
            success=fact_input.success,
            degrade_reason=fact_input.degrade_reason,
            missing_fields=fact_input.missing_fields,
            reason_code=fact_input.reason_code,
        )
        subject = SandboxFactSubjectRef(subject_type="browser", subject_key=f"{fact_input.action}:{payload.url_hash_after or payload.url_hash_before or ''}")
        return payload.model_dump(mode="json"), subject, sanitize_summary(fact_input.summary or "Browser action fact")

    if isinstance(fact_input, SearchResultFactInput):
        top_results = []
        for item in fact_input.top_results:
            top_results.append(
                {
                    "title": _sanitize_payload_text(item.title)[0],
                    "origin": url_origin(item.url),
                    "url_hash": hash_url(item.url) or hash_text(item.url),
                    "snippet_excerpt": _sanitize_payload_text(item.snippet)[0],
                }
            )
        query_excerpt = _sanitize_payload_text(fact_input.query)[0]
        payload = SearchResultPayload(
            query_hash=hash_text(query_excerpt),
            query_excerpt=query_excerpt,
            result_count=fact_input.result_count,
            top_results=top_results,
            is_truncated=fact_input.is_truncated,
            missing_fields=fact_input.missing_fields,
            reason_code=fact_input.reason_code,
        )
        subject = SandboxFactSubjectRef(subject_type="search", subject_key=payload.query_hash)
        return payload.model_dump(mode="json"), subject, sanitize_summary(fact_input.summary or "Search result fact")

    if isinstance(fact_input, FetchedPageFactInput):
        excerpt, _ = _sanitize_payload_text(fact_input.content)
        fetched_url_hash = hash_url(fact_input.fetched_url) or hash_text(fact_input.fetched_url)
        payload = FetchedPagePayload(
            fetched_url_hash=fetched_url_hash,
            final_url_origin=url_origin(fact_input.final_url),
            status_code=fact_input.status_code,
            content_type=fact_input.content_type,
            title=_sanitize_payload_text(fact_input.title)[0],
            excerpt=excerpt,
            is_truncated=fact_input.is_truncated,
            missing_fields=fact_input.missing_fields,
            reason_code=fact_input.reason_code,
        )
        subject = SandboxFactSubjectRef(subject_type="page", subject_key=fetched_url_hash, url_hash=fetched_url_hash)
        return payload.model_dump(mode="json"), subject, sanitize_summary(fact_input.summary or "Fetched page fact")

    if isinstance(fact_input, DocumentContextFactInput):
        payload = DocumentContextPayload(**fact_input.model_dump(include={
            "file_id",
            "filename_extension",
            "mime_type",
            "parse_status",
            "reason_code",
            "full_file_sha256",
            "read_content_sha256",
            "is_truncated",
            "excerpt_char_count",
        }))
        subject = SandboxFactSubjectRef(subject_type="document", subject_key=fact_input.file_id)
        return payload.model_dump(mode="json"), subject, sanitize_summary(fact_input.summary or "Document context fact")

    if isinstance(fact_input, ProfileReferenceFactInput):
        raise ValueError("ProfileReferenceFactInput 需要由 service 结合 context.profile_ref 归一")

    if isinstance(fact_input, ToolFailureFactInput):
        message, _ = _sanitize_payload_text(fact_input.message)
        payload = ToolFailurePayload(
            function_name=fact_input.function_name,
            reason_code=fact_input.reason_code,
            message_excerpt=message,
            retry_count=fact_input.retry_count,
            timeout=fact_input.timeout,
            diagnostic_type=fact_input.diagnostic_type,
        )
        subject = SandboxFactSubjectRef(subject_type="command", subject_key=f"tool_failure:{fact_input.function_name}:{fact_input.reason_code}")
        return payload.model_dump(mode="json"), subject, sanitize_summary(fact_input.summary or "Tool failure fact")

    if isinstance(fact_input, HumanInteractionFactInput):
        message, _ = _sanitize_payload_text(fact_input.message)
        payload = HumanInteractionPayload(
            interaction_type=fact_input.interaction_type,
            message_excerpt=message,
            confirmed=fact_input.confirmed,
            reason_code=fact_input.reason_code,
        )
        subject = SandboxFactSubjectRef(subject_type="interaction", subject_key=f"{fact_input.interaction_type}:{hash_text(message)}")
        return payload.model_dump(mode="json"), subject, sanitize_summary(fact_input.summary or "Human interaction fact")

    raise ValueError(f"不支持的 sandbox fact 输入类型: {type(fact_input).__name__}")


class SandboxFactLedgerService:
    """Sandbox Fact 写入与查询的应用层唯一入口。"""

    def __init__(self, *, uow_factory) -> None:
        self._uow_factory = uow_factory

    async def record_fact(
            self,
            *,
            context: SandboxFactProjectionContext,
            fact_input: SandboxFactInput,
    ) -> SandboxFactRecord:
        self._validate_context(context)
        normalized_run_id, normalized_step_id = self._resolve_scope(context=context, fact_input=fact_input)
        source_event_id = context.source_event_id
        source_ref = SandboxFactSourceRef(
            source_type=fact_input.source_type,
            source_event_id=source_event_id,
            source_event_status="available" if source_event_id else "missing",
            tool_event_id=fact_input.tool_event_id,
            tool_call_id=fact_input.tool_call_id,
            function_name=fact_input.function_name,
            artifact_id=fact_input.artifact_id,
            file_id=fact_input.file_id,
            document_source_id=fact_input.document_source_id,
        )
        payload, subject_ref, summary = self._normalize_payload_with_context(
            context=context,
            fact_input=fact_input,
        )
        normalized_payload = validate_sandbox_fact_payload(
            fact_kind=fact_input.fact_kind,
            payload=payload,
        ).model_dump(mode="json")
        payload_hash = build_sandbox_fact_payload_hash(normalized_payload)
        idempotency_key = build_sandbox_fact_idempotency_key(
            user_id=context.scope.user_id,
            session_id=str(context.scope.session_id),
            workspace_id=str(context.scope.workspace_id),
            fact_scope=fact_input.fact_scope,
            run_id=normalized_run_id,
            step_id=normalized_step_id,
            fact_kind=fact_input.fact_kind,
            source_event_id=source_ref.source_event_id,
            tool_call_id=source_ref.tool_call_id,
            subject_key=subject_ref.subject_key,
            payload_hash=payload_hash,
        )
        classification = classify_sandbox_fact_data(
            fact_kind=fact_input.fact_kind,
            source_type=fact_input.source_type,
        )
        fact = SandboxFactRecord(
            id=str(uuid.uuid4()),
            user_id=context.scope.user_id,
            session_id=str(context.scope.session_id),
            workspace_id=str(context.scope.workspace_id),
            fact_scope=fact_input.fact_scope,
            run_id=normalized_run_id,
            step_id=normalized_step_id,
            sandbox_id=context.sandbox_id,
            fact_kind=fact_input.fact_kind,
            source_ref=source_ref,
            subject_ref=subject_ref,
            profile_ref=context.profile_ref,
            summary=summary,
            payload=normalized_payload,
            payload_hash=payload_hash,
            idempotency_key=idempotency_key,
            visibility=fact_input.visibility,
            origin=classification.origin,
            trust_level=classification.trust_level,
            privacy_level=classification.privacy_level,
            retention_policy=classification.retention_policy,
        )
        async with self._uow_factory() as uow:
            return await uow.sandbox_fact.save_once(fact)

    async def list_facts(
            self,
            *,
            context: SandboxFactProjectionContext,
            fact_scope: SandboxFactScope | None = None,
            run_id: str | None = None,
            step_id: str | None = None,
            fact_kinds: list[SandboxFactKind] | None = None,
            limit: int = 100,
    ) -> list[SandboxFactRecord]:
        self._validate_context(context)
        async with self._uow_factory() as uow:
            return await uow.sandbox_fact.list_by_scope(
                user_id=context.scope.user_id,
                session_id=str(context.scope.session_id),
                fact_scope=fact_scope,
                run_id=run_id,
                step_id=step_id,
                fact_kinds=fact_kinds,
                limit=limit,
            )

    def _normalize_payload_with_context(
            self,
            *,
            context: SandboxFactProjectionContext,
            fact_input: SandboxFactInput,
    ) -> tuple[dict[str, Any], SandboxFactSubjectRef, str]:
        if isinstance(fact_input, ProfileReferenceFactInput):
            profile = context.profile_ref
            missing_fields = []
            if profile.profile_id is None:
                missing_fields.append("profile_id")
            if profile.profile_hash is None:
                missing_fields.append("profile_hash")
            if profile.sandbox_id is None:
                missing_fields.append("sandbox_id")
            payload = ProfileReferencePayload(
                profile_id=profile.profile_id,
                profile_hash=profile.profile_hash,
                sandbox_id=profile.sandbox_id,
                generated_at=profile.generated_at,
                status=profile.status,
                reason_code=fact_input.reason_code,
                missing_fields=missing_fields or None,
            )
            subject_key = profile.profile_hash or profile.profile_id or profile.status
            subject = SandboxFactSubjectRef(subject_type="profile", subject_key=subject_key)
            return payload.model_dump(mode="json"), subject, sanitize_summary(fact_input.summary or "Profile reference fact")
        return normalize_fact_input(fact_input)

    @staticmethod
    def _validate_context(context: SandboxFactProjectionContext) -> None:
        scope = context.scope
        if not str(scope.user_id or "").strip():
            raise SandboxFactScopeError("Sandbox fact context 必须包含 user_id")
        if not str(scope.session_id or "").strip():
            raise SandboxFactScopeError("Sandbox fact context 必须包含 session_id")
        if not str(scope.workspace_id or "").strip():
            raise SandboxFactScopeError("Sandbox fact context 必须包含 workspace_id")
        profile_sandbox_id = str(context.profile_ref.sandbox_id or "").strip()
        context_sandbox_id = str(context.sandbox_id or "").strip()
        if profile_sandbox_id and context_sandbox_id and profile_sandbox_id != context_sandbox_id:
            raise SandboxFactProfileMismatchError("profile_ref.sandbox_id 与 context.sandbox_id 不一致")

    @staticmethod
    def _resolve_scope(
            *,
            context: SandboxFactProjectionContext,
            fact_input: SandboxFactInput,
    ) -> tuple[str | None, str | None]:
        scope = context.scope
        run_id = fact_input.run_id if fact_input.run_id is not None else scope.run_id
        current_step_id = context.current_step_id or scope.current_step_id
        step_id = fact_input.step_id if fact_input.step_id is not None else current_step_id
        if fact_input.fact_scope == SandboxFactScope.STEP:
            if not run_id:
                raise SandboxFactScopeError("STEP scope fact 必须包含 run_id")
            if not current_step_id:
                raise SandboxFactScopeError("STEP scope fact 必须包含当前 step_id")
            if not step_id or step_id != current_step_id:
                raise SandboxFactScopeError("STEP scope fact 的 step_id 必须等于当前 step")
            if scope.run_id and run_id != scope.run_id:
                raise SandboxFactScopeError("fact run_id 与 access scope 不一致")
            return run_id, step_id
        if fact_input.fact_scope == SandboxFactScope.RUN:
            if fact_input.step_id is not None:
                raise SandboxFactScopeError("RUN scope fact 禁止传入 step_id")
            if not run_id:
                raise SandboxFactScopeError("RUN scope fact 必须包含 run_id")
            if scope.run_id and run_id != scope.run_id:
                raise SandboxFactScopeError("fact run_id 与 access scope 不一致")
            return run_id, None
        if fact_input.step_id is not None:
            raise SandboxFactScopeError("WORKSPACE scope fact 禁止传入 step_id")
        return run_id, None
