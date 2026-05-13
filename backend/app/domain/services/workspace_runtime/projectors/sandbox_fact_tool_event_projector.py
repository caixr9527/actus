#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ToolEvent 到 Sandbox Fact 的投影器。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel

from app.application.service.sandbox_fact_ledger_service import (
    BrowserActionFactInput,
    BrowserSnapshotFactInput,
    CommandExecutionFactInput,
    FetchedPageFactInput,
    FileListEntryFactInput,
    FileListFactInput,
    FileMutationFactInput,
    FileReadFactInput,
    FileSearchFactInput,
    FileSearchMatchInput,
    SandboxFactInput,
    SandboxFactLedgerService,
    SearchResultFactInput,
    SearchResultItemInput,
    ShellOutputFactInput,
    ToolFailureFactInput,
)
from app.domain.models import ToolEvent, ToolEventStatus
from app.domain.models.sandbox_fact import SandboxFactRecord, SandboxFactScope, SandboxFactSourceType
from app.domain.services.runtime.contracts.evidence_key_normalizer import (
    build_file_mutation_intent_hash,
    short_hash,
)
from app.domain.services.runtime.contracts.sandbox_fact_ports import (
    SandboxFactProjectionContext,
    SandboxFactRecorderPort,
)

SHELL_FUNCTIONS = {"exec_command", "shell_execute"}
SHELL_OUTPUT_FUNCTIONS = {"read_shell_output", "wait_process", "write_shell_input", "kill_process"}
FILE_READ_FUNCTIONS = {"read_file"}
FILE_WRITE_FUNCTIONS = {"write_file", "replace_in_file"}
FILE_DELETE_FUNCTIONS = {"delete_file"}
FILE_LIST_FUNCTIONS = {"list_files", "find_files", "check_file_exists"}
FILE_SEARCH_FUNCTIONS = {"search_in_file"}
SEARCH_FUNCTIONS = {"search_web"}
FETCH_FUNCTIONS = {"fetch_page"}
DOCUMENT_FUNCTIONS = {"document_context"}
BROWSER_SNAPSHOT_FUNCTIONS = {
    "browser_view",
    "browser_read_current_page_structured",
    "browser_extract_main_content",
    "browser_extract_cards",
    "browser_get_actionable_elements",
}
BROWSER_ACTION_FUNCTIONS = {
    "browser_click",
    "browser_type",
    "browser_scroll_down",
    "browser_scroll_up",
    "browser_go_back",
    "browser_go_forward",
    "browser_navigate",
    "browser_find_link_by_text",
    "browser_open",
}


class SandboxFactToolEventProjector(SandboxFactRecorderPort):
    """将 ToolEvent 稳定转换为 PR2 领域输入并写入 fact ledger。"""

    def __init__(self, *, ledger_service: SandboxFactLedgerService) -> None:
        self._ledger_service = ledger_service

    async def record_from_tool_event(
            self,
            *,
            context: SandboxFactProjectionContext,
            event: ToolEvent,
    ) -> list[SandboxFactRecord]:
        fact_inputs = self._build_fact_inputs(context=context, event=event)
        facts = []
        for fact_input in fact_inputs:
            facts.append(await self._ledger_service.record_fact(context=context, fact_input=fact_input))
        return facts

    def _build_fact_inputs(
            self,
            *,
            context: SandboxFactProjectionContext,
            event: ToolEvent,
    ) -> list[SandboxFactInput]:
        if event.status != ToolEventStatus.CALLED:
            return []
        function_name = _normalize_text(event.function_name).lower()
        result = event.function_result
        if result is None:
            return [self._tool_failure_input(context=context, event=event, reason_code="tool_result_missing")]
        if not bool(result.success):
            return [self._tool_failure_input(context=context, event=event, reason_code="tool_failed")]
        if _is_evidence_reuse_virtual_success(event):
            return []
        if function_name in DOCUMENT_FUNCTIONS:
            return []

        base = self._base_fields(context=context, event=event)
        data = _to_mapping(result.data)
        args = _to_mapping(event.function_args)
        if function_name in SHELL_FUNCTIONS:
            return [self._command_input(base=base, event=event, args=args, data=data)]
        if function_name in SHELL_OUTPUT_FUNCTIONS:
            return [self._shell_output_input(base=base, event=event, args=args, data=data)]
        if function_name in FILE_READ_FUNCTIONS:
            return [self._file_read_input(base=base, args=args, data=data)]
        if function_name in FILE_WRITE_FUNCTIONS:
            return [self._file_mutation_input(base=base, args=args, data=data, fact_kind="file_write", operation="write")]
        if function_name in FILE_DELETE_FUNCTIONS:
            return [self._file_mutation_input(base=base, args=args, data=data, fact_kind="file_delete", operation="delete")]
        if function_name in FILE_LIST_FUNCTIONS:
            return [self._file_list_input(base=base, function_name=function_name, args=args, data=data, raw_data=result.data)]
        if function_name in FILE_SEARCH_FUNCTIONS:
            return [self._file_search_input(base=base, args=args, data=data)]
        if function_name in SEARCH_FUNCTIONS:
            return [self._search_input(base=base, args=args, data=data)]
        if function_name in FETCH_FUNCTIONS:
            return [self._fetched_page_input(base=base, args=args, data=data)]
        if function_name in BROWSER_SNAPSHOT_FUNCTIONS:
            return [self._browser_snapshot_input(base=base, data=data)]
        if function_name in BROWSER_ACTION_FUNCTIONS or function_name.startswith("browser_"):
            return [self._browser_action_input(base=base, event=event, args=args, data=data)]
        return []

    @staticmethod
    def _base_fields(*, context: SandboxFactProjectionContext, event: ToolEvent) -> dict[str, Any]:
        fact_scope, missing_fields, reason_code = _resolve_projection_scope(context)
        return {
            "fact_scope": fact_scope,
            "summary": f"{event.function_name} tool fact",
            "source_type": SandboxFactSourceType.SANDBOX_API,
            "tool_event_id": event.id,
            "tool_call_id": event.tool_call_id,
            "function_name": event.function_name,
            "missing_fields": missing_fields,
            "reason_code": reason_code,
        }

    def _tool_failure_input(
            self,
            *,
            context: SandboxFactProjectionContext,
            event: ToolEvent,
            reason_code: str,
    ) -> ToolFailureFactInput:
        base = self._base_fields(context=context, event=event)
        result = event.function_result
        data = _to_mapping(result.data if result is not None else None)
        message = _first_text(
            result.message if result is not None else "",
            data.get("message"),
            data.get("error"),
            data.get("reason"),
            default="Tool execution failed",
        )
        timeout = bool(data.get("timeout") or data.get("timed_out"))
        return ToolFailureFactInput(
            **_with_overrides(
                base,
                function_name=event.function_name,
                reason_code=_first_text(data.get("reason_code"), reason_code, default="tool_failed"),
            ),
            message=message,
            timeout=timeout,
            diagnostic_type=_first_text(data.get("diagnostic_type"), "tool_execution", default="tool_execution"),
        )

    @staticmethod
    def _command_input(
            *,
            base: dict[str, Any],
            event: ToolEvent,
            args: Mapping[str, Any],
            data: Mapping[str, Any],
    ) -> CommandExecutionFactInput:
        command = _first_text(args.get("command"), data.get("command"), data.get("input"), default=event.function_name)
        stdout = _first_text(data.get("stdout"), data.get("output"), data.get("console"), default="")
        stderr = _first_text(data.get("stderr"), data.get("error"), default="")
        return CommandExecutionFactInput(
            **base,
            command=command,
            cwd=_first_text(data.get("cwd"), args.get("cwd"), default=""),
            exit_code=_optional_int(data.get("exit_code")),
            duration_ms=_optional_int(data.get("duration_ms")),
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=bool(data.get("stdout_truncated") or data.get("output_truncated")),
            stderr_truncated=bool(data.get("stderr_truncated")),
            changed_paths=_string_list(data.get("changed_paths")),
            timeout=bool(data.get("timeout") or data.get("timed_out")),
        )

    @staticmethod
    def _shell_output_input(
            *,
            base: dict[str, Any],
            event: ToolEvent,
            args: Mapping[str, Any],
            data: Mapping[str, Any],
    ) -> ShellOutputFactInput:
        records = _sequence(data.get("console_records"))
        return ShellOutputFactInput(
            **base,
            session_ref=_first_text(args.get("session_ref"), data.get("session_ref"), event.tool_call_id, default=event.id),
            output=_first_text(data.get("output"), data.get("console"), default=""),
            output_truncated=bool(data.get("output_truncated") or data.get("truncated")),
            console_record_count=len(records),
            process_status=_first_text(data.get("process_status"), data.get("status"), default="unknown"),
            exit_code=_optional_int(data.get("exit_code")),
            duration_ms=_optional_int(data.get("duration_ms")),
        )

    @staticmethod
    def _file_read_input(
            *,
            base: dict[str, Any],
            args: Mapping[str, Any],
            data: Mapping[str, Any],
    ) -> FileReadFactInput:
        path = _first_text(data.get("filepath"), data.get("path"), args.get("filepath"), args.get("file_path"), default="")
        content = _first_text(data.get("content"), data.get("excerpt"), default="")
        content_sha256 = _optional_text(
            data.get("content_sha256")
            or data.get("read_content_sha256")
            or data.get("sha256")
        )
        if not content_sha256 and content:
            content_sha256 = short_hash(content)
        content_sha256_kind = _sha256_kind(
            data.get("content_sha256_kind")
            or data.get("sha256_kind")
            or ("read_content_sha256" if content_sha256 else None)
        )
        return FileReadFactInput(
            **_with_overrides(
                base,
                missing_fields=_missing_fields(base, ["content_sha256"] if not content_sha256 else []),
                reason_code=_reason_code(base, "content_sha256_missing" if not content_sha256 else None),
            ),
            path=path,
            exists=bool(data.get("exists", True)),
            size=_optional_int(data.get("size") or data.get("content_length")),
            content_sha256=content_sha256,
            content_sha256_kind=content_sha256_kind,
            mime_type=_first_text(data.get("mime_type"), data.get("content_type"), default="text/plain"),
            line_range=_line_range(args=args, data=data),
            content=content,
            is_truncated=bool(data.get("is_truncated") or data.get("truncated")),
        )

    @staticmethod
    def _file_mutation_input(
            *,
            base: dict[str, Any],
            args: Mapping[str, Any],
            data: Mapping[str, Any],
            fact_kind: str,
            operation: str,
    ) -> FileMutationFactInput:
        path = _first_text(data.get("filepath"), data.get("path"), args.get("filepath"), args.get("file_path"), default="")
        return FileMutationFactInput(
            **base,
            fact_kind=fact_kind,
            path=path,
            operation=operation,
            mutation_intent_hash=build_file_mutation_intent_hash(
                path=path,
                operation=operation,
                content=str(args.get("content") or ""),
                old_str=str(args.get("old_str") or args.get("old_string") or ""),
                new_str=str(args.get("new_str") or args.get("new_string") or ""),
                append=bool(args.get("append")),
                leading_newline=bool(args.get("leading_newline")),
                trailing_newline=bool(args.get("trailing_newline")),
            ),
            exists=bool(data.get("exists", operation != "delete")),
            before_content_sha256=_optional_text(data.get("before_content_sha256")),
            after_content_sha256=_optional_text(data.get("after_content_sha256") or data.get("content_sha256") or data.get("sha256")),
            content_sha256_kind=_sha256_kind(data.get("content_sha256_kind") or data.get("sha256_kind")),
            size_after=_optional_int(data.get("size_after") or data.get("size")),
            changed=bool(data.get("changed", True)),
        )

    @staticmethod
    def _file_list_input(
            *,
            base: dict[str, Any],
            function_name: str,
            args: Mapping[str, Any],
            data: Mapping[str, Any],
            raw_data: Any,
    ) -> FileListFactInput:
        if function_name == "check_file_exists":
            path = _first_text(args.get("filepath"), args.get("file_path"), data.get("filepath"), data.get("path"), default="")
            exists = bool(data.get("exists", raw_data if isinstance(raw_data, bool) else False)) if data else bool(raw_data)
            entries = [FileListEntryFactInput(name=path, type="file")] if exists and path else []
            return FileListFactInput(
                **_with_overrides(
                    base,
                    missing_fields=_missing_fields(base, ["file_absent"] if not exists else []),
                    reason_code=_reason_code(base, "file_absent" if not exists else None),
                ),
                dir_path=path,
                entry_count=1 if exists else 0,
                entries=entries,
            )

        dir_path = _first_text(data.get("dir_path"), args.get("dir_path"), default="")
        raw_entries = _sequence(data.get("entries") or data.get("files"))
        entries = [_file_list_entry(item) for item in raw_entries]
        return FileListFactInput(
            **base,
            dir_path=dir_path,
            entry_count=_optional_int(data.get("entry_count")) if data.get("entry_count") is not None else len(entries),
            entries=entries,
            is_truncated=bool(data.get("is_truncated") or data.get("truncated")),
        )

    @staticmethod
    def _file_search_input(
            *,
            base: dict[str, Any],
            args: Mapping[str, Any],
            data: Mapping[str, Any],
    ) -> FileSearchFactInput:
        matches = [
            FileSearchMatchInput(
                path=_first_text(_to_mapping(item).get("path"), args.get("filepath"), args.get("file_path"), default=""),
                line_number=_optional_int(_to_mapping(item).get("line_number")),
                excerpt=_first_text(_to_mapping(item).get("excerpt"), _to_mapping(item).get("line"), default=""),
            )
            for item in _sequence(data.get("matches"))
        ]
        return FileSearchFactInput(
            **base,
            path=_first_text(data.get("path"), args.get("filepath"), args.get("file_path"), default=""),
            regex=_first_text(data.get("regex"), args.get("regex"), default=""),
            match_count=_optional_int(data.get("match_count")) if data.get("match_count") is not None else len(matches),
            matches=matches,
            is_truncated=bool(data.get("is_truncated") or data.get("truncated")),
        )

    @staticmethod
    def _search_input(
            *,
            base: dict[str, Any],
            args: Mapping[str, Any],
            data: Mapping[str, Any],
    ) -> SearchResultFactInput:
        raw_results = _sequence(data.get("results"))
        items: list[SearchResultItemInput] = []
        for item in raw_results:
            item_data = _to_mapping(item)
            url = _first_text(item_data.get("url"), default="")
            if not url:
                continue
            items.append(
                SearchResultItemInput(
                    title=_first_text(item_data.get("title"), default=""),
                    url=url,
                    snippet=_first_text(item_data.get("snippet"), item_data.get("content"), default=""),
                )
            )
        raw_count = data.get("total_results") if data.get("total_results") is not None else data.get("result_count")
        parsed_count = _optional_int(raw_count) if raw_count is not None else None
        result_count = parsed_count if parsed_count and parsed_count > 0 else len(raw_results)
        return SearchResultFactInput(
            **base,
            query=_first_text(args.get("query"), data.get("query"), default=""),
            result_count=result_count,
            top_results=items,
        )

    @staticmethod
    def _fetched_page_input(
            *,
            base: dict[str, Any],
            args: Mapping[str, Any],
            data: Mapping[str, Any],
    ) -> FetchedPageFactInput:
        return FetchedPageFactInput(
            **base,
            fetched_url=_first_text(data.get("url"), args.get("url"), default=""),
            final_url=_first_text(data.get("final_url"), data.get("url"), args.get("url"), default=""),
            status_code=_optional_int(data.get("status_code")),
            content_type=_optional_text(data.get("content_type")),
            title=_first_text(data.get("title"), default=""),
            content=_first_text(data.get("excerpt"), data.get("content"), default=""),
            is_truncated=bool(data.get("truncated") or data.get("is_truncated")),
        )

    @staticmethod
    def _browser_snapshot_input(
            *,
            base: dict[str, Any],
            data: Mapping[str, Any],
    ) -> BrowserSnapshotFactInput:
        actionable_elements = _sequence(data.get("actionable_elements") or data.get("elements"))
        screenshot_file = _to_mapping(data.get("screenshot_file"))
        return BrowserSnapshotFactInput(
            **base,
            url=_first_text(data.get("url"), default=""),
            title=_first_text(data.get("title"), default=""),
            screenshot_artifact_id=None,
            screenshot_artifact_path=None,
            screenshot_file_id=_optional_text(screenshot_file.get("file_id")),
            screenshot_filename=_optional_text(screenshot_file.get("filename")),
            screenshot_filepath=_optional_text(screenshot_file.get("filepath")),
            screenshot_key=_optional_text(screenshot_file.get("key")),
            screenshot_mime_type=_optional_text(screenshot_file.get("mime_type")),
            screenshot_size=_optional_int(screenshot_file.get("size")),
            structured_summary=_first_text(data.get("content_summary"), data.get("main_content_preview"), data.get("excerpt"), data.get("content"), default=""),
            actionable_element_count=len(actionable_elements),
            degrade_reason=_optional_text(data.get("degrade_reason")),
        )

    @staticmethod
    def _browser_action_input(
            *,
            base: dict[str, Any],
            event: ToolEvent,
            args: Mapping[str, Any],
            data: Mapping[str, Any],
    ) -> BrowserActionFactInput:
        return BrowserActionFactInput(
            **base,
            action=event.function_name,
            target_summary=_first_text(args.get("selector"), args.get("text"), args.get("url"), data.get("matched_text"), data.get("url"), default=event.function_name),
            url_before=_optional_text(data.get("url_before")),
            url_after=_optional_text(data.get("url_after") or data.get("url")),
            success=True,
            degrade_reason=_optional_text(data.get("degrade_reason")),
        )


def _resolve_projection_scope(context: SandboxFactProjectionContext) -> tuple[SandboxFactScope, list[str] | None, str | None]:
    if context.current_step_id or context.scope.current_step_id:
        return SandboxFactScope.STEP, None, None
    if context.scope.run_id:
        return SandboxFactScope.RUN, ["current_step_id"], "current_step_missing"
    return SandboxFactScope.WORKSPACE, ["run_id"], "run_id_missing"


def _missing_fields(base: Mapping[str, Any], extra: list[str]) -> list[str] | None:
    values = [str(item) for item in (base.get("missing_fields") or [])]
    values.extend(extra)
    return values or None


def _with_overrides(base: Mapping[str, Any], **overrides: Any) -> dict[str, Any]:
    values = dict(base)
    values.update(overrides)
    return values


def _reason_code(base: Mapping[str, Any], fallback: str | None) -> str | None:
    return str(base.get("reason_code") or fallback or "").strip() or None


def _to_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return value
    return {}


def _is_evidence_reuse_virtual_success(event: ToolEvent) -> bool:
    result = event.function_result
    if result is None or not bool(result.success):
        return False
    data = result.data if isinstance(result.data, dict) else {}
    return (
        data.get("result_handle_resolved") is True
        or data.get("duplicate_decision") == "reuse_existing_evidence"
    )


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    return []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _sequence(value) if str(item or "").strip()]


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _optional_text(value: Any) -> str | None:
    normalized = _normalize_text(value)
    return normalized or None


def _first_text(*values: Any, default: str) -> str:
    for value in values:
        normalized = _optional_text(value)
        if normalized is not None:
            return normalized
    return default


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _sha256_kind(value: Any) -> str:
    normalized = _normalize_text(value)
    if normalized in {"full_file_sha256", "read_content_sha256"}:
        return normalized
    return "unknown"


def _line_range(*, args: Mapping[str, Any], data: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = data.get("line_range")
    if isinstance(raw, Mapping):
        return dict(raw)
    start_line = args.get("start_line")
    end_line = args.get("end_line")
    if start_line is None and end_line is None:
        return None
    return {"start_line": start_line, "end_line": end_line}


def _file_list_entry(item: Any) -> FileListEntryFactInput:
    if isinstance(item, Mapping):
        return FileListEntryFactInput(
            name=_first_text(item.get("name"), item.get("path"), item.get("filepath"), default=""),
            type=_first_text(item.get("type"), item.get("kind"), default="file"),
            size=_optional_int(item.get("size")),
        )
    return FileListEntryFactInput(name=str(item or "").strip(), type="file")
