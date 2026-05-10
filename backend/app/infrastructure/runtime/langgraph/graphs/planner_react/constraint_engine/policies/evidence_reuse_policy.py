#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""执行约束策略：Evidence reuse 去重。"""

from __future__ import annotations

import logging
from typing import Optional

from app.domain.models.evidence import (
    EvidenceDuplicateDecision,
    EvidenceReuseSnapshot,
)
from app.domain.services.runtime.contracts.langgraph_settings import ASK_USER_FUNCTION_NAME
from app.domain.services.runtime.contracts.runtime_logging import log_runtime
from app.domain.services.runtime.contracts.evidence_key_normalizer import (
    build_evidence_action_subject_key_from_tool_call,
    hash_query,
    hash_url,
)

from ..contracts import ConstraintDecision, ConstraintInput, ConstraintToolResultPayload
from ..reason_codes import (
    REASON_ALLOW,
    REASON_EVIDENCE_DUPLICATE_BLOCKED,
    REASON_EVIDENCE_REUSE_PENDING_RESOLUTION,
    REASON_EVIDENCE_REUSE_REQUIRES_VERIFICATION,
    REASON_EVIDENCE_REUSE_SNAPSHOT_MISSING,
)


def evaluate_evidence_reuse_policy(constraint_input: ConstraintInput) -> Optional[ConstraintDecision]:
    """基于 EvidenceReuseSnapshot 在真实 executor 前拦截重复动作。"""

    function_name = str(constraint_input.normalized_function_name or constraint_input.function_name or "").strip().lower()
    if function_name == ASK_USER_FUNCTION_NAME:
        return None

    snapshot = _strict_snapshot(constraint_input.external_signals_snapshot.get("evidence_reuse_snapshot"))
    has_previous_completed_steps = bool(
        constraint_input.external_signals_snapshot.get("has_previous_completed_steps")
    )
    if snapshot is None:
        if has_previous_completed_steps:
            return ConstraintDecision(
                action="block",
                reason_code=REASON_EVIDENCE_REUSE_SNAPSHOT_MISSING,
                block_mode="hard_block_break",
                loop_break_reason=REASON_EVIDENCE_REUSE_SNAPSHOT_MISSING,
                tool_result_payload=ConstraintToolResultPayload(
                    success=False,
                    message="前序 completed step 存在，但 execute context 缺少 evidence reuse snapshot。",
                    data={"duplicate_decision": "snapshot_missing_with_previous_completed_step"},
                ),
                message_for_model="前序步骤 evidence context 缺失，已停止真实工具调用。",
            )
        return None

    function_args = dict(constraint_input.function_args or {})
    verification_rewrite = _build_verification_rewrite_decision(
        snapshot=snapshot,
        function_name=function_name,
        function_args=function_args,
    )
    if verification_rewrite is not None:
        return verification_rewrite

    key_result = build_evidence_action_subject_key_from_tool_call(function_name, function_args)
    if key_result.normalization_status != "normalized" or not key_result.action_key or not key_result.subject_key:
        return None

    matched = None
    for item in snapshot.do_not_repeat:
        if item.action_key == key_result.action_key and item.subject_key == key_result.subject_key:
            matched = item
            break
    if matched is None:
        verification_mismatch = _verification_candidate_for_function(
            snapshot=snapshot,
            function_name=function_name,
            function_args=function_args,
        )
        if verification_mismatch is not None:
            data = _base_data(verification_mismatch)
            data["verification_gap"] = {
                "reason_code": "verification_action_missing",
                "verification_mismatch": "action_subject_key_mismatch",
            }
            return ConstraintDecision(
                action="block",
                reason_code=REASON_EVIDENCE_REUSE_REQUIRES_VERIFICATION,
                block_mode="soft_block_continue",
                loop_break_reason="",
                tool_result_payload=ConstraintToolResultPayload(
                    success=False,
                    message=verification_mismatch.message_for_model or "验证动作与原 evidence key 不一致。",
                    data=data,
                ),
                message_for_model=verification_mismatch.message_for_model,
            )
        return None

    if matched.duplicate_decision == EvidenceDuplicateDecision.REUSE_EXISTING_EVIDENCE_PENDING_RESOLUTION:
        handle = None
        for candidate in snapshot.result_handles:
            if candidate.result_handle_id == matched.result_handle_id:
                handle = candidate
                break
        if handle is None:
            return _block_duplicate(matched, reason_code="result_handle_missing")
        data = _base_data(matched)
        data["result_handle"] = handle.model_dump(mode="json")
        return ConstraintDecision(
            action="block",
            reason_code=REASON_EVIDENCE_REUSE_PENDING_RESOLUTION,
            block_mode="virtual_success_pending_resolution",
            loop_break_reason="virtual_success_pending_resolution",
            tool_result_payload=ConstraintToolResultPayload(
                success=False,
                message=matched.message_for_model or "已发现可复用前序 evidence，等待解析 result handle。",
                data=data,
            ),
            message_for_model=matched.message_for_model,
        )

    if matched.duplicate_decision == EvidenceDuplicateDecision.REQUIRE_VERIFICATION:
        if _is_allowed_verification_call(
                matched=matched,
                function_name=function_name,
                function_args=function_args,
        ):
            _log_verification_allowed(
                constraint_input=constraint_input,
                matched=matched,
                function_name=function_name,
            )
            return None
        data = _base_data(matched)
        data["verification_gap"] = {
            "reason_code": "verification_action_missing",
            "allowed_verification_actions": (
                list(matched.reuse_result_ref.allowed_verification_actions)
                if matched.reuse_result_ref is not None
                else []
            ),
            "verification_reason_code": (
                matched.reuse_result_ref.reason_code
                if matched.reuse_result_ref is not None
                else None
            ),
        }
        return ConstraintDecision(
            action="block",
            reason_code=REASON_EVIDENCE_REUSE_REQUIRES_VERIFICATION,
            block_mode="soft_block_continue",
            loop_break_reason="",
            tool_result_payload=ConstraintToolResultPayload(
                success=False,
                message=matched.message_for_model or "已有 evidence 需要先验证后才能复用。",
                data=data,
            ),
            message_for_model=matched.message_for_model,
        )

    return _block_duplicate(matched, reason_code=REASON_EVIDENCE_DUPLICATE_BLOCKED)


def _strict_snapshot(raw) -> EvidenceReuseSnapshot | None:
    if raw is None:
        return None
    if isinstance(raw, EvidenceReuseSnapshot):
        return raw
    return EvidenceReuseSnapshot.model_validate(raw)


def _base_data(item) -> dict:
    return {
        "duplicate_decision": item.duplicate_decision.value,
        "evidence_ids": list(item.evidence_ids or []),
        "fact_ids": [],
        "artifact_ids": [],
        "action_key": item.action_key,
        "subject_key": item.subject_key,
        "source_step_id": item.source_step_id,
        "reuse_summary": item.reuse_summary,
        "reuse_result_ref": item.reuse_result_ref.model_dump(mode="json") if item.reuse_result_ref else None,
        "result_handle_id": item.result_handle_id,
    }


def _build_verification_rewrite_decision(
        *,
        snapshot: EvidenceReuseSnapshot,
        function_name: str,
        function_args: dict,
) -> ConstraintDecision | None:
    normalized_function = str(function_name or "").strip().lower()
    if normalized_function not in {"search_web", "fetch_page"}:
        return None
    if function_args.get("verification_reason_code"):
        return None
    if normalized_function == "search_web":
        query = str((function_args or {}).get("query") or "").strip()
        if not query:
            return None
        metadata_hash = hash_query(query)
        expected_subject_key = f"query:{metadata_hash}"
        audit_key = "query_hash"
    else:
        url = str((function_args or {}).get("url") or "").strip()
        if not url:
            return None
        metadata_hash = hash_url(url)
        expected_subject_key = f"page:{metadata_hash}"
        audit_key = "url_hash"

    for item in snapshot.do_not_repeat:
        if item.duplicate_decision != EvidenceDuplicateDecision.REQUIRE_VERIFICATION:
            continue
        result_ref = item.reuse_result_ref
        if result_ref is None:
            continue
        allowed_actions = {
            str(action or "").strip().lower()
            for action in list(result_ref.allowed_verification_actions or [])
            if str(action or "").strip()
        }
        if normalized_function not in allowed_actions:
            continue
        expected_subject_keys = {
            str(value or "").strip()
            for value in (item.subject_key, result_ref.subject_key)
            if str(value or "").strip()
        }
        if expected_subject_key not in expected_subject_keys:
            continue
        reason_code = str(result_ref.reason_code or "").strip()
        if not reason_code:
            continue
        rewritten_args = dict(function_args or {})
        rewritten_args[audit_key] = metadata_hash
        rewritten_args["verification_reason_code"] = reason_code
        return ConstraintDecision(
            action="rewrite",
            reason_code=REASON_EVIDENCE_REUSE_REQUIRES_VERIFICATION,
            rewrite_target={
                "function_name": normalized_function,
                "normalized_function_name": normalized_function,
                "function_args": rewritten_args,
            },
            metadata={
                "rewrite_type": "evidence_verification_audit_metadata",
                "action_key": item.action_key,
                "subject_key": item.subject_key,
                "verification_reason_code": reason_code,
            },
        )
    return None


def _is_allowed_verification_call(*, matched, function_name: str, function_args: dict) -> bool:
    result_ref = matched.reuse_result_ref
    if result_ref is None:
        return False
    allowed_actions = {
        str(action or "").strip().lower()
        for action in list(result_ref.allowed_verification_actions or [])
        if str(action or "").strip()
    }
    normalized_function = str(function_name or "").strip().lower()
    if normalized_function not in allowed_actions:
        return False
    expected_reason_code = str(result_ref.reason_code or "").strip()
    actual_reason_code = str((function_args or {}).get("verification_reason_code") or "").strip()
    if not expected_reason_code or actual_reason_code != expected_reason_code:
        return False
    if not _has_executable_verification_args(
            function_name=normalized_function,
            function_args=function_args or {},
    ):
        return False
    if not _verification_hash_matches_source(
            matched=matched,
            function_name=normalized_function,
            function_args=function_args or {},
    ):
        return False
    return True


def _has_executable_verification_args(*, function_name: str, function_args: dict) -> bool:
    if function_name == "search_web":
        return bool(str((function_args or {}).get("query") or "").strip())
    if function_name == "fetch_page":
        return bool(str((function_args or {}).get("url") or "").strip())
    return True


def _verification_hash_matches_source(*, matched, function_name: str, function_args: dict) -> bool:
    expected_subject_keys = {
        str(value or "").strip()
        for value in (
            matched.subject_key,
            getattr(matched.reuse_result_ref, "subject_key", None),
        )
        if str(value or "").strip()
    }
    if function_name == "search_web":
        query = str((function_args or {}).get("query") or "").strip()
        metadata_hash = str((function_args or {}).get("query_hash") or "").strip()
        if not query or not metadata_hash:
            return False
        if metadata_hash != hash_query(query):
            return False
        return f"query:{metadata_hash}" in expected_subject_keys
    if function_name == "fetch_page":
        url = str((function_args or {}).get("url") or "").strip()
        metadata_hash = str(
            (function_args or {}).get("url_hash")
            or (function_args or {}).get("fetched_url_hash")
            or ""
        ).strip()
        if not url or not metadata_hash:
            return False
        if metadata_hash != hash_url(url):
            return False
        return f"page:{metadata_hash}" in expected_subject_keys
    return True


def _verification_candidate_for_function(*, snapshot: EvidenceReuseSnapshot, function_name: str, function_args: dict):
    verification_reason_code = str((function_args or {}).get("verification_reason_code") or "").strip()
    if not verification_reason_code:
        return None
    normalized_function = str(function_name or "").strip().lower()
    for item in snapshot.do_not_repeat:
        if item.duplicate_decision != EvidenceDuplicateDecision.REQUIRE_VERIFICATION:
            continue
        result_ref = item.reuse_result_ref
        if result_ref is None:
            continue
        allowed_actions = {
            str(action or "").strip().lower()
            for action in list(result_ref.allowed_verification_actions or [])
            if str(action or "").strip()
        }
        if normalized_function in allowed_actions:
            return item
    return None


def _log_verification_allowed(*, constraint_input: ConstraintInput, matched, function_name: str) -> None:
    result_ref = matched.reuse_result_ref
    logger = logging.getLogger(__name__)
    log_runtime(
        logger,
        logging.INFO,
        "evidence_reuse_verification_allowed",
        step_id=str(getattr(constraint_input.step, "id", "") or ""),
        function_name=str(function_name or "").strip().lower(),
        action_key=str(matched.action_key or ""),
        subject_key=str(matched.subject_key or ""),
        verification_reason_code=str(getattr(result_ref, "reason_code", "") or ""),
        allowed_action=str(function_name or "").strip().lower(),
    )


def _block_duplicate(item, *, reason_code: str) -> ConstraintDecision:
    return ConstraintDecision(
        action="block",
        reason_code=reason_code,
        block_mode="hard_block_break",
        loop_break_reason=reason_code,
        tool_result_payload=ConstraintToolResultPayload(
            success=False,
            message=item.message_for_model or "重复动作已被 Evidence Ledger 阻断。",
            data=_base_data(item),
        ),
        message_for_model=item.message_for_model,
    )
