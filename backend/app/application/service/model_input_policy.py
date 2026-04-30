#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型输入能力与文档输入策略。"""

from typing import Any, Dict, List, Protocol

from app.domain.services.runtime.contracts.document_input_contract import (
    ALLOWED_DOCUMENT_INPUT_TYPES,
    LEGACY_NATIVE_INPUT_TYPES,
    normalize_document_supported_input_types,
)


class SupportsModelInputPolicy(Protocol):
    @property
    def multimodal(self) -> bool:
        ...

    @property
    def supported(self) -> list[str]:
        ...


def normalize_supported_input_types(raw_supported: object) -> list[str]:
    return normalize_document_supported_input_types(raw_supported)


def _normalize_input_part_type(raw_part: Dict[str, Any]) -> str:
    part_type = str(raw_part.get("type") or "").strip().lower()
    if part_type in ALLOWED_DOCUMENT_INPUT_TYPES or part_type in LEGACY_NATIVE_INPUT_TYPES:
        return part_type
    return ""


def _build_unsupported_part(raw_part: Dict[str, Any], *, input_type: str, reason: str) -> Dict[str, Any]:
    unsupported_part = {
        "type": input_type,
        "reason": reason,
    }
    filepath = str(raw_part.get("filepath") or raw_part.get("sandbox_filepath") or "").strip()
    if filepath:
        unsupported_part["filepath"] = filepath
    return unsupported_part


def resolve_model_input_policy(
        *,
        llm: SupportsModelInputPolicy,
        input_parts: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    native_user_content_parts: List[Dict[str, Any]] = []
    unsupported_parts: List[Dict[str, Any]] = []

    supported_types = normalize_supported_input_types(getattr(llm, "supported", []))
    multimodal_enabled = bool(getattr(llm, "multimodal", False))

    for raw_part in list(input_parts or []):
        if not isinstance(raw_part, dict):
            continue
        input_type = _normalize_input_part_type(raw_part)
        if not input_type:
            continue
        if input_type in LEGACY_NATIVE_INPUT_TYPES:
            # P0-5 禁止 image/audio/video/file/file_ref 原生透传；请求前 4xx 由 PR2 接入。
            unsupported_parts.append(
                _build_unsupported_part(
                    raw_part,
                    input_type=input_type,
                    reason="document_input_required",
                )
            )
            continue
        # PR1 只阻断 native 透传；document context 构造和 prompt 注入分别由 PR2/PR3 接入。
        if not multimodal_enabled:
            unsupported_parts.append(
                _build_unsupported_part(
                    raw_part,
                    input_type=input_type,
                    reason="model_multimodal_disabled",
                )
            )
            continue
        if input_type not in supported_types:
            unsupported_parts.append(
                _build_unsupported_part(
                    raw_part,
                    input_type=input_type,
                    reason="model_supported_not_contains_type",
                )
            )
            continue

    return {
        "native_user_content_parts": native_user_content_parts,
        "unsupported_parts": unsupported_parts,
    }
