#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""模型输入能力与多模态输入策略。"""

from typing import Any, Dict, List, Protocol

ALLOWED_SUPPORTED_INPUT_TYPES = {"image", "audio", "video", "file"}


class SupportsModelInputPolicy(Protocol):
    @property
    def multimodal(self) -> bool:
        ...

    @property
    def supported(self) -> list[str]:
        ...


def normalize_supported_input_types(raw_supported: object) -> list[str]:
    if isinstance(raw_supported, str):
        candidates = [raw_supported]
    elif isinstance(raw_supported, list):
        candidates = raw_supported
    else:
        return []

    normalized_supported: list[str] = []
    for item in candidates:
        input_type = str(item or "").strip().lower()
        if not input_type or input_type not in ALLOWED_SUPPORTED_INPUT_TYPES:
            continue
        if input_type not in normalized_supported:
            normalized_supported.append(input_type)
    return normalized_supported


def _normalize_input_part_type(raw_part: Dict[str, Any]) -> str:
    part_type = str(raw_part.get("type") or "").strip().lower()
    if part_type == "file_ref":
        return "file"
    if part_type in ALLOWED_SUPPORTED_INPUT_TYPES:
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


def _build_native_user_content_part(raw_part: Dict[str, Any], *, input_type: str) -> Dict[str, Any]:
    native_part: Dict[str, Any] = {"type": input_type}
    base64_payload = str(raw_part.get("base64") or raw_part.get("base64_payload") or "").strip()
    if base64_payload:
        native_part["base64"] = base64_payload
    mime_type = str(raw_part.get("mime_type") or "").strip()
    if mime_type:
        native_part["mime_type"] = mime_type
    file_url = str(raw_part.get("file_url") or "").strip()
    if file_url:
        native_part["file_url"] = file_url
    return native_part


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
        native_user_content_parts.append(
            _build_native_user_content_part(raw_part, input_type=input_type)
        )

    return {
        "native_user_content_parts": native_user_content_parts,
        "unsupported_parts": unsupported_parts,
    }
