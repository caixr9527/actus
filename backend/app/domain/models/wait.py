#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""等待用户输入的中断载荷模型。"""

from __future__ import annotations

import json
import logging
from typing import Any, Annotated, Dict, List, Literal, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError, field_validator, model_validator

logger = logging.getLogger(__name__)

WaitUserTakeover = Literal["none", "browser"]


class WaitChoice(BaseModel):
    """可选项等待态中的单个选项。"""

    label: str
    resume_value: Any
    description: str = ""

    @field_validator("label", mode="before")
    @classmethod
    def _normalize_label(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("description", mode="before")
    @classmethod
    def _normalize_description(cls, value: Any) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def _validate_choice(self) -> "WaitChoice":
        if not self.label:
            raise ValueError("等待选项缺少 label")
        return self


class BaseWaitPayload(BaseModel):
    """等待态通用字段。"""

    kind: Literal["input_text", "confirm", "select"]
    title: str = ""
    prompt: str
    details: str = ""
    attachments: List[str] = Field(default_factory=list)
    suggest_user_takeover: WaitUserTakeover = "none"

    @field_validator("title", "prompt", "details", mode="before")
    @classmethod
    def _normalize_text(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("attachments", mode="before")
    @classmethod
    def _normalize_attachments(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            candidate = value.strip()
            return [candidate] if candidate else []
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    @field_validator("suggest_user_takeover", mode="before")
    @classmethod
    def _normalize_takeover(cls, value: Any) -> WaitUserTakeover:
        takeover = str(value or "none").strip().lower()
        return "browser" if takeover == "browser" else "none"

    @model_validator(mode="after")
    def _validate_prompt(self) -> "BaseWaitPayload":
        if not self.prompt:
            raise ValueError("等待态 prompt 不能为空")
        return self


class TextInputWaitPayload(BaseWaitPayload):
    """等待文本输入。"""

    kind: Literal["input_text"] = "input_text"
    placeholder: str = ""
    submit_label: str = "继续执行"
    response_key: str = "message"
    default_value: str = ""
    multiline: bool = True
    allow_empty: bool = False

    @field_validator("placeholder", "submit_label", "response_key", "default_value", mode="before")
    @classmethod
    def _normalize_fields(cls, value: Any) -> str:
        return str(value or "").strip()

    @model_validator(mode="after")
    def _validate_input_text(self) -> "TextInputWaitPayload":
        if not self.response_key:
            raise ValueError("文本输入等待态缺少 response_key")
        if not self.submit_label:
            self.submit_label = "继续执行"
        return self


class ConfirmWaitPayload(BaseWaitPayload):
    """等待确认。"""

    kind: Literal["confirm"] = "confirm"
    confirm_label: str = "继续"
    cancel_label: str = "取消"
    confirm_resume_value: Any = True
    cancel_resume_value: Any = False
    emphasis: Literal["default", "destructive"] = "default"

    @field_validator("confirm_label", "cancel_label", mode="before")
    @classmethod
    def _normalize_labels(cls, value: Any) -> str:
        return str(value or "").strip()

    @field_validator("emphasis", mode="before")
    @classmethod
    def _normalize_emphasis(cls, value: Any) -> Literal["default", "destructive"]:
        emphasis = str(value or "default").strip().lower()
        return "destructive" if emphasis == "destructive" else "default"

    @model_validator(mode="after")
    def _validate_confirm(self) -> "ConfirmWaitPayload":
        if not self.confirm_label:
            self.confirm_label = "继续"
        if not self.cancel_label:
            self.cancel_label = "取消"
        return self


class SelectWaitPayload(BaseWaitPayload):
    """等待单选。"""

    kind: Literal["select"] = "select"
    options: List[WaitChoice]
    default_resume_value: Any = None

    @model_validator(mode="after")
    def _validate_select(self) -> "SelectWaitPayload":
        if len(self.options) == 0:
            raise ValueError("选项等待态至少需要一个 option")
        return self


WaitPayload = Annotated[
    Union[
        TextInputWaitPayload,
        ConfirmWaitPayload,
        SelectWaitPayload,
    ],
    Field(discriminator="kind"),
]

_WAIT_PAYLOAD_ADAPTER = TypeAdapter(WaitPayload)


def normalize_wait_payload(raw: Any) -> Dict[str, Any]:
    """将等待载荷规范化为稳定字典结构。"""
    if raw is None:
        return {}
    if isinstance(raw, dict) and len(raw) == 0:
        return {}

    try:
        payload = _WAIT_PAYLOAD_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        logger.warning("等待态载荷校验失败，忽略该中断: %s", exc)
        return {}

    return payload.model_dump(mode="json")


def resolve_wait_resume_message(payload_raw: Any, resume_value: Any) -> str:
    """将恢复值转换为后续规划可消费的文本。"""
    payload = normalize_wait_payload(payload_raw)
    if not payload:
        return _fallback_resume_message(resume_value)

    kind = payload.get("kind")
    if kind == "confirm":
        if resume_value == payload.get("confirm_resume_value"):
            return str(payload.get("confirm_label") or "").strip()
        if resume_value == payload.get("cancel_resume_value"):
            return str(payload.get("cancel_label") or "").strip()
        return _fallback_resume_message(resume_value)

    if kind == "select":
        for option in list(payload.get("options") or []):
            if not isinstance(option, dict):
                continue
            if resume_value == option.get("resume_value"):
                return str(option.get("label") or "").strip()
        return _fallback_resume_message(resume_value)

    if kind == "input_text" and isinstance(resume_value, dict):
        response_key = str(payload.get("response_key") or "message").strip()
        candidate = resume_value.get(response_key)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()

    return _fallback_resume_message(resume_value)


def validate_wait_resume_value(payload_raw: Any, resume_value: Any) -> bool:
    """校验 resume 值是否符合等待态契约。"""
    payload = normalize_wait_payload(payload_raw)
    if not payload:
        return False

    kind = payload.get("kind")
    if kind == "confirm":
        return (
            resume_value == payload.get("confirm_resume_value")
            or resume_value == payload.get("cancel_resume_value")
        )

    if kind == "select":
        for option in list(payload.get("options") or []):
            if not isinstance(option, dict):
                continue
            if resume_value == option.get("resume_value"):
                return True
        return False

    if kind == "input_text":
        if not isinstance(resume_value, dict):
            return False
        response_key = str(payload.get("response_key") or "message").strip()
        candidate = resume_value.get(response_key)
        if not isinstance(candidate, str):
            return False
        if candidate.strip():
            return True
        return bool(payload.get("allow_empty"))

    return False


def _fallback_resume_message(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("message", "text", "answer", "input"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    if value is None:
        return ""
    return json.dumps(value, ensure_ascii=False, default=str)
