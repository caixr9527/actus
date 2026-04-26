#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/1/26 15:33
@Author : caixiaorong01@outlook.com
@File   : message.py
"""
from typing import Any, Dict, Optional, Union, List

from .base import BaseTool, tool
from ...models import ToolResult
from ...models.wait import (
    ConfirmWaitPayload,
    SelectWaitPayload,
    TextInputWaitPayload,
    WaitChoice,
)


class MessageTool(BaseTool):
    name: str = "message"

    def __init__(self) -> None:
        super().__init__()

    @tool(
        name="message_notify_user",
        description="向用户发送消息，且无需用户回复。用于确认收到消息、提供进度更新、报告任务完成情况，或解释处理方式的变更。",
        parameters={
            "text": {
                "type": "string",
                "description": "要显示给用户的消息文本",
            },
        },
        required=["text"]
    )
    async def message_notify_user(self, text: str) -> ToolResult:
        """发送通知消息给用户，不需要用户响应"""
        return ToolResult(success=True, data="Continue")

    @tool(
        name="message_ask_user",
        description="向用户提问并等待回复。用于：请求澄清、寻求确认、或收集额外信息。",
        parameters={
            "text": {
                "type": "string",
                "description": "要展示给用户的问题文本",
            },
            "kind": {
                "type": "string",
                "enum": ["input_text", "confirm", "select"],
                "description": "等待用户输入的交互类型。",
            },
            "title": {
                "type": "string",
                "description": "(可选)等待卡片标题。",
            },
            "details": {
                "type": "string",
                "description": "(可选)补充说明，会显示在主问题下方。",
            },
            "attachments": {
                "anyOf": [
                    {"type": "string"},
                    {"items": {"type": "string"}, "type": "array"},
                ],
                "description": "(可选)与问题相关的文件或参考资料",
            },
            "suggest_user_takeover": {
                "type": "string",
                "enum": ["none", "browser"],
                "description": "(可选)建议用户接管的操作（例如由用户在浏览器中手动完成某些事）。"
            },
            "placeholder": {
                "type": "string",
                "description": "(input_text) 输入框占位文案。",
            },
            "submit_label": {
                "type": "string",
                "description": "(input_text) 提交按钮文案。",
            },
            "response_key": {
                "type": "string",
                "description": "(input_text) 恢复时文本会写入该字段名。",
            },
            "default_value": {
                "type": "string",
                "description": "(input_text) 默认填入输入框的文本。",
            },
            "allow_empty": {
                "type": "boolean",
                "description": "(input_text) 是否允许提交空文本。",
            },
            "multiline": {
                "type": "boolean",
                "description": "(input_text) 是否使用多行输入框。",
            },
            "confirm_label": {
                "type": "string",
                "description": "(confirm) 确认按钮文案。",
            },
            "cancel_label": {
                "type": "string",
                "description": "(confirm) 取消按钮文案。",
            },
            "confirm_resume_value": {
                "description": "(confirm) 点击确认后透传给 resume 的值。",
            },
            "cancel_resume_value": {
                "description": "(confirm) 点击取消后透传给 resume 的值。",
            },
            "emphasis": {
                "type": "string",
                "enum": ["default", "destructive"],
                "description": "(confirm) 操作强调样式。",
            },
            "options": {
                "type": "array",
                "description": "(select) 候选项列表。",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "description": {"type": "string"},
                        "resume_value": {},
                    },
                    "required": ["label", "resume_value"],
                },
            },
            "default_resume_value": {
                "description": "(select) 默认选中的恢复值。",
            },
        },
        required=["text"],
    )
    async def message_ask_user(
            self,
            text: str,
            kind: Optional[str] = None,
            title: Optional[str] = None,
            details: Optional[str] = None,
            attachments: Optional[Union[str, List[str]]] = None,
            suggest_user_takeover: Optional[str] = None,
            placeholder: Optional[str] = None,
            submit_label: Optional[str] = None,
            response_key: Optional[str] = None,
            default_value: Optional[str] = None,
            allow_empty: Optional[bool] = None,
            multiline: Optional[bool] = None,
            confirm_label: Optional[str] = None,
            cancel_label: Optional[str] = None,
            confirm_resume_value: Any = True,
            cancel_resume_value: Any = False,
            emphasis: Optional[str] = None,
            options: Optional[List[Dict[str, Any]]] = None,
            default_resume_value: Any = None,
    ) -> ToolResult:
        """提问用户并等待响应"""
        normalized_attachments: List[str]
        if attachments is None:
            normalized_attachments = []
        elif isinstance(attachments, str):
            normalized_attachments = [attachments] if attachments.strip() else []
        else:
            normalized_attachments = [str(item).strip() for item in attachments if str(item).strip()]

        normalized_kind = str(kind or "input_text").strip().lower()
        payload_base = {
            "title": str(title or "").strip(),
            "prompt": str(text or "").strip(),
            "details": str(details or "").strip(),
            "attachments": normalized_attachments,
            "suggest_user_takeover": (
                suggest_user_takeover
                if suggest_user_takeover in {"none", "browser"}
                else "none"
            ),
        }
        interrupt_payload: Dict[str, Any]
        if normalized_kind == "confirm":
            interrupt_payload = ConfirmWaitPayload(
                **payload_base,
                kind="confirm",
                confirm_label=str(confirm_label or "").strip() or "继续",
                cancel_label=str(cancel_label or "").strip() or "取消",
                confirm_resume_value=confirm_resume_value,
                cancel_resume_value=cancel_resume_value,
                emphasis=str(emphasis or "default").strip().lower(),
            ).model_dump(mode="json")
        elif normalized_kind == "select":
            interrupt_payload = SelectWaitPayload(
                **payload_base,
                kind="select",
                options=[
                    WaitChoice.model_validate(option).model_dump(mode="json")
                    for option in list(options or [])
                ],
                default_resume_value=default_resume_value,
            ).model_dump(mode="json")
        else:
            interrupt_payload = TextInputWaitPayload(
                **payload_base,
                kind="input_text",
                placeholder=str(placeholder or "").strip(),
                submit_label=str(submit_label or "").strip() or "继续执行",
                response_key=str(response_key or "").strip() or "message",
                default_value=str(default_value or "").strip(),
                allow_empty=bool(allow_empty) if allow_empty is not None else False,
                multiline=bool(multiline) if multiline is not None else True,
            ).model_dump(mode="json")

        return ToolResult(
            success=True,
            data={
                "interrupt": {
                    **interrupt_payload,
                }
            },
        )
