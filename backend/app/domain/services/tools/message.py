#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/1/26 15:33
@Author : caixiaorong01@outlook.com
@File   : message.py
"""
from typing import Optional, Union, List

from .base import BaseTool, tool
from ...models import ToolResult


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
        },
        required=["text"],
    )
    async def message_ask_user(
            self,
            text: str,
            attachments: Optional[Union[str, List[str]]] = None,
            suggest_user_takeover: Optional[str] = None,
    ) -> ToolResult:
        """提问用户并等待响应"""
        normalized_attachments: List[str]
        if attachments is None:
            normalized_attachments = []
        elif isinstance(attachments, str):
            normalized_attachments = [attachments] if attachments.strip() else []
        else:
            normalized_attachments = [str(item).strip() for item in attachments if str(item).strip()]

        return ToolResult(
            success=True,
            data={
                "interrupt": {
                    "kind": "ask_user",
                    "question": str(text or "").strip(),
                    "attachments": normalized_attachments,
                    "suggest_user_takeover": (
                        suggest_user_takeover
                        if suggest_user_takeover in {"none", "browser"}
                        else "none"
                    ),
                }
            },
        )
