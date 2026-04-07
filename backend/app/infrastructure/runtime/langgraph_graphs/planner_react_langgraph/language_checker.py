#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/4/7 21:15
@Author : caixiaorong01@outlook.com
@File   : language_checker.py
"""
import re
from typing import Any, Dict

EXPLICIT_LANGUAGE_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("zh", (r"\bchinese\b", r"\bmandarin\b", "中文", "汉语", "普通话")),
    ("en", (r"\benglish\b", "英文", "英语")),
    ("ja", (r"\bjapanese\b", "日语", "日本語")),
    ("ko", (r"\bkorean\b", "韩语", "韓語", "한국어")),
    ("fr", (r"\bfrench\b", "法语", "法文", "francais", "français")),
    ("de", (r"\bgerman\b", "德语", "德文", "deutsch")),
    ("es", (r"\bspanish\b", "西班牙语", "西语", "español", "espanol")),
    ("ru", (r"\brussian\b", "俄语", "俄文", "русский")),
)


def infer_working_language_from_message(user_message: Any) -> str:
    normalized_message = str(user_message or "").strip()
    if not normalized_message:
        return "zh"

    normalized_lower = normalized_message.lower()
    for language, patterns in EXPLICIT_LANGUAGE_PATTERNS:
        for pattern in patterns:
            if re.search(pattern, normalized_lower, re.IGNORECASE):
                return language

    if re.search(r"[\u3040-\u30ff]", normalized_message):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", normalized_message):
        return "ko"
    if re.search(r"[\u0400-\u04ff]", normalized_message):
        return "ru"
    if re.search(r"[\u4e00-\u9fff]", normalized_message):
        return "zh"
    if re.search(r"[A-Za-z]", normalized_message):
        return "en"
    return "zh"


def build_direct_path_copy(language: str) -> Dict[str, str]:
    if language == "zh":
        return {
            "direct_answer_fallback": "直接回复",
            "direct_execute_fallback": "直接执行",
            "direct_execute_step_title": "直接执行用户请求",
            "direct_execute_message": "已进入直接执行路径。",
            "direct_wait_fallback": "等待确认后执行",
            "direct_wait_title": "等待用户确认",
            "direct_wait_description": "在继续执行原始任务前，先等待用户确认是否继续",
            "direct_wait_execute_title": "执行原始任务",
            "direct_wait_message": "当前任务需要先确认后再继续执行。",
            "direct_wait_prompt": "继续执行该任务前，请先确认是否允许继续。",
            "direct_wait_confirm_label": "继续",
            "direct_wait_cancel_label": "取消",
        }
    return {
        "direct_answer_fallback": "Direct answer",
        "direct_execute_fallback": "Direct execution",
        "direct_execute_step_title": "Execute the user request directly",
        "direct_execute_message": "Entered direct execution path.",
        "direct_wait_fallback": "Wait for confirmation before execution",
        "direct_wait_title": "Wait for user confirmation",
        "direct_wait_description": "Wait for user confirmation before continuing the original task",
        "direct_wait_execute_title": "Execute the original task",
        "direct_wait_message": "This task requires confirmation before execution can continue.",
        "direct_wait_prompt": "Please confirm whether execution is allowed before continuing this task.",
        "direct_wait_confirm_label": "Continue",
        "direct_wait_cancel_label": "Cancel",
    }
