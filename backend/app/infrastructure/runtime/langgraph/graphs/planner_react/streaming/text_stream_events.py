#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""构造回放式文本流事件。

业务约束：
- 本模块只负责把已经生成完成的文本拆成 `text_stream_*` 临时事件；
- 这些事件只用于当前 SSE 连接的草稿展示，不写入 graph state 的 `emitted_events`；
- 真正的历史真相源仍然是后续的 `PlanEvent` / `MessageEvent`。
"""

from __future__ import annotations

from typing import List, Optional, Literal

from app.domain.models import (
    BaseEvent,
    TextStreamChannel,
    TextStreamDeltaEvent,
    TextStreamEndEvent,
    TextStreamStartEvent,
)
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState

DEFAULT_CHUNK_SIZE = 120
MAX_CHUNK_SIZE = 200
MAX_CHUNK_COUNT = 120
TEXT_STREAM_BREAKPOINTS = ("\n\n", "\n", "。", "；", "：", "，", ",", ";", ":")


def build_text_stream_id(
        *,
        channel: TextStreamChannel,
        state: PlannerReActLangGraphState,
) -> str:
    """构造单次文本流 ID，用于前端按 stream 归并 delta。"""
    run_id = str(state.get("run_id") or "").strip()
    session_id = str(state.get("session_id") or "").strip()
    thread_id = str(state.get("thread_id") or "").strip()
    scope = run_id or session_id or thread_id or "runtime"
    return f"{scope}:{channel.value}"


def _find_split_index(window: str) -> int:
    """在目标窗口内优先按自然语言断点切分，避免把中文句子硬切得过碎。"""
    best_index = -1
    for breakpoint in TEXT_STREAM_BREAKPOINTS:
        index = window.rfind(breakpoint)
        if index > best_index:
            best_index = index + len(breakpoint)
    return best_index


def split_text_for_stream(
        text: str,
        *,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_chunk_size: int = MAX_CHUNK_SIZE,
        max_chunk_count: int = MAX_CHUNK_COUNT,
) -> List[str]:
    """把完整文本拆成有限数量的回放片段。"""
    normalized_text = str(text or "")
    if not normalized_text:
        return []

    safe_chunk_size = max(1, min(int(chunk_size), int(max_chunk_size)))
    if len(normalized_text) / safe_chunk_size > max_chunk_count:
        # 极长文本下优先保护 SSE 事件数量，避免大量 delta 拖慢前端重绘。
        safe_chunk_size = max(safe_chunk_size, (len(normalized_text) // max_chunk_count) + 1)

    chunks: List[str] = []
    cursor = 0
    while cursor < len(normalized_text):
        hard_end = min(len(normalized_text), cursor + safe_chunk_size)
        if hard_end >= len(normalized_text):
            chunks.append(normalized_text[cursor:])
            break

        soft_window = normalized_text[cursor:hard_end]
        split_index = _find_split_index(soft_window)
        if split_index <= 0:
            split_index = min(len(normalized_text) - cursor, int(max_chunk_size))
        chunk = normalized_text[cursor: cursor + split_index]
        if not chunk:
            chunk = normalized_text[cursor:hard_end]
        chunks.append(chunk)
        cursor += len(chunk)

    return chunks


def build_text_stream_events(
        *,
        channel: TextStreamChannel,
        text: str,
        state: PlannerReActLangGraphState,
        stage: Literal["planner", "summary", "final"],
        stream_id: Optional[str] = None,
) -> List[BaseEvent]:
    """构造完整的 start/delta/end 文本流事件序列。"""
    chunks = split_text_for_stream(text)
    if not chunks:
        return []

    resolved_stream_id = stream_id or build_text_stream_id(channel=channel, state=state)
    events: List[BaseEvent] = [
        TextStreamStartEvent(
            stream_id=resolved_stream_id,
            channel=channel,
            run_id=str(state.get("run_id") or "").strip() or None,
            session_id=str(state.get("session_id") or "").strip() or None,
            stage=stage,
            is_replay=True,
        )
    ]
    events.extend(
        TextStreamDeltaEvent(
            stream_id=resolved_stream_id,
            channel=channel,
            text=chunk,
            sequence=index,
        )
        for index, chunk in enumerate(chunks, start=1)
    )
    events.append(
        TextStreamEndEvent(
            stream_id=resolved_stream_id,
            channel=channel,
            full_text_length=len(str(text or "")),
            reason="completed",
        )
    )
    return events
