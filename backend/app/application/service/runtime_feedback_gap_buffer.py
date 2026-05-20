#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""当前 runtime 调用内的 transient feedback gap buffer。"""

from __future__ import annotations

from app.domain.models.feedback import FeedbackGapResult

_DEFAULT_MAX_GAPS = 20


class RuntimeFeedbackGapBuffer:
    """只在当前 runner 实例内保存 gap，不落库、不跨请求恢复。"""

    def __init__(self, *, max_gaps: int = _DEFAULT_MAX_GAPS) -> None:
        self._max_gaps = max(1, int(max_gaps))
        self._gaps: list[FeedbackGapResult] = []

    def append_feedback_gap(self, gap: FeedbackGapResult) -> None:
        gap_key = self._gap_key(gap)
        if any(self._gap_key(existing) == gap_key for existing in self._gaps):
            return
        if len(self._gaps) >= self._max_gaps:
            return
        self._gaps.append(gap)

    def get_feedback_gaps(self) -> list[FeedbackGapResult]:
        return list(self._gaps)

    @staticmethod
    def _gap_key(gap: FeedbackGapResult) -> tuple[str, str, str | None, str | None]:
        source_event_id = gap.source_ref.source_event_id if gap.source_ref is not None else None
        target_id = gap.target_ref.target_id if gap.target_ref is not None else None
        return (
            gap.gap_kind.value,
            gap.reason_code.value,
            source_event_id,
            target_id,
        )
