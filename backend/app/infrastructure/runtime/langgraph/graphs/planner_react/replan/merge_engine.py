#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：重规划合并与漂移过滤引擎。"""

from __future__ import annotations

import re
import uuid
from typing import List, Optional, Tuple

from app.domain.models import Plan, Step
from app.domain.services.runtime.normalizers import normalize_success_criteria


class ReplanMergeEngine:
    """封装 replan 步骤过滤、语义去重与计划合并。"""

    def __init__(
        self,
        *,
        replan_meta_validation_step_pattern: re.Pattern[str],
        replan_meta_validation_allow_pattern: re.Pattern[str],
        replan_meta_validation_deny_pattern: re.Pattern[str],
    ) -> None:
        self._replan_meta_validation_step_pattern = replan_meta_validation_step_pattern
        self._replan_meta_validation_allow_pattern = replan_meta_validation_allow_pattern
        self._replan_meta_validation_deny_pattern = replan_meta_validation_deny_pattern

    def filter_replan_drift_steps(self, new_steps: List[Step], *, user_message: str) -> Tuple[List[Step], int]:
        if self._user_explicitly_requests_tool_validation(user_message):
            return new_steps, 0

        filtered_steps: List[Step] = []
        dropped_count = 0
        for step in new_steps:
            if self._is_replan_meta_validation_step(step):
                dropped_count += 1
                continue
            filtered_steps.append(step)
        return filtered_steps, dropped_count

    def merge_replanned_steps_into_plan(self, plan: Plan, new_steps: List[Step]) -> Tuple[List[Step], str]:
        first_pending_index: Optional[int] = None
        for index, current_step in enumerate(plan.steps):
            if not current_step.done:
                first_pending_index = index
                break

        if first_pending_index is None:
            completed_steps = list(plan.steps)
            merged_steps = list(completed_steps)
            merged_steps.extend(self._dedupe_replanned_steps(completed_steps, new_steps))
            return merged_steps, "append_after_completed_batch"

        preserved_steps = plan.steps[:first_pending_index]
        merged_steps = list(preserved_steps)
        merged_steps.extend(self._dedupe_replanned_steps(preserved_steps, new_steps))
        return merged_steps, "replace_remaining_pending_steps"

    def _dedupe_replanned_steps(self, existing_steps: List[Step], new_steps: List[Step]) -> List[Step]:
        seen_step_ids = {
            str(step.id).strip()
            for step in existing_steps
            if str(step.id).strip()
        }
        deduped_steps: List[Step] = []
        seen_semantic_signatures = {
            self._build_step_semantic_signature(step)
            for step in existing_steps
        }
        for step in new_steps:
            semantic_signature = self._build_step_semantic_signature(step)
            if semantic_signature and semantic_signature in seen_semantic_signatures:
                continue
            if semantic_signature:
                seen_semantic_signatures.add(semantic_signature)
            step_id = str(step.id).strip()
            normalized_step = step
            if not step_id or step_id in seen_step_ids:
                # 懒拷贝：只有需要改写 ID 时才深拷贝，避免重规划批量步骤的无效复制开销。
                normalized_step = step.model_copy(deep=True)
                normalized_step.id = str(uuid.uuid4())
                step_id = normalized_step.id
            seen_step_ids.add(step_id)
            deduped_steps.append(normalized_step)
        return deduped_steps

    @staticmethod
    def _build_step_semantic_signature(step: Step) -> str:
        title = str(getattr(step, "title", "") or "").strip().lower()
        description = str(getattr(step, "description", "") or "").strip().lower()
        success_criteria_items = [
            str(item or "").strip().lower()
            for item in normalize_success_criteria(
                getattr(step, "success_criteria", []),
                fallback_description=description,
            )[0]
            if str(item or "").strip()
        ]
        success_criteria = "|".join(sorted(set(success_criteria_items)))
        signature_parts = [title, description, success_criteria]
        return "||".join([item for item in signature_parts if item]).strip()

    def _user_explicitly_requests_tool_validation(self, user_message: str) -> bool:
        normalized_message = str(user_message or "").strip()
        if not normalized_message:
            return False
        if self._replan_meta_validation_deny_pattern.search(normalized_message):
            return False
        return bool(self._replan_meta_validation_allow_pattern.search(normalized_message))

    def _is_replan_meta_validation_step(self, step: Step) -> bool:
        normalized_success_criteria, _ = normalize_success_criteria(
            getattr(step, "success_criteria", []),
            fallback_description=str(getattr(step, "description", "") or "").strip(),
        )
        candidate_text = " ".join(
            [
                str(getattr(step, "title", "") or "").strip(),
                str(getattr(step, "description", "") or "").strip(),
                " ".join(
                    [
                        str(item).strip()
                        for item in normalized_success_criteria
                        if str(item).strip()
                    ]
                ),
            ]
        ).strip()
        if not candidate_text:
            return False
        return bool(self._replan_meta_validation_step_pattern.search(candidate_text))
