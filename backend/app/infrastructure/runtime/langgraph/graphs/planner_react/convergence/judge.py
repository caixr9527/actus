#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：步骤收敛判定器。"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.domain.models import Step, StepArtifactPolicy, StepDeliveryRole, StepOutputMode
from app.domain.services.runtime.normalizers import normalize_controlled_value


@dataclass(slots=True)
class ConvergenceEvaluationResult:
    """步骤收敛判定结果。"""

    should_break: bool
    payload: Optional[Dict[str, Any]] = None
    reason_code: str = ""


class ConvergenceJudge:
    """基于关键事实的早停判定，避免已达成目标仍空转。"""

    def evaluate_file_processing_progress(
        self,
        *,
        step: Step,
        task_mode: str,
        recent_function_name: str,
        tool_result_success: bool,
        step_file_context: Dict[str, Any],
        runtime_recent_action: Optional[Dict[str, Any]],
    ) -> ConvergenceEvaluationResult:
        """在文件处理任务中判定“已满足事实，直接收敛成功”。"""
        if task_mode != "file_processing":
            return ConvergenceEvaluationResult(should_break=False)
        if not tool_result_success:
            return ConvergenceEvaluationResult(should_break=False)

        self._accumulate_file_context(
            context=step_file_context,
            function_name=recent_function_name,
        )

        if not self._is_file_processing_inline_final_step(step):
            return ConvergenceEvaluationResult(should_break=False)
        if not self._has_minimal_file_progress(step_file_context):
            return ConvergenceEvaluationResult(should_break=False)

        summary = self._build_file_processing_convergence_summary(step=step, context=step_file_context)
        payload = {
            "success": True,
            "summary": summary,
            "result": summary,
            "delivery_text": summary,
            "attachments": [],
            "blockers": [],
            "open_questions": [],
            "next_hint": "",
            "runtime_recent_action": runtime_recent_action or {},
        }
        return ConvergenceEvaluationResult(
            should_break=True,
            payload=payload,
            reason_code="file_processing_facts_ready",
        )

    @staticmethod
    def build_max_iteration_convergence_payload(
        *,
        step: Step,
        task_mode: str,
        runtime_recent_action: Optional[Dict[str, Any]],
        step_file_context: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """在达到最大轮次时，优先尝试按事实收敛，避免误判失败。"""
        if task_mode != "file_processing":
            return None
        if not ConvergenceJudge._is_file_processing_inline_final_step(step):
            return None
        if not ConvergenceJudge._has_minimal_file_progress(step_file_context):
            return None
        summary = ConvergenceJudge._build_file_processing_convergence_summary(
            step=step,
            context=step_file_context,
        )
        return {
            "success": True,
            "summary": summary,
            "result": summary,
            "delivery_text": summary,
            "attachments": [],
            "blockers": [],
            "open_questions": [],
            "next_hint": "",
            "runtime_recent_action": runtime_recent_action or {},
        }

    @staticmethod
    def _accumulate_file_context(*, context: Dict[str, Any], function_name: str) -> None:
        called_functions = set(context.get("called_functions") or set())
        called_functions.add(str(function_name or "").strip().lower())
        context["called_functions"] = called_functions

    @staticmethod
    def _has_minimal_file_progress(context: Dict[str, Any]) -> bool:
        called_functions = set(context.get("called_functions") or set())
        has_listing = "list_files" in called_functions
        has_materialized_file = bool({"write_file", "read_file", "replace_in_file"}.intersection(called_functions))
        return has_listing and has_materialized_file

    @staticmethod
    def _is_file_processing_inline_final_step(step: Step) -> bool:
        output_mode = normalize_controlled_value(getattr(step, "output_mode", None), StepOutputMode)
        delivery_role = normalize_controlled_value(getattr(step, "delivery_role", None), StepDeliveryRole)
        artifact_policy = normalize_controlled_value(getattr(step, "artifact_policy", None), StepArtifactPolicy)
        return (
            output_mode == StepOutputMode.INLINE.value
            and delivery_role == StepDeliveryRole.FINAL.value
            and artifact_policy != StepArtifactPolicy.REQUIRE_FILE_OUTPUT.value
        )

    @staticmethod
    def _build_file_processing_convergence_summary(*, step: Step, context: Dict[str, Any]) -> str:
        called_functions = sorted(set(context.get("called_functions") or set()))
        function_summary = "、".join(called_functions[:6]) if called_functions else "list_files/read_file/write_file"
        return (
            f"已完成关键文件操作并具备目录事实，步骤收敛成功：{step.description}。"
            f" 已执行工具: {function_summary}。"
        )

