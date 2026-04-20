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
            function_args: Dict[str, Any],
            tool_result_data: Any,
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
            function_args=function_args,
            tool_result_data=tool_result_data,
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
    def _accumulate_file_context(
            *,
            context: Dict[str, Any],
            function_name: str,
            function_args: Dict[str, Any],
            tool_result_data: Any,
    ) -> None:
        called_functions = set(context.get("called_functions") or set())
        normalized_function_name = str(function_name or "").strip().lower()
        called_functions.add(normalized_function_name)
        context["called_functions"] = called_functions
        # P3-一次性收口：基于工具事实沉淀可交付上下文，避免模板化/幻觉收口。
        if normalized_function_name in {"list_files", "find_files"}:
            if isinstance(tool_result_data, dict):
                dir_path = str(
                    tool_result_data.get("dir_path")
                    or function_args.get("dir_path")
                    or ""
                ).strip()
                if dir_path:
                    context["last_dir_path"] = dir_path
                files = tool_result_data.get("files")
                if isinstance(files, list):
                    normalized_files = [str(item).strip() for item in files if str(item).strip()]
                    if normalized_files:
                        context["listed_files"] = normalized_files[:12]
        elif normalized_function_name == "read_file":
            if isinstance(tool_result_data, dict):
                filepath = str(
                    tool_result_data.get("filepath")
                    or function_args.get("filepath")
                    or ""
                ).strip()
                content = str(tool_result_data.get("content") or "").strip()
                if filepath:
                    context["last_read_file"] = filepath
                if content:
                    # P3-一次性收口：只记录长度等非敏感事实，不记录正文预览。
                    context["last_read_content_length"] = len(content)
        elif normalized_function_name in {"write_file", "replace_in_file"}:
            filepath = ""
            if isinstance(tool_result_data, dict):
                filepath = str(
                    tool_result_data.get("filepath")
                    or function_args.get("filepath")
                    or ""
                ).strip()
            else:
                filepath = str(function_args.get("filepath") or "").strip()
            if filepath:
                written_files = list(context.get("written_files") or [])
                if filepath not in written_files:
                    written_files.append(filepath)
                context["written_files"] = written_files[:8]
        elif normalized_function_name == "check_file_exists":
            filepath = str(function_args.get("file_path") or function_args.get("filepath") or "").strip()
            if filepath:
                context["last_checked_file"] = filepath

    @staticmethod
    def _has_minimal_file_progress(context: Dict[str, Any]) -> bool:
        # P3-一次性收口：收敛判定基于“事实信号”而非固定函数组合，避免场景化补丁。
        called_functions = set(context.get("called_functions") or set())
        has_observation = bool(
            str(context.get("last_dir_path") or "").strip()
            or list(context.get("listed_files") or [])
            or str(context.get("last_checked_file") or "").strip()
            # 调用痕迹只做兜底信号，不作为主判定真相源。
            or bool({"list_files", "find_files", "check_file_exists"}.intersection(called_functions))
        )
        has_delivery_evidence = bool(
            str(context.get("last_read_file") or "").strip()
            or list(context.get("written_files") or [])
            or bool({"read_file", "write_file", "replace_in_file"}.intersection(called_functions))
        )
        return has_observation and has_delivery_evidence

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
        listed_files = list(context.get("listed_files") or [])
        written_files = list(context.get("written_files") or [])
        last_dir_path = str(context.get("last_dir_path") or "").strip()
        last_read_file = str(context.get("last_read_file") or "").strip()
        last_read_content_length = int(context.get("last_read_content_length") or 0)
        last_checked_file = str(context.get("last_checked_file") or "").strip()

        lines: List[str] = [f"步骤已按工具事实完成并收敛成功：{step.description}。"]
        if last_dir_path:
            lines.append(f"当前目录：{last_dir_path}")
        if listed_files:
            lines.append("目录文件：" + "、".join(listed_files[:8]))
        if written_files:
            lines.append("已写入文件：" + "、".join(written_files[:5]))
        if last_checked_file:
            lines.append(f"已校验文件存在：{last_checked_file}")
        if last_read_file:
            lines.append(f"已读取文件：{last_read_file}")
        if last_read_content_length > 0:
            lines.append(f"读取内容长度：{last_read_content_length} 字符")
        if called_functions:
            lines.append("已执行工具：" + "、".join(called_functions[:6]))
        return "\n".join(lines)
