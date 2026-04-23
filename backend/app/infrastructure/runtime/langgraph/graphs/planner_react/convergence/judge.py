#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 解耦：步骤收敛判定器。"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

from app.domain.models import Step, StepArtifactPolicy, StepOutputMode
from app.domain.services.runtime.normalizers import normalize_controlled_value

_SIMPLE_FILE_ACTION_PATTERN = re.compile(
    r"(创建|写入|读取|读出|列出|保存|生成文件|create|write|read|list|save)",
    re.IGNORECASE,
)
_FILE_TARGET_PATTERN = re.compile(
    r"(文件|目录|file|directory|folder|/[^\s，。；;]+|\b[\w.-]+\.(?:txt|md|json|csv|yaml|yml|log|py|js|ts|tsx|html|css)\b)",
    re.IGNORECASE,
)
_COMPLEX_CODING_TASK_PATTERN = re.compile(
    r"(实现|开发|修复|重构|测试|运行测试|编译|构建|接口|服务|组件|模块|算法|依赖|install|implement|develop|fix|refactor|test|build|compile)",
    re.IGNORECASE,
)
_RAW_CONTENT_RETURN_PATTERN = re.compile(
    r"(原样返回|原样输出|直接返回|直接输出|返回内容|输出内容|显示内容|原文返回|raw\s+content|return\s+content|print\s+content)",
    re.IGNORECASE,
)


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
        if task_mode not in {"file_processing", "coding"}:
            return ConvergenceEvaluationResult(should_break=False)
        if not tool_result_success:
            return ConvergenceEvaluationResult(should_break=False)
        if task_mode == "coding" and not self._is_simple_coding_file_task(step):
            return ConvergenceEvaluationResult(should_break=False)

        self._accumulate_file_context(
            context=step_file_context,
            function_name=recent_function_name,
            function_args=function_args,
            tool_result_data=tool_result_data,
        )

        if not self._step_allows_file_fact_convergence(step):
            return ConvergenceEvaluationResult(should_break=False)
        if not self._has_minimal_file_progress(step_file_context):
            return ConvergenceEvaluationResult(should_break=False)
        raw_content_payload = self._build_raw_file_content_payload_if_applicable(
            step=step,
            recent_function_name=recent_function_name,
            tool_result_data=tool_result_data,
            runtime_recent_action=runtime_recent_action,
        )
        if raw_content_payload is not None:
            return ConvergenceEvaluationResult(
                should_break=True,
                payload=raw_content_payload,
                reason_code="file_processing_raw_content_ready",
            )

        summary = self._build_file_processing_convergence_summary(step=step, context=step_file_context)
        payload = {
            "success": True,
            "summary": summary,
            "result": summary,
            "attachments": [],
            "blockers": [],
            "facts_learned": [summary],
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
        if task_mode not in {"file_processing", "coding"}:
            return None
        if task_mode == "coding" and not ConvergenceJudge._is_simple_coding_file_task(step):
            return None
        if not ConvergenceJudge._step_allows_file_fact_convergence(step):
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
            "attachments": [],
            "blockers": [],
            "facts_learned": [summary],
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
            content = str(function_args.get("content") or "").strip()
            if content:
                context["last_written_content_length"] = len(content)
        elif normalized_function_name == "check_file_exists":
            filepath = str(function_args.get("file_path") or function_args.get("filepath") or "").strip()
            if filepath:
                context["last_checked_file"] = filepath

    @staticmethod
    def _has_minimal_file_progress(context: Dict[str, Any]) -> bool:
        # P3-一次性收口：收敛判定基于“事实信号”而非固定函数组合，避免场景化补丁。
        called_functions = set(context.get("called_functions") or set())
        has_written_file = bool(list(context.get("written_files") or []))
        has_read_file = bool(str(context.get("last_read_file") or "").strip())
        has_observation = bool(
            str(context.get("last_dir_path") or "").strip()
            or list(context.get("listed_files") or [])
            or str(context.get("last_checked_file") or "").strip()
            # 调用痕迹只做兜底信号，不作为主判定真相源。
            or bool({"list_files", "find_files", "check_file_exists"}.intersection(called_functions))
        )
        has_delivery_evidence = bool(
            has_read_file
            or has_written_file
            or bool({"read_file", "write_file", "replace_in_file"}.intersection(called_functions))
        )
        return (has_observation and has_delivery_evidence) or has_read_file or has_written_file

    @staticmethod
    def _is_inline_non_file_required_step(step: Step) -> bool:
        _ = step
        return False

    @staticmethod
    def _step_allows_file_fact_convergence(step: Step) -> bool:
        """文件事实收敛只基于执行事实，不依赖任何已废弃的步骤交付角色语义。"""
        output_mode = normalize_controlled_value(getattr(step, "output_mode", None), StepOutputMode)
        artifact_policy = normalize_controlled_value(getattr(step, "artifact_policy", None), StepArtifactPolicy)
        return output_mode in {"", StepOutputMode.NONE.value} and artifact_policy != StepArtifactPolicy.REQUIRE_FILE_OUTPUT.value

    @staticmethod
    def _is_simple_coding_file_task(step: Step) -> bool:
        """只让简单文件读写类 coding 任务走确定性早停，避免复杂编码任务过早收敛。"""
        candidate_text = " ".join(
            [
                str(getattr(step, "title", "") or ""),
                str(getattr(step, "description", "") or ""),
                " ".join([str(item or "") for item in list(getattr(step, "success_criteria", []) or [])]),
            ]
        ).strip()
        if not candidate_text:
            return False
        is_simple_file_task = bool(
            _SIMPLE_FILE_ACTION_PATTERN.search(candidate_text)
            and _FILE_TARGET_PATTERN.search(candidate_text)
        )
        if _COMPLEX_CODING_TASK_PATTERN.search(candidate_text) and not is_simple_file_task:
            return False
        return is_simple_file_task

    @staticmethod
    def _build_file_processing_convergence_summary(*, step: Step, context: Dict[str, Any]) -> str:
        called_functions = sorted(set(context.get("called_functions") or set()))
        listed_files = list(context.get("listed_files") or [])
        written_files = list(context.get("written_files") or [])
        last_dir_path = str(context.get("last_dir_path") or "").strip()
        last_read_file = str(context.get("last_read_file") or "").strip()
        last_read_content_length = int(context.get("last_read_content_length") or 0)
        last_written_content_length = int(context.get("last_written_content_length") or 0)
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
        if last_written_content_length > 0:
            lines.append(f"写入内容长度：{last_written_content_length} 字符")
        if called_functions:
            lines.append("已执行工具：" + "、".join(called_functions[:6]))
        return "\n".join(lines)

    @staticmethod
    def _build_raw_file_content_payload_if_applicable(
            *,
            step: Step,
            recent_function_name: str,
            tool_result_data: Any,
            runtime_recent_action: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """显式要求“原样返回文件内容”时，直接把 read_file 正文作为交付结果。"""
        if str(recent_function_name or "").strip().lower() != "read_file":
            return None
        candidate_text = " ".join(
            [
                str(getattr(step, "title", "") or ""),
                str(getattr(step, "description", "") or ""),
                " ".join([str(item or "") for item in list(getattr(step, "success_criteria", []) or [])]),
            ]
        ).strip()
        if not candidate_text or not _RAW_CONTENT_RETURN_PATTERN.search(candidate_text):
            return None
        if not isinstance(tool_result_data, dict):
            return None
        content = str(tool_result_data.get("content") or "")
        filepath = str(tool_result_data.get("filepath") or "").strip()
        if not content:
            return None
        return {
            "success": True,
            "summary": f"已读取并原样返回文件内容：{filepath or step.description}",
            "result": content,
            "attachments": [],
            "blockers": [],
            "facts_learned": [
                f"已读取文件：{filepath or step.description}",
                f"已获取原始内容，长度 {len(content)} 字符",
            ],
            "open_questions": [],
            "next_hint": "",
            "runtime_recent_action": runtime_recent_action or {},
        }
