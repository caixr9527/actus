#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通用 general 步骤收敛判定。

本模块在 `general` 模式下只保留“文件观察类”的执行收敛：
- 当目录/文件观察事实已经充分时，可以停止继续调工具；
- 其他 general 步骤不再基于“整理/展示/总结”语义提前收敛。

它不生成反馈经验、不修改计划，也不替代最终 summary 节点。
"""

import re
from typing import Any, Dict, List, Optional

from app.domain.models import Step
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.contracts import ConvergenceDecision

_GENERAL_FILE_OBSERVATION_PATTERN = re.compile(
    r"(目录|文件列表|文件名|当前目录|目录状态|列出文件|查看目录|list files|directory|folder|current directory|file names)",
    re.IGNORECASE,
)


class GeneralConvergenceJudge:
    """基于执行事实判断 general 步骤是否应停止继续调工具。"""

    def evaluate_after_iteration(
            self,
            *,
            step: Step,
            task_mode: str,
            runtime_recent_action: Optional[Dict[str, Any]],
            iteration: int,
    ) -> ConvergenceDecision:
        """每轮工具调用后判断 general 步骤是否已具备无工具完成条件。"""
        _ = iteration
        if str(task_mode or "").strip().lower() != "general":
            return ConvergenceDecision(should_break=False)
        if not self._is_file_observation_step(step):
            return ConvergenceDecision(should_break=False)

        file_evidence_lines = self._collect_file_evidence_lines(runtime_recent_action)
        if len(file_evidence_lines) == 0:
            return ConvergenceDecision(should_break=False)
        return ConvergenceDecision(
            should_break=True,
            payload=self._build_payload(
                step=step,
                runtime_recent_action=runtime_recent_action,
                evidence_lines=file_evidence_lines,
                reason_code="general_file_observation_ready",
            ),
            reason_code="general_file_observation_ready",
        )

    @staticmethod
    def build_max_iteration_convergence_payload(
            *,
            step: Step,
            task_mode: str,
            runtime_recent_action: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """最大轮次兜底：只有文件观察类 general 才允许基于已有事实成功收敛。"""
        judge = GeneralConvergenceJudge()
        result = judge.evaluate_after_iteration(
            step=step,
            task_mode=task_mode,
            runtime_recent_action=runtime_recent_action,
            iteration=-1,
        )
        return result.payload if result.should_break else None

    @staticmethod
    def _is_file_observation_step(step: Step) -> bool:
        candidate_text = " ".join(
            [
                str(getattr(step, "title", "") or ""),
                str(getattr(step, "description", "") or ""),
                " ".join([str(item or "") for item in list(getattr(step, "success_criteria", []) or [])]),
            ]
        ).strip()
        if not candidate_text:
            return False
        return bool(_GENERAL_FILE_OBSERVATION_PATTERN.search(candidate_text))

    @staticmethod
    def _collect_file_evidence_lines(runtime_recent_action: Optional[Dict[str, Any]]) -> List[str]:
        recent_action = dict(runtime_recent_action or {})
        evidence_lines: List[str] = []
        snapshot = dict(recent_action.get("file_observation_snapshot") or {})
        if snapshot:
            function_name = str(snapshot.get("function_name") or "").strip().lower()
            data = {
                "dir_path": snapshot.get("dir_path"),
                "files": list(snapshot.get("files") or []) if isinstance(snapshot.get("files"), list) else [],
            }
        else:
            last_success = dict(recent_action.get("last_successful_tool_call") or {})
            function_name = str(last_success.get("function_name") or "").strip().lower()
            data = last_success.get("data")
        if function_name == "list_files" and isinstance(data, dict):
            dir_path = data.get("dir_path")
            raw_files = data.get("files")
            file_names: List[str] = []
            if isinstance(raw_files, list):
                for item in raw_files:
                    if isinstance(item, dict):
                        name = str(item.get("name") or item.get("filename") or "").strip()
                    else:
                        name = str(item or "").strip()
                    if name:
                        file_names.append(name)
            if dir_path:
                evidence_lines.append(f"当前目录：{dir_path}")
            if file_names:
                evidence_lines.append("文件列表：" + "、".join(file_names))
        if function_name == "find_files" and isinstance(data, dict):
            dir_path = data.get("dir_path")
            raw_files = data.get("files")
            file_names = [str(item.get("name") or item.get("filename") or item or "").strip() for item in
                          list(raw_files or [])]
            file_names = [item for item in file_names if item]
            if dir_path:
                evidence_lines.append(f"搜索目录：{dir_path}")
            if file_names:
                evidence_lines.append("匹配文件：" + "、".join(file_names))
        return _dedupe_non_empty(evidence_lines)

    @staticmethod
    def _build_payload(
            *,
            step: Step,
            runtime_recent_action: Optional[Dict[str, Any]],
            evidence_lines: List[str],
            reason_code: str,
            summary_prefix: str = "当前步骤已基于文件观察事实完成",
    ) -> Dict[str, Any]:
        summary = f"{summary_prefix}：{step.description}"
        runtime_action = dict(runtime_recent_action or {})
        runtime_action["general_convergence"] = {
            "reason_code": reason_code,
            "evidence_count": len(evidence_lines),
        }
        return {
            "success": True,
            "summary": summary,
            "result": summary,
            "attachments": [],
            "blockers": [],
            "facts_learned": evidence_lines,
            "open_questions": [],
            "next_hint": "",
            "runtime_recent_action": runtime_action,
        }


def _dedupe_non_empty(values: List[str]) -> List[str]:
    seen: set[str] = set()
    result: List[str] = []
    for value in values:
        normalized = str(value or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result
