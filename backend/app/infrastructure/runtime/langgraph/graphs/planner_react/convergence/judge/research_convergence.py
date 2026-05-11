#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研究链路执行收敛判定。

本模块只处理 research 单步执行期间的“是否已有足够证据可以停止工具循环”。
它不负责约束决策、不写反馈层，也不改变 planner/replan 的业务语义。
"""

from typing import Any, Dict, List, Optional

from app.domain.models import Step, StepArtifactPolicy, StepOutputMode
from app.domain.services.runtime.normalizers import normalize_controlled_value
from app.infrastructure.runtime.langgraph.graphs.planner_react.convergence.contracts import ConvergenceDecision
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.execution_state import (
    ExecutionState,
)


class ResearchConvergenceJudge:
    """基于已成立的研究证据，决定研究步骤是否应停止工具循环。"""

    def evaluate_after_iteration(
            self,
            *,
            step: Step,
            task_mode: str,
            recent_function_name: str,
            execution_state: ExecutionState,
    ) -> ConvergenceDecision:
        """每轮 search/fetch 后执行的收敛判定。

        业务语义：
        - search_web 的 snippet 只能作为候选来源信号，不允许单独决定 strong completion；
        - research 步骤只有在 runtime 显式携带 evidence-backed 信号时，才允许轻量收敛；
        - web_reading 不能使用该判定器直接收敛，必须交给页面阅读证据链路处理。
        """
        if str(task_mode or "").strip().lower() != "research":
            return ConvergenceDecision(should_break=False)
        normalized_function_name = str(recent_function_name or "").strip().lower()
        if normalized_function_name not in {"search_web", "fetch_page"}:
            return ConvergenceDecision(should_break=False)
        if _step_requires_file_output(step):
            return ConvergenceDecision(should_break=False)
        if _has_current_step_research_evidence(execution_state.runtime_recent_action):
            reason_code = "research_evidence_ready"
            return ConvergenceDecision(
                should_break=True,
                payload=self._build_evidence_payload(
                    step=step,
                    execution_state=execution_state,
                    reason_code=reason_code,
                ),
                reason_code=reason_code,
            )

        return ConvergenceDecision(should_break=False)

    @staticmethod
    def build_max_iteration_convergence_payload(
            *,
            step: Step,
            task_mode: str,
            runtime_recent_action: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """达到最大轮次时，不基于搜索摘要降级为成功收敛。"""
        if str(task_mode or "").strip().lower() != "research":
            return None
        recent_action = dict(runtime_recent_action or {})
        if not _has_current_step_research_evidence(recent_action):
            return None
        return ResearchConvergenceJudge._build_payload_from_evidence(
            step=step,
            runtime_recent_action=recent_action,
            reason_code="research_max_iteration_evidence_ready",
        )

    @staticmethod
    def _build_evidence_payload(
            *,
            step: Step,
            execution_state: ExecutionState,
            reason_code: str,
    ) -> Dict[str, Any]:
        return ResearchConvergenceJudge._build_payload_from_evidence(
            step=step,
            runtime_recent_action=execution_state.runtime_recent_action,
            reason_code=reason_code,
        )

    @staticmethod
    def _build_payload_from_evidence(
            *,
            step: Step,
            runtime_recent_action: Optional[Dict[str, Any]],
            reason_code: str,
    ) -> Dict[str, Any]:
        evidence_lines = _extract_evidence_backed_lines(dict(runtime_recent_action or {}))
        summary = f"当前研究步骤已基于结构化 evidence 完成：{step.description}"
        facts_learned = [
            line for line in evidence_lines
            if str(line).strip()
        ]
        runtime_action = dict(runtime_recent_action or {})
        runtime_action["research_convergence"] = {
            "reason_code": reason_code,
            "evidence_count": len(evidence_lines),
        }
        return {
            "success": True,
            "summary": summary,
            "result": summary,
            "attachments": [],
            "blockers": [],
            # Phase C：步骤收敛只保留结构化证据，不再产出步骤级正文。
            "facts_learned": facts_learned,
            "open_questions": [],
            "next_hint": "",
            "runtime_recent_action": runtime_action,
        }


def _has_current_step_research_evidence(runtime_recent_action: Dict[str, Any]) -> bool:
    """只消费显式 evidence-backed 信号，不从搜索摘要反推 evidence。"""
    return len(_extract_evidence_backed_lines(runtime_recent_action)) > 0


def _extract_evidence_backed_lines(runtime_recent_action: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    for item in list(runtime_recent_action.get("evidence_backed_facts") or []):
        text = _extract_projection_text(item)
        if text:
            lines.append(text)
    research_progress = runtime_recent_action.get("research_progress")
    if isinstance(research_progress, dict):
        for item in list(research_progress.get("evidence_backed_facts") or []):
            text = _extract_projection_text(item)
            if text:
                lines.append(text)
    return lines


def _extract_projection_text(item: Any) -> str:
    if isinstance(item, dict):
        return str(item.get("text") or item.get("summary") or "").strip()
    return str(getattr(item, "text", "") or getattr(item, "summary", "") or "").strip()


def _step_requires_file_output(step: Step) -> bool:
    output_mode = normalize_controlled_value(getattr(step, "output_mode", None), StepOutputMode)
    artifact_policy = normalize_controlled_value(getattr(step, "artifact_policy", None), StepArtifactPolicy)
    candidate_text = _build_step_candidate_text(step)
    return (
            output_mode == StepOutputMode.FILE.value
            or artifact_policy == StepArtifactPolicy.REQUIRE_FILE_OUTPUT.value
            or "保存" in candidate_text
            or "写入" in candidate_text
            or "输出文件" in candidate_text
            or "save" in candidate_text.lower()
    )


def _build_step_candidate_text(step: Step) -> str:
    return " ".join(
        [
            str(getattr(step, "title", "") or ""),
            str(getattr(step, "description", "") or ""),
            " ".join([str(item or "") for item in list(getattr(step, "success_criteria", []) or [])]),
        ]
    )
