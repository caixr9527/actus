#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""研究链路执行收敛判定。

本模块只处理 research 单步执行期间的“是否已有足够证据可以停止工具循环”。
它不负责约束决策、不写反馈层，也不改变 planner/replan 的业务语义。
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from app.domain.models import Step
from app.domain.services.workspace_runtime.policies import truncate_tool_text
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.execution_state import (
    ExecutionState,
)


@dataclass(slots=True)
class ResearchConvergenceResult:
    """研究链路收敛结果。"""

    should_break: bool
    payload: Optional[Dict[str, Any]] = None
    reason_code: str = ""


class ResearchConvergenceJudge:
    """基于搜索摘要证据与抓取失败事实，决定研究步骤是否应停止工具循环。"""

    def evaluate_after_iteration(
            self,
            *,
            step: Step,
            task_mode: str,
            recent_function_name: str,
            execution_state: ExecutionState,
    ) -> ResearchConvergenceResult:
        """每轮 search/fetch 后执行的收敛判定。

        业务语义：
        - search_web 的 snippet 是 research 步骤的一等证据；当摘要证据已足够时，不继续强制 fetch；
        - fetch_page 低价值/失败后，如果已有 snippet 证据，则降级基于 snippet 收敛；
        - web_reading 不能使用该判定器直接按 snippet 收敛，必须交给页面阅读证据链路处理。
        """
        if str(task_mode or "").strip().lower() != "research":
            return ResearchConvergenceResult(should_break=False)
        normalized_function_name = str(recent_function_name or "").strip().lower()
        if normalized_function_name not in {"search_web", "fetch_page"}:
            return ResearchConvergenceResult(should_break=False)

        if bool(execution_state.research_snippet_sufficient):
            reason_code = (
                "research_fetch_low_value_snippet_fallback"
                if normalized_function_name == "fetch_page"
                else "research_snippet_evidence_ready"
            )
            return ResearchConvergenceResult(
                should_break=True,
                payload=self._build_snippet_payload(
                    step=step,
                    execution_state=execution_state,
                    reason_code=reason_code,
                ),
                reason_code=reason_code,
            )

        if (
                normalized_function_name == "fetch_page"
                and int(execution_state.consecutive_fetch_failure_count or 0) >= 2
                and len(list(execution_state.research_search_evidence_items or [])) > 0
        ):
            reason_code = "research_fetch_unavailable_snippet_fallback"
            return ResearchConvergenceResult(
                should_break=True,
                payload=self._build_snippet_payload(
                    step=step,
                    execution_state=execution_state,
                    reason_code=reason_code,
                ),
                reason_code=reason_code,
            )

        return ResearchConvergenceResult(should_break=False)

    @staticmethod
    def build_max_iteration_convergence_payload(
            *,
            step: Step,
            task_mode: str,
            runtime_recent_action: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """达到最大轮次时，若已有可用搜索摘要证据，则按阶段性结果成功收敛。"""
        if str(task_mode or "").strip().lower() != "research":
            return None
        recent_action = dict(runtime_recent_action or {})
        evidence_items = _extract_runtime_search_evidence_items(recent_action)
        if len(evidence_items) == 0:
            return None
        diagnosis = dict(recent_action.get("research_diagnosis") or {})
        reason_code = str(diagnosis.get("code") or "")
        if reason_code and reason_code not in {
            "search_snippet_sufficient",
            "fetch_low_value",
            "fetch_unavailable",
        }:
            return None
        return ResearchConvergenceJudge._build_payload_from_items(
            step=step,
            evidence_items=evidence_items,
            runtime_recent_action=recent_action,
            reason_code="research_max_iteration_snippet_fallback",
        )

    @staticmethod
    def _build_snippet_payload(
            *,
            step: Step,
            execution_state: ExecutionState,
            reason_code: str,
    ) -> Dict[str, Any]:
        return ResearchConvergenceJudge._build_payload_from_items(
            step=step,
            evidence_items=list(execution_state.research_search_evidence_items or []),
            runtime_recent_action=execution_state.runtime_recent_action,
            reason_code=reason_code,
        )

    @staticmethod
    def _build_payload_from_items(
            *,
            step: Step,
            evidence_items: List[Dict[str, Any]],
            runtime_recent_action: Optional[Dict[str, Any]],
            reason_code: str,
    ) -> Dict[str, Any]:
        evidence_lines: List[str] = []
        source_links: List[str] = []
        for index, item in enumerate(list(evidence_items or [])[:5], start=1):
            if not isinstance(item, dict):
                continue
            title = truncate_tool_text(item.get("title"), max_chars=100)
            url = str(item.get("url") or "").strip()
            snippet = truncate_tool_text(item.get("snippet"), max_chars=260)
            if not snippet and not url:
                continue
            if url:
                source_links.append(url)
            label = title or url or f"来源{index}"
            evidence_lines.append(f"{index}. {label}：{snippet}" if snippet else f"{index}. {label}")

        # research 收敛只形成步骤级阶段结论，最终面向用户的正文仍由 summary 节点统一组织。
        summary = f"当前研究步骤已基于搜索摘要证据完成：{step.description}"
        facts_learned = [
            line for line in evidence_lines
            if str(line).strip()
        ]
        facts_learned.extend([f"来源链接：{url}" for url in source_links[:5]])
        runtime_action = dict(runtime_recent_action or {})
        runtime_action["research_convergence"] = {
            "reason_code": reason_code,
            "source_link_count": len(source_links),
            "evidence_count": len(evidence_lines),
        }
        return {
            "success": True,
            "summary": summary,
            "result": summary,
            "attachments": [],
            "blockers": [],
            # Phase C：步骤收敛只保留结构化证据，不再产出步骤级正文。
            "facts_learned": facts_learned[:10],
            "open_questions": [],
            "next_hint": "",
            "runtime_recent_action": runtime_action,
        }


def _extract_runtime_search_evidence_items(runtime_recent_action: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从工具反馈沉淀的 recent_action 中提取 search snippet 证据。"""
    raw_items = runtime_recent_action.get("search_evidence_summaries")
    if isinstance(raw_items, list):
        return [item for item in raw_items if isinstance(item, dict)]
    research_progress = runtime_recent_action.get("research_progress")
    if isinstance(research_progress, dict):
        raw_progress_items = research_progress.get("search_evidence_summaries")
        if isinstance(raw_progress_items, list):
            return [item for item in raw_progress_items if isinstance(item, dict)]
    return []
