#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""网页阅读步骤收敛判定。

本模块统一定义 `web_reading` 的完成合同：
- 只有拿到强页面证据时，才允许按“已完成阅读”收敛；
- 或者显式 URL 已进入受控降级时，才允许按“读取失败但停止重复”收敛；
- 该合同同时被工具循环中的 convergence 与 no-tool / max-iteration finalizer 共用。
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

from app.domain.models import Step
from app.domain.services.workspace_runtime.policies import truncate_tool_text
from app.infrastructure.runtime.langgraph.graphs.planner_react.execution.execution_state import ExecutionState
from app.infrastructure.runtime.langgraph.graphs.planner_react.research.research_diagnosis import (
    get_page_reading_contract_state,
)


@dataclass(slots=True)
class WebReadingConvergenceResult:
    """网页阅读收敛结果。"""

    should_break: bool
    payload: Optional[Dict[str, Any]] = None
    reason_code: str = ""


class WebReadingConvergenceJudge:
    """根据页面证据与显式 URL 降级状态判断 web_reading 是否应停止工具循环。"""

    @staticmethod
    def get_completion_contract_state(
            *,
            runtime_recent_action: Optional[Dict[str, Any]],
            execution_state: Optional[ExecutionState] = None,
    ) -> Dict[str, Any]:
        """统一返回 web_reading 完成合同状态，避免多处各写一套判断条件。"""
        return get_page_reading_contract_state(
            runtime_recent_action=runtime_recent_action,
            execution_state=execution_state,
        )

    @classmethod
    def get_completion_progress(
            cls,
            *,
            step: Step,
            runtime_recent_action: Optional[Dict[str, Any]],
            execution_state: Optional[ExecutionState] = None,
    ) -> Dict[str, Any]:
        """统一返回 web_reading 当前合同进度，供约束层与收敛层复用。"""
        contract_state = cls.get_completion_contract_state(
            runtime_recent_action=runtime_recent_action,
            execution_state=execution_state,
        )
        strong_evidence_items = list(contract_state.get("strong_evidence_items") or [])
        requirement = cls._infer_completion_requirement(step)
        unique_urls = {
            str(item.get("url") or "").strip()
            for item in strong_evidence_items
            if isinstance(item, dict) and str(item.get("url") or "").strip()
        }
        current_page_count = len(unique_urls) if unique_urls else len(strong_evidence_items)
        current_source_link_count = len(unique_urls)
        required_page_count = max(1, int(requirement["min_page_count"]))
        require_source_links = bool(requirement["require_source_links"])
        contract_satisfied = cls._has_enough_evidence(
            evidence_items=strong_evidence_items,
            min_page_count=required_page_count,
            require_source_links=require_source_links,
        )
        remaining_page_count = max(0, required_page_count - current_page_count)
        progress = {
            "current_page_count": current_page_count,
            "current_source_link_count": current_source_link_count,
            "required_page_count": required_page_count,
            "require_source_links": require_source_links,
            "remaining_page_count": remaining_page_count,
            "contract_satisfied": contract_satisfied,
        }
        runtime_action = dict(runtime_recent_action or {})
        runtime_action["web_reading_progress"] = dict(progress)
        return {
            **contract_state,
            "requirement": requirement,
            "progress": progress,
            "runtime_recent_action": runtime_action,
        }

    @classmethod
    def should_allow_model_only_success(
            cls,
            *,
            task_mode: str,
            runtime_recent_action: Optional[Dict[str, Any]],
    ) -> bool:
        """无工具成功返回时，只有满足页面证据合同才允许 web_reading 直接完成。"""
        if str(task_mode or "").strip().lower() != "web_reading":
            return True
        contract_state = cls.get_completion_contract_state(runtime_recent_action=runtime_recent_action)
        return bool(contract_state["has_strong_evidence"] or contract_state["degraded"])

    def evaluate_after_iteration(
            self,
            *,
            step: Step,
            task_mode: str,
            recent_function_name: str,
            execution_state: ExecutionState,
    ) -> WebReadingConvergenceResult:
        if str(task_mode or "").strip().lower() != "web_reading":
            return WebReadingConvergenceResult(should_break=False)
        normalized_function_name = str(recent_function_name or "").strip().lower()
        if normalized_function_name not in {
            "fetch_page",
            "browser_read_current_page_structured",
            "browser_extract_main_content",
            "browser_find_actionable_elements",
        }:
            return WebReadingConvergenceResult(should_break=False)

        contract_state = self.get_completion_contract_state(
            runtime_recent_action=execution_state.runtime_recent_action,
            execution_state=execution_state,
        )
        contract_progress_state = self.get_completion_progress(
            step=step,
            runtime_recent_action=execution_state.runtime_recent_action,
            execution_state=execution_state,
        )
        strong_evidence_items = list(contract_state.get("strong_evidence_items") or [])
        execution_state.runtime_recent_action["web_reading_progress"] = dict(
            contract_progress_state.get("progress") or {}
        )
        if bool(dict(contract_progress_state.get("progress") or {}).get("contract_satisfied")):
            reason_code = (
                "explicit_url_browser_evidence_ready"
                if bool(execution_state.explicit_url_degraded)
                else "web_reading_page_evidence_ready"
            )
            return WebReadingConvergenceResult(
                should_break=True,
                payload=self._build_payload(
                    step=step,
                    execution_state=execution_state,
                    evidence_items=strong_evidence_items,
                    reason_code=reason_code,
                ),
                reason_code=reason_code,
            )

        if bool(execution_state.explicit_url_degraded):
            reason_code = "explicit_url_fetch_degraded"
            return WebReadingConvergenceResult(
                should_break=True,
                payload=self._build_degraded_payload(
                    step=step,
                    execution_state=execution_state,
                    reason_code=reason_code,
                ),
                reason_code=reason_code,
            )

        return WebReadingConvergenceResult(should_break=False)

    @staticmethod
    def build_max_iteration_convergence_payload(
            *,
            step: Step,
            task_mode: str,
            runtime_recent_action: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if str(task_mode or "").strip().lower() != "web_reading":
            return None
        recent_action = dict(runtime_recent_action or {})
        contract_state = WebReadingConvergenceJudge.get_completion_contract_state(
            runtime_recent_action=recent_action,
        )
        progress_state = WebReadingConvergenceJudge.get_completion_progress(
            step=step,
            runtime_recent_action=recent_action,
        )
        recent_action = dict(progress_state.get("runtime_recent_action") or recent_action)
        strong_evidence_items = list(contract_state.get("strong_evidence_items") or [])
        if bool(dict(progress_state.get("progress") or {}).get("contract_satisfied")):
            return WebReadingConvergenceJudge._build_payload_from_runtime_action(
                step=step,
                runtime_recent_action=recent_action,
                evidence_items=strong_evidence_items,
                reason_code="web_reading_max_iteration_evidence_fallback",
            )
        explicit_url_state = dict(contract_state.get("explicit_url_state") or {})
        if bool(explicit_url_state.get("degraded")):
            return WebReadingConvergenceJudge._build_degraded_payload_from_runtime_action(
                step=step,
                runtime_recent_action=recent_action,
                reason_code="explicit_url_degraded_before_replan",
            )
        return None

    @staticmethod
    def _infer_completion_requirement(step: Step) -> Dict[str, Any]:
        """从步骤文本中提取页面阅读完成合同，只处理通用数量/链接要求。"""
        candidate_text = " ".join(
            [
                str(getattr(step, "title", "") or ""),
                str(getattr(step, "description", "") or ""),
                " ".join([str(item or "") for item in list(getattr(step, "success_criteria", []) or [])]),
            ]
        ).strip()
        return {
            "min_page_count": _infer_required_count(
                candidate_text,
                pattern=r"(?:至少|不少于|读取|阅读|参考|覆盖)\s*(\d+|[一二两三四五六七八九十]+)\s*(?:个|篇|条|个不同)?(?:页面|网页|来源|站点|链接)",
                default=1,
            ),
            "require_source_links": bool(re.search(r"(来源链接|引用链接|参考链接|链接信息|source\s+link|sources?)", candidate_text, re.IGNORECASE)),
        }

    @staticmethod
    def _has_enough_evidence(
            *,
            evidence_items: List[Dict[str, Any]],
            min_page_count: int,
            require_source_links: bool,
    ) -> bool:
        unique_urls = {
            str(item.get("url") or "").strip()
            for item in evidence_items
            if isinstance(item, dict) and str(item.get("url") or "").strip()
        }
        evidence_count = len(unique_urls) if unique_urls else len(evidence_items)
        if evidence_count < max(1, min_page_count):
            return False
        if require_source_links and len(unique_urls) < max(1, min_page_count):
            return False
        return True

    @staticmethod
    def _build_payload(
            *,
            step: Step,
            execution_state: ExecutionState,
            evidence_items: List[Dict[str, Any]],
            reason_code: str,
    ) -> Dict[str, Any]:
        return WebReadingConvergenceJudge._build_payload_from_runtime_action(
            step=step,
            runtime_recent_action=execution_state.runtime_recent_action,
            evidence_items=evidence_items,
            reason_code=reason_code,
        )

    @staticmethod
    def _build_payload_from_runtime_action(
            *,
            step: Step,
            runtime_recent_action: Dict[str, Any],
            evidence_items: List[Dict[str, Any]],
            reason_code: str,
    ) -> Dict[str, Any]:
        evidence_lines: List[str] = []
        source_links: List[str] = []
        for index, item in enumerate(evidence_items[:6], start=1):
            title = truncate_tool_text(item.get("title"), max_chars=100)
            url = str(item.get("url") or "").strip()
            summary = truncate_tool_text(item.get("summary"), max_chars=260)
            link_count = int(item.get("link_count") or 0)
            if url:
                source_links.append(url)
            label = title or url or f"页面证据{index}"
            suffix = f"；链接数 {link_count}" if link_count > 0 else ""
            evidence_lines.append(f"{index}. {label}：{summary}{suffix}" if summary else f"{index}. {label}{suffix}")

        # web_reading 收敛只形成页面阅读阶段结果，不直接充当最终对用户的正文。
        summary_text = f"当前网页阅读步骤已基于页面证据完成：{step.description}"
        runtime_action = dict(runtime_recent_action or {})
        runtime_action["web_reading_convergence"] = {
            "reason_code": reason_code,
            "evidence_count": len(evidence_lines),
            "source_link_count": len(source_links),
        }
        runtime_action.setdefault("web_reading_progress", {})
        return {
            "success": True,
            "summary": summary_text,
            "result": summary_text,
            "attachments": [],
            "blockers": [],
            "facts_learned": evidence_lines[:8] + [f"来源链接：{url}" for url in source_links[:5]],
            "open_questions": [],
            "next_hint": "",
            "runtime_recent_action": runtime_action,
        }

    @staticmethod
    def _build_degraded_payload(
            *,
            step: Step,
            execution_state: ExecutionState,
            reason_code: str,
    ) -> Dict[str, Any]:
        return WebReadingConvergenceJudge._build_degraded_payload_from_runtime_action(
            step=step,
            runtime_recent_action=execution_state.runtime_recent_action,
            reason_code=reason_code,
        )

    @staticmethod
    def _build_degraded_payload_from_runtime_action(
            *,
            step: Step,
            runtime_recent_action: Dict[str, Any],
            reason_code: str,
    ) -> Dict[str, Any]:
        explicit_url_state = dict(runtime_recent_action.get("explicit_url_read_state") or {})
        failed_urls = [str(item).strip() for item in list(explicit_url_state.get("failed_urls") or []) if str(item).strip()]
        failed_url_text = "、".join(failed_urls[:3]) if failed_urls else "用户提供的显式 URL"
        browser_evidence_ready = bool(explicit_url_state.get("browser_evidence_ready"))
        summary = f"显式 URL 页面读取已降级：{failed_url_text}"
        degraded_fact = (
            f"{summary}。原因：fetch_page 未获得足够正文内容，且继续重复抓取不会增加有效信息。"
            + (
                " 浏览器已打开页面并拿到部分结构信号，但未获得足够稳定的正文摘要。"
                if browser_evidence_ready
                else ""
            )
            + " 当前步骤停止重复读取，后续只能基于已有可用信息或明确向用户说明页面读取失败。"
        )
        runtime_action = dict(runtime_recent_action or {})
        runtime_action["web_reading_convergence"] = {
            "reason_code": reason_code,
            "evidence_count": 0,
        }
        return {
            "success": True,
            "summary": summary,
            "result": summary,
            "attachments": [],
            "blockers": [],
            "facts_learned": [degraded_fact],
            "open_questions": [],
            "next_hint": "",
            "runtime_recent_action": runtime_action,
        }


def _infer_required_count(candidate_text: str, *, pattern: str, default: int) -> int:
    matched = re.search(pattern, candidate_text, re.IGNORECASE)
    if matched is None:
        return default
    return max(default, _parse_count_text(matched.group(1)))


def _parse_count_text(raw_value: str) -> int:
    value = str(raw_value or "").strip()
    if value.isdigit():
        return int(value)
    chinese_digits = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    if value in chinese_digits:
        return chinese_digits[value]
    if value.startswith("十") and len(value) == 2:
        return 10 + chinese_digits.get(value[1], 0)
    if value.endswith("十") and len(value) == 2:
        return chinese_digits.get(value[0], 1) * 10
    if "十" in value:
        left, right = value.split("十", 1)
        return chinese_digits.get(left, 1) * 10 + chinese_digits.get(right, 0)
    return 1
