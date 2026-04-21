#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通用整理/最终交付步骤收敛判定。

本模块只处理两类 `general` 收敛：
1. 整理型步骤：已有 research / web_reading 证据时，直接组织交付；
2. 文件观察型步骤：已有目录/文件事实时，直接内联返回结果。

它不生成反馈经验、不修改计划，也不替代最终 summary 节点。
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

from app.domain.models import Step, StepDeliveryRole, StepOutputMode
from app.domain.services.runtime.normalizers import normalize_controlled_value
from app.domain.services.workspace_runtime.policies import truncate_tool_text


_GENERAL_SYNTHESIS_PATTERN = re.compile(
    r"(整理|总结|归纳|提炼|梳理|设计|计划|安排|预算|估算|整合|呈现|交付|要点|结论|摘要|概括|筛选|"
    r"summarize|synthesis|organize|plan|budget|deliver)",
    re.IGNORECASE,
)
_GENERAL_PRESENTATION_PATTERN = re.compile(
    r"(列出|罗列|展示|说明|返回|输出|概括|汇总|筛选|列明|print|list|show|return|present|outline|summarize)",
    re.IGNORECASE,
)
_GENERAL_FILE_OBSERVATION_PATTERN = re.compile(
    r"(目录|文件列表|文件名|当前目录|目录状态|列出文件|查看目录|list files|directory|folder|current directory|file names)",
    re.IGNORECASE,
)


@dataclass(slots=True)
class GeneralConvergenceResult:
    """general 步骤收敛结果。"""

    should_break: bool
    payload: Optional[Dict[str, Any]] = None
    reason_code: str = ""


class GeneralConvergenceJudge:
    """基于已有证据判断 general/final 步骤是否应停止继续调工具。"""

    def evaluate_after_iteration(
            self,
            *,
            step: Step,
            task_mode: str,
            runtime_recent_action: Optional[Dict[str, Any]],
            iteration: int,
    ) -> GeneralConvergenceResult:
        """每轮工具调用后判断 general 步骤是否已具备无工具完成条件。"""
        if str(task_mode or "").strip().lower() != "general":
            return GeneralConvergenceResult(should_break=False)
        if self._is_file_observation_step(step):
            file_evidence_lines = self._collect_file_evidence_lines(runtime_recent_action)
            if len(file_evidence_lines) == 0:
                return GeneralConvergenceResult(should_break=False)
            return GeneralConvergenceResult(
                should_break=True,
                payload=self._build_payload(
                    step=step,
                    runtime_recent_action=runtime_recent_action,
                    evidence_lines=file_evidence_lines,
                    reason_code="general_file_observation_ready",
                    summary_prefix="已基于目录/文件事实完成当前步骤",
                    heading="文件事实：",
                ),
                reason_code="general_file_observation_ready",
            )
        if not self._is_synthesis_step(step):
            return GeneralConvergenceResult(should_break=False)
        evidence_snapshot = self._collect_evidence_snapshot(runtime_recent_action)
        if len(evidence_snapshot["evidence_lines"]) == 0:
            return GeneralConvergenceResult(should_break=False)
        contract_requirement = self._infer_delivery_requirement(step)
        if not self._is_delivery_contract_satisfied(
                evidence_snapshot=evidence_snapshot,
                requirement=contract_requirement,
        ):
            return GeneralConvergenceResult(should_break=False)
        reason_code = (
            "general_final_delivery_ready"
            if self._is_final_inline_step(step)
            else "general_context_ready"
        )
        return GeneralConvergenceResult(
                should_break=True,
                payload=self._build_payload(
                    step=step,
                    runtime_recent_action=runtime_recent_action,
                    evidence_lines=evidence_snapshot["evidence_lines"],
                    reason_code=reason_code,
                ),
                reason_code=reason_code,
        )

    @staticmethod
    def build_max_iteration_convergence_payload(
            *,
            step: Step,
            task_mode: str,
            runtime_recent_action: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """最大轮次兜底：general 已有证据时不再失败重规划。"""
        judge = GeneralConvergenceJudge()
        result = judge.evaluate_after_iteration(
            step=step,
            task_mode=task_mode,
            runtime_recent_action=runtime_recent_action,
            iteration=-1,
        )
        return result.payload if result.should_break else None

    @staticmethod
    def _is_synthesis_step(step: Step) -> bool:
        candidate_text = " ".join(
            [
                str(getattr(step, "title", "") or ""),
                str(getattr(step, "description", "") or ""),
                " ".join([str(item or "") for item in list(getattr(step, "success_criteria", []) or [])]),
            ]
        ).strip()
        if not candidate_text:
            return False
        if not _GENERAL_PRESENTATION_PATTERN.search(candidate_text):
            return False
        return bool(_GENERAL_SYNTHESIS_PATTERN.search(candidate_text) or "来源链接" in candidate_text or "要点" in candidate_text)

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
        return bool(
            _GENERAL_FILE_OBSERVATION_PATTERN.search(candidate_text)
            and _GENERAL_PRESENTATION_PATTERN.search(candidate_text)
        )

    @staticmethod
    def _is_final_inline_step(step: Step) -> bool:
        output_mode = normalize_controlled_value(getattr(step, "output_mode", None), StepOutputMode)
        delivery_role = normalize_controlled_value(getattr(step, "delivery_role", None), StepDeliveryRole)
        return (
                output_mode == StepOutputMode.INLINE.value
                and delivery_role == StepDeliveryRole.FINAL.value
        )

    @staticmethod
    def _collect_evidence_lines(runtime_recent_action: Optional[Dict[str, Any]]) -> List[str]:
        return GeneralConvergenceJudge._collect_evidence_snapshot(runtime_recent_action)["evidence_lines"]

    @staticmethod
    def _collect_evidence_snapshot(runtime_recent_action: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        recent_action = dict(runtime_recent_action or {})
        evidence_lines: List[str] = []
        source_links: List[str] = []
        point_like_count = 0
        for item in _extract_list(recent_action, "search_evidence_summaries"):
            if not isinstance(item, dict):
                continue
            title = truncate_tool_text(item.get("title"), max_chars=80)
            snippet = truncate_tool_text(item.get("snippet"), max_chars=180)
            url = truncate_tool_text(item.get("url"), max_chars=180)
            if snippet:
                evidence_lines.append(f"{title or url or '搜索证据'}：{snippet}")
                point_like_count += 1
            elif url:
                evidence_lines.append(f"来源链接：{url}")
            if url:
                source_links.append(url)
        research_progress = dict(recent_action.get("research_progress") or {})
        for item in _extract_list(research_progress, "search_evidence_summaries"):
            if not isinstance(item, dict):
                continue
            snippet = truncate_tool_text(item.get("snippet"), max_chars=180)
            url = truncate_tool_text(item.get("url"), max_chars=180)
            if snippet:
                evidence_lines.append(f"{truncate_tool_text(item.get('title'), max_chars=80) or url or '搜索证据'}：{snippet}")
                point_like_count += 1
            elif url:
                evidence_lines.append(f"来源链接：{url}")
            if url:
                source_links.append(url)
        for item in _extract_list(recent_action, "web_reading_evidence_summaries"):
            if isinstance(item, dict):
                summary = truncate_tool_text(item.get("summary"), max_chars=200)
                url = truncate_tool_text(item.get("url"), max_chars=180)
                if summary:
                    evidence_lines.append(f"{url or '页面证据'}：{summary}")
                    point_like_count += 1
                if url:
                    source_links.append(url)
        last_success = dict(recent_action.get("last_successful_tool_call") or {})
        success_message = truncate_tool_text(last_success.get("message"), max_chars=180)
        if success_message:
            evidence_lines.append(success_message)
            point_like_count += 1
        deduped_lines = _dedupe_non_empty(evidence_lines)[:8]
        deduped_links = _dedupe_non_empty(source_links)[:8]
        return {
            "evidence_lines": deduped_lines,
            "source_links": deduped_links,
            "point_like_count": min(point_like_count, len(deduped_lines)),
        }

    @staticmethod
    def _infer_delivery_requirement(step: Step) -> Dict[str, Any]:
        """从步骤文本提取 general 整理步骤的结构化交付要求。"""
        candidate_text = " ".join(
            [
                str(getattr(step, "title", "") or ""),
                str(getattr(step, "description", "") or ""),
                " ".join([str(item or "") for item in list(getattr(step, "success_criteria", []) or [])]),
            ]
        ).strip()
        point_count = _infer_count_from_text(
            candidate_text,
            patterns=[
                r"(\d+|[一二两三四五六七八九十]+)\s*(?:条|点|项|个)\s*(?:要点|结论|摘要|建议|信息)",
                r"(?:整理|总结|提炼|概括|列出)\s*(\d+|[一二两三四五六七八九十]+)\s*(?:条|点|项|个)",
            ],
        )
        return {
            "min_point_count": max(1, point_count or 1),
            "require_source_links": bool(re.search(r"(来源链接|每条.*链接|附上链接|引用链接|参考链接|source\s+links?)", candidate_text, re.IGNORECASE)),
        }

    @staticmethod
    def _is_delivery_contract_satisfied(
            *,
            evidence_snapshot: Dict[str, Any],
            requirement: Dict[str, Any],
    ) -> bool:
        if int(evidence_snapshot.get("point_like_count") or 0) < int(requirement.get("min_point_count") or 1):
            return False
        if bool(requirement.get("require_source_links")):
            source_link_count = len(list(evidence_snapshot.get("source_links") or []))
            if source_link_count < int(requirement.get("min_point_count") or 1):
                return False
        return True

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
            dir_path = truncate_tool_text(data.get("dir_path"), max_chars=160)
            raw_files = data.get("files")
            file_names: List[str] = []
            if isinstance(raw_files, list):
                for item in raw_files[:20]:
                    if isinstance(item, dict):
                        name = str(item.get("name") or item.get("filename") or "").strip()
                    else:
                        name = str(item or "").strip()
                    if name:
                        file_names.append(name)
            if dir_path:
                evidence_lines.append(f"当前目录：{dir_path}")
            if file_names:
                evidence_lines.append("文件列表：" + "、".join(file_names[:12]))
        if function_name == "find_files" and isinstance(data, dict):
            dir_path = truncate_tool_text(data.get("dir_path"), max_chars=160)
            raw_files = data.get("files")
            file_names = [str(item.get("name") or item.get("filename") or item or "").strip() for item in list(raw_files or [])[:20]]
            file_names = [item for item in file_names if item]
            if dir_path:
                evidence_lines.append(f"搜索目录：{dir_path}")
            if file_names:
                evidence_lines.append("匹配文件：" + "、".join(file_names[:12]))
        return _dedupe_non_empty(evidence_lines)[:8]

    @staticmethod
    def _build_payload(
            *,
            step: Step,
            runtime_recent_action: Optional[Dict[str, Any]],
            evidence_lines: List[str],
            reason_code: str,
            summary_prefix: str = "已基于现有上下文完成整理步骤",
            heading: str = "可用依据：",
    ) -> Dict[str, Any]:
        summary = f"{summary_prefix}：{step.description}"
        delivery_lines = [summary, heading, *[f"- {line}" for line in evidence_lines[:6]]]
        delivery_text = "\n".join(delivery_lines)
        runtime_action = dict(runtime_recent_action or {})
        runtime_action["general_convergence"] = {
            "reason_code": reason_code,
            "evidence_count": len(evidence_lines),
        }
        return {
            "success": True,
            "summary": summary,
            "result": delivery_text,
            "delivery_text": delivery_text,
            "attachments": [],
            "blockers": [],
            "facts_learned": evidence_lines[:8],
            "open_questions": [],
            "next_hint": "",
            "runtime_recent_action": runtime_action,
        }


def _extract_list(source: Dict[str, Any], key: str) -> List[Any]:
    value = source.get(key)
    return list(value) if isinstance(value, list) else []


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


def _infer_count_from_text(candidate_text: str, *, patterns: List[str]) -> int:
    for pattern in patterns:
        matched = re.search(pattern, candidate_text, re.IGNORECASE)
        if matched is None:
            continue
        return _parse_count_token(matched.group(1))
    return 0


def _parse_count_token(raw_value: str) -> int:
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
    return 0
