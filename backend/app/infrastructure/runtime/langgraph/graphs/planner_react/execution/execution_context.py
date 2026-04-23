#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P3 重构：执行循环上下文构建与步骤语义判断。"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from app.domain.models import (
    Step,
    StepArtifactPolicy,
    StepOutputMode,
    StepTaskModeHint,
)
from app.domain.services.runtime.contracts.langgraph_settings import (
    BROWSER_HIGH_LEVEL_FUNCTION_NAMES,
    EXPLICIT_FILE_OUTPUT_REQUEST_PATTERN,
    READ_ONLY_FILE_FUNCTION_NAMES,
    TASK_MODE_ALLOWED_PREFIXES,
    TASK_MODE_MAX_TOOL_ITERATIONS,
)
from app.domain.services.runtime.normalizers import normalize_controlled_value
from app.domain.services.workspace_runtime.policies import (
    analyze_text_intent as _analyze_text_intent,
    build_browser_capability_gap_allowlist as _build_browser_capability_gap_allowlist,
    build_step_candidate_text as _build_step_candidate_text,
    build_task_mode_disallowed_names as _build_task_mode_disallowed_names,
    classify_file_access_intent as _classify_file_access_intent,
)

@dataclass(slots=True)
class ExecutionContext:
    normalized_user_content: List[Dict[str, Any]]
    available_tools: List[Dict[str, Any]]
    available_function_names: Set[str]
    browser_route_enabled: bool
    blocked_function_names: Set[str]
    read_only_file_blocked_function_names: Set[str]
    research_file_context_blocked_function_names: Set[str]
    file_processing_shell_blocked_function_names: Set[str]
    artifact_policy_blocked_function_names: Set[str]
    requested_max_tool_iterations: int
    effective_max_tool_iterations: int
    allow_ask_user: bool
    research_route_enabled: bool
    research_has_explicit_url: bool
    # 当前轮用户原始输入文本（不含系统拼接提示），供研究链路显式 URL 判定使用。
    current_user_message_text: str = ""


def step_allows_user_wait(step: Step, function_args: Dict[str, Any]) -> bool:
    takeover = str(function_args.get("suggest_user_takeover") or "").strip().lower()
    if takeover == "browser":
        return True

    if normalize_controlled_value(getattr(step, "task_mode_hint", None), StepTaskModeHint) == "human_wait":
        return True

    candidate_text = _build_step_candidate_text(step)
    if not candidate_text.strip():
        return False
    return bool(_analyze_text_intent(candidate_text)["needs_human_wait"])


def build_execution_context(
        *,
        step: Step,
        task_mode: str,
        max_tool_iterations: int,
        user_content: Optional[List[Dict[str, Any]]],
        has_available_file_context: bool,
        available_tools: List[Dict[str, Any]],
        available_function_names: Set[str],
        user_message_text: str = "",
        read_only_intent_text: str = "",
        file_output_intent_text: str = "",
) -> ExecutionContext:
    normalized_user_content = list(user_content or [])
    if len(normalized_user_content) == 0:
        prompt_text = str(step.description or "").strip()
        normalized_user_content = [{"type": "text", "text": prompt_text}]

    browser_route_enabled = (
            task_mode in {"web_reading", "browser_interaction"}
            and any(function_name in available_function_names for function_name in BROWSER_HIGH_LEVEL_FUNCTION_NAMES)
    )
    blocked_function_names = _build_task_mode_disallowed_names(
        list(available_function_names),
        task_mode=task_mode,
    )
    research_file_context_blocked_function_names: Set[str] = set()
    read_only_file_blocked_function_names: Set[str] = set()
    file_processing_shell_blocked_function_names: Set[str] = set()
    artifact_policy_blocked_function_names: Set[str] = set()

    # P3-1A 收敛修复：显式只读诉求下，禁止写文件/改文件/执行命令，避免“读请求写副作用”。
    if _step_requests_read_only_file_access(
            step=step,
            read_only_intent_text=read_only_intent_text,
    ):
        read_only_file_blocked_function_names.update({"write_file", "replace_in_file", "shell_execute"})
        blocked_function_names.update(read_only_file_blocked_function_names)

    if task_mode == "research" and not has_available_file_context:
        research_file_context_blocked_function_names.update(READ_ONLY_FILE_FUNCTION_NAMES)
        blocked_function_names.update(research_file_context_blocked_function_names)
    if _step_is_general_summary_only(step=step, task_mode=task_mode):
        # Step 纯执行化：general 的纯整理步骤不应继续发起 research/fetch 链路。
        # 这类步骤只能消费已有上下文事实，由 summary 阶段统一组织最终正文。
        blocked_function_names.update({"search_web", "fetch_page"})
    if task_mode == "file_processing" and not _step_explicitly_requests_shell_execution(step, normalized_user_content):
        # P3-CASE3 修复：文件处理默认只走文件工具，显式命令意图才允许 shell_execute。
        file_processing_shell_blocked_function_names.add("shell_execute")
        blocked_function_names.update(file_processing_shell_blocked_function_names)
    if _step_forbids_file_output(step):
        artifact_policy_blocked_function_names.update({"write_file", "replace_in_file"})
        blocked_function_names.update(artifact_policy_blocked_function_names)
    if task_mode in {"web_reading", "browser_interaction"} and not browser_route_enabled:
        for function_name in _build_browser_capability_gap_allowlist(task_mode=task_mode):
            blocked_function_names.discard(function_name)

    blocked_function_names.difference_update(
        {
            function_name
            for function_name in read_only_file_blocked_function_names
            if function_name not in available_function_names
        }
    )
    blocked_function_names.difference_update(
        {
            function_name
            for function_name in research_file_context_blocked_function_names
            if function_name not in available_function_names
        }
    )
    blocked_function_names.difference_update(
        {
            function_name
            for function_name in file_processing_shell_blocked_function_names
            if function_name not in available_function_names
        }
    )
    blocked_function_names.difference_update(
        {
            function_name
            for function_name in artifact_policy_blocked_function_names
            if function_name not in available_function_names
        }
    )

    requested_max_tool_iterations = max(1, int(max_tool_iterations))
    effective_max_tool_iterations = _resolve_effective_max_tool_iterations(
        task_mode=task_mode,
        requested_max_tool_iterations=requested_max_tool_iterations,
    )
    allow_ask_user = task_mode == "human_wait" or step_allows_user_wait(step, {})
    research_route_enabled = (
            task_mode in {"research", "web_reading"}
            and {"search_web", "fetch_page"}.issubset(available_function_names)
    )
    research_has_explicit_url = research_route_enabled and _step_or_user_content_has_url(
        step,
        normalized_user_content,
        user_message_text=user_message_text,
    )

    return ExecutionContext(
        normalized_user_content=normalized_user_content,
        available_tools=available_tools,
        available_function_names=available_function_names,
        browser_route_enabled=browser_route_enabled,
        blocked_function_names=blocked_function_names,
        read_only_file_blocked_function_names=read_only_file_blocked_function_names,
        research_file_context_blocked_function_names=research_file_context_blocked_function_names,
        file_processing_shell_blocked_function_names=file_processing_shell_blocked_function_names,
        artifact_policy_blocked_function_names=artifact_policy_blocked_function_names,
        requested_max_tool_iterations=requested_max_tool_iterations,
        effective_max_tool_iterations=effective_max_tool_iterations,
        allow_ask_user=allow_ask_user,
        research_route_enabled=research_route_enabled,
        research_has_explicit_url=research_has_explicit_url,
        current_user_message_text=str(user_message_text or "").strip(),
    )


def _step_forbids_file_output(step: Step) -> bool:
    """结构化产物策略优先决定是否禁止当前步骤产出文件。"""
    return normalize_controlled_value(getattr(step, "artifact_policy", None),
                                      StepArtifactPolicy) == "forbid_file_output"


def _step_explicitly_requests_file_output(*, step: Step, file_output_intent_text: str) -> bool:
    artifact_policy = normalize_controlled_value(getattr(step, "artifact_policy", None), StepArtifactPolicy)
    if artifact_policy in {
        StepArtifactPolicy.ALLOW_FILE_OUTPUT.value,
        StepArtifactPolicy.REQUIRE_FILE_OUTPUT.value,
    }:
        return True
    candidate_text = str(file_output_intent_text or "").strip() or _build_step_candidate_text(step)
    return bool(EXPLICIT_FILE_OUTPUT_REQUEST_PATTERN.search(candidate_text))


def _resolve_effective_max_tool_iterations(*, task_mode: str, requested_max_tool_iterations: int) -> int:
    mode_cap = TASK_MODE_MAX_TOOL_ITERATIONS.get(str(task_mode or "").strip().lower())
    if mode_cap is None:
        return requested_max_tool_iterations
    return min(requested_max_tool_iterations, max(1, int(mode_cap)))


def _extract_text_from_user_content(user_content: Optional[List[Dict[str, Any]]]) -> str:
    fragments: List[str] = []
    for item in list(user_content or []):
        if not isinstance(item, dict):
            continue
        if str(item.get("type") or "").strip().lower() != "text":
            continue
        text = str(item.get("text") or "").strip()
        if text:
            fragments.append(text)
    return "\n".join(fragments)


def _step_explicitly_requests_shell_execution(step: Step, user_content: Optional[List[Dict[str, Any]]]) -> bool:
    candidate_text = "\n".join(
        [
            _build_step_candidate_text(step),
            _extract_text_from_user_content(user_content),
        ]
    )
    signals = _analyze_text_intent(candidate_text)
    if bool(signals.get("has_shell_command")):
        return True
    normalized_text = str(signals.get("text") or "").strip().lower()
    explicit_markers = (
        "shell_execute",
        "执行命令",
        "运行命令",
        "终端命令",
        "命令行执行",
        "run command",
        "execute command",
        "terminal command",
    )
    return any(marker in normalized_text for marker in explicit_markers)


def _collect_shell_function_names(available_function_names: Set[str]) -> Set[str]:
    shell_prefixes = tuple(TASK_MODE_ALLOWED_PREFIXES.get("general") or ())
    matched_names: Set[str] = set()
    for function_name in set(available_function_names or set()):
        normalized_name = str(function_name or "").strip().lower()
        if not normalized_name:
            continue
        if any(normalized_name.startswith(prefix) for prefix in shell_prefixes):
            matched_names.add(normalized_name)
    return matched_names


def _step_requests_read_only_file_access(
        *,
        step: Step,
        read_only_intent_text: str,
) -> bool:
    candidate_text = str(read_only_intent_text or "").strip() or _build_step_candidate_text(step)
    return _classify_file_access_intent(candidate_text) == "read_only_intent"


def _step_is_general_summary_only(*, step: Step, task_mode: str) -> bool:
    """识别 general 模式下的纯整理步骤，用于执行期硬限制。

    约束：
    - 只拦截“基于已有信息整理/归纳/输出要点”的 general 步骤；
    - 不拦截带有明确文件观察、编码、文件写入等执行语义的 general 步骤。
    """
    if str(task_mode or "").strip().lower() != "general":
        return False
    candidate_text = _build_step_candidate_text(step)
    if not candidate_text:
        return False
    lowered = candidate_text.lower()
    summary_markers = (
        "整理",
        "总结",
        "归纳",
        "提炼",
        "梳理",
        "汇总",
        "要点",
        "结论",
        "summary",
        "summarize",
        "final answer",
        "final response",
    )
    execution_markers = (
        "目录",
        "文件",
        "创建",
        "写入",
        "修改",
        "执行",
        "运行",
        "导出",
        "保存",
        "read_file",
        "write_file",
        "shell",
        "code",
    )
    if not any(marker in lowered for marker in summary_markers):
        return False
    if any(marker in lowered for marker in execution_markers):
        return False
    return True


def _step_or_user_content_has_url(
        step: Step,
        user_content: Optional[List[Dict[str, Any]]],
        *,
        user_message_text: str = "",
) -> bool:
    # P3-一次性收口：显式 URL 判定优先使用“步骤语义 + 当前轮用户原始输入”，
    # 避免系统拼接提示词/历史上下文中的 URL 污染当前轮研究路由。
    current_user_text = str(user_message_text or "").strip()
    candidate_text = "\n".join(
        [
            _build_step_candidate_text(step),
            current_user_text,
            # 兼容当前调用形态：在没有显式用户文本时再退回 user_content。
            _extract_text_from_user_content(user_content) if not current_user_text else "",
        ]
    )
    return bool(_analyze_text_intent(candidate_text)["has_url"])
