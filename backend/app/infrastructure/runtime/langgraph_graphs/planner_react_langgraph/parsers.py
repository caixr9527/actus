#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Planner-ReAct LangGraph 纯函数解析与归一化工具。"""

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, TypedDict

from app.domain.models import (
    MessageEvent,
    Plan,
    Step,
    ToolEvent,
    ToolEventStatus,
    ExecutionStatus,
)
from app.domain.services.prompts import SUMMARIZE_PROMPT
from app.domain.services.runtime.langgraph_state import PlannerReActLangGraphState

logger = logging.getLogger(__name__)

_SIMPLE_GREETING_NORMALIZED_SET = {
    "你好",
    "您好",
    "嗨",
    "哈喽",
    "在吗",
    "hi",
    "hello",
    "hey",
}

_TEXT_EXTENSIONS: set[str] = {
    "txt", "md", "markdown", "csv", "tsv", "json", "yaml", "yml", "xml", "html", "htm",
    "log", "ini", "cfg", "conf", "toml", "py", "js", "ts", "tsx", "jsx", "java", "c", "cpp",
    "h", "hpp", "go", "rs", "sql", "sh", "bash", "zsh",
}

_SUPPORTED_INPUT_TYPES: set[str] = {"text", "image", "audio", "pdf", "file_ref"}


class ModelInputPolicy(TypedDict):
    """模型输入策略。"""

    multimodal: bool
    supported: List[str]
    text_attachment_paths: List[str]
    native_user_content_parts: List[Dict[str, Any]]
    unsupported_parts: List[Dict[str, Any]]
    inline_text_from_attachments: bool


def safe_parse_json(content: str | None) -> Dict[str, Any]:
    """宽松解析模型 JSON 输出，兼容 fenced code 与前后赘述。"""
    if not content:
        return {}

    candidates: List[str] = [content]

    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
    if fenced_match:
        candidates.insert(0, fenced_match.group(1).strip())

    start_index = content.find("{")
    end_index = content.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        candidates.append(content[start_index:end_index + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    logger.warning("LangGraph 解析JSON失败，使用回退逻辑")
    return {}


def normalize_attachments(raw_attachments: Any) -> List[str]:
    if isinstance(raw_attachments, str):
        return [raw_attachments]
    if isinstance(raw_attachments, list):
        return [str(item) for item in raw_attachments if str(item).strip()]
    return []


def _is_text_file_part(part: Dict[str, Any]) -> bool:
    mime_type = str(part.get("mime_type") or "").strip().lower()
    extension = str(part.get("extension") or "").strip().lower().lstrip(".")
    if mime_type.startswith("text/"):
        return True
    return extension in _TEXT_EXTENSIONS


def _build_native_text_file_reference_part(part: Dict[str, Any]) -> Dict[str, Any]:
    """将文本文件引用转换为模型可消费的文本片段。"""
    file_id = str(part.get("file_id") or "").strip()
    text_content = str(part.get("text_content") or "").strip()
    source_label = file_id or "text_attachment"

    lines: List[str] = [f"以下是用户上传的文本附件内容（source: {source_label}）。"]
    if text_content:
        lines.append("附件正文已完整提供，请直接基于正文回答，不要调用 read_file。")
        lines.append(text_content)
    else:
        lines.append("附件正文暂不可直接读取。")
        lines.append("如需正文，请调用工具读取。")
    return {"type": "text", "text": "\n".join(lines)}


def normalize_supported_input_types(raw_supported: Any) -> List[str]:
    """归一化模型声明的 supported 输入类型。"""
    normalized: List[str] = []
    if isinstance(raw_supported, str):
        candidate = raw_supported.strip().lower()
        if candidate in _SUPPORTED_INPUT_TYPES:
            normalized.append(candidate)
    elif isinstance(raw_supported, list):
        for item in raw_supported:
            candidate = str(item or "").strip().lower()
            if candidate in _SUPPORTED_INPUT_TYPES and candidate not in normalized:
                normalized.append(candidate)

    if "text" not in normalized:
        normalized.insert(0, "text")
    return normalized


def resolve_model_input_policy(
        *,
        llm: Any,
        input_parts: List[Dict[str, Any]],
) -> ModelInputPolicy:
    """根据模型能力与输入片段构建处理策略。"""
    multimodal = bool(getattr(llm, "multimodal", False))
    supported = normalize_supported_input_types(getattr(llm, "supported", ["text"]))

    text_attachment_paths: List[str] = []
    native_user_content_parts: List[Dict[str, Any]] = []
    unsupported_parts: List[Dict[str, Any]] = []
    inline_text_from_attachments = False

    for raw_part in input_parts:
        if not isinstance(raw_part, dict):
            continue

        part_type = str(raw_part.get("type") or "").strip().lower()
        if part_type == "text":
            continue

        filepath = str(raw_part.get("filepath") or "").strip()
        uri = str(raw_part.get("uri") or "").strip()
        is_text_file_ref = part_type == "file_ref" and _is_text_file_part(raw_part)

        # 文本附件策略：
        # 1) 模型支持多模态且支持 text 时，优先以“模型可见文本片段”方式透传；
        # 2) 模型不支持（或 text 不在支持集合）时，回退沙箱 read_file 路径。
        if is_text_file_ref:
            text_content = str(raw_part.get("text_content") or "").strip()
            if multimodal and "text" in supported and text_content:
                native_user_content_parts.append(_build_native_text_file_reference_part(raw_part))
                inline_text_from_attachments = True
                continue
            if filepath:
                text_attachment_paths.append(filepath)
            if multimodal and "text" not in supported:
                unsupported_parts.append(
                    {
                        "type": "file_ref",
                        "filepath": filepath or None,
                        "reason": "model_supported_not_contains_text",
                    }
                )
            elif multimodal and "text" in supported and not text_content:
                unsupported_parts.append(
                    {
                        "type": "file_ref",
                        "filepath": filepath or None,
                        "reason": "text_content_unavailable_fallback_to_sandbox",
                    }
                )
            continue

        if not multimodal:
            if part_type != "file_ref":
                unsupported_parts.append(
                    {
                        "type": part_type or "unknown",
                        "filepath": filepath or None,
                        "reason": "model_multimodal_disabled",
                    }
                )
            continue

        if part_type not in supported:
            unsupported_parts.append(
                {
                    "type": part_type or "unknown",
                    "filepath": filepath or None,
                    "reason": "model_supported_not_contains_type",
                }
            )
            continue

        # 当前阶段仅对 image 做原生多模态输入映射。
        if part_type == "image":
            if not uri:
                unsupported_parts.append(
                    {
                        "type": "image",
                        "filepath": filepath or None,
                        "reason": "image_uri_missing",
                    }
                )
                continue
            native_user_content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": uri},
                }
            )
            continue

        if part_type in {"audio", "pdf"}:
            unsupported_parts.append(
                {
                    "type": part_type,
                    "filepath": filepath or None,
                    "reason": "native_content_mapper_not_implemented",
                }
            )
            continue

    return {
        "multimodal": multimodal,
        "supported": supported,
        "text_attachment_paths": merge_attachment_paths(text_attachment_paths),
        "native_user_content_parts": native_user_content_parts,
        "unsupported_parts": unsupported_parts,
        "inline_text_from_attachments": inline_text_from_attachments,
    }


def merge_attachment_paths(*path_groups: List[str]) -> List[str]:
    """按出现顺序去重合并附件路径。"""
    merged_paths: List[str] = []
    seen: set[str] = set()
    for path_group in path_groups:
        for raw_path in path_group:
            path = str(raw_path or "").strip()
            if not path or path in seen:
                continue
            seen.add(path)
            merged_paths.append(path)
    return merged_paths


def extract_write_file_paths_from_tool_events(tool_events: List[ToolEvent]) -> List[str]:
    """从 write_file 工具事件提取产物路径，作为附件兜底来源。"""
    attachment_paths: List[str] = []
    for tool_event in tool_events:
        if tool_event.status != ToolEventStatus.CALLED:
            continue
        function_name = str(tool_event.function_name or "").strip().lower()
        if function_name != "write_file":
            continue

        arg_path = str(tool_event.function_args.get("filepath") or "").strip()
        if arg_path:
            attachment_paths.append(arg_path)
            continue

        function_result = tool_event.function_result
        result_data = function_result.data if function_result is not None and hasattr(function_result, "data") else None
        if isinstance(result_data, dict):
            result_path = str(
                result_data.get("filepath")
                or result_data.get("file_path")
                or result_data.get("path")
                or ""
            ).strip()
            if result_path:
                attachment_paths.append(result_path)

    return merge_attachment_paths(attachment_paths)


def collect_plan_attachment_paths(plan: Optional[Plan]) -> List[str]:
    """汇总计划中已完成步骤产生的附件路径。"""
    if plan is None:
        return []

    collected_paths: List[str] = []
    for step in plan.steps:
        if not step.done:
            continue
        collected_paths.extend(normalize_attachments(step.attachments))
    return merge_attachment_paths(collected_paths)


def normalize_user_text(text: str) -> str:
    """归一化用户输入，便于做轻量意图判断。"""
    return re.sub(r"[\s\W_]+", "", text.strip().lower(), flags=re.UNICODE)


def is_simple_greeting_message(user_message: str) -> bool:
    """识别纯问候输入，避免进入完整 planner/step 链路。"""
    normalized = normalize_user_text(user_message)
    return normalized in _SIMPLE_GREETING_NORMALIZED_SET


def build_simple_greeting_reply(user_message: str) -> str:
    """根据输入语种返回简洁问候回复。"""
    has_ascii_letters = re.search(r"[a-z]", user_message, flags=re.IGNORECASE) is not None
    has_cjk = re.search(r"[\u4e00-\u9fff]", user_message) is not None
    if has_ascii_letters and not has_cjk:
        return "Hello! I'm your assistant, happy to help."
    return "你好！我是助手，很高兴为您服务。"


def build_fallback_plan_title(user_message: str) -> str:
    """规划 JSON 缺失标题时，生成可读回退标题。"""
    normalized = user_message.strip()
    if not normalized:
        return "任务计划"

    title = normalized[:24]
    return title if len(normalized) <= 24 else f"{title}..."


def get_last_assistant_message_event(events: List[Any]) -> Optional[MessageEvent]:
    """获取事件列表中最后一条 assistant 消息事件。"""
    for event in reversed(events):
        if isinstance(event, MessageEvent) and event.role == "assistant":
            return event
    return None


def collect_message_attachment_paths(message_event: Optional[MessageEvent]) -> List[str]:
    """提取消息事件中的附件路径，并按顺序去重。"""
    if message_event is None:
        return []

    attachment_paths: List[str] = []
    for attachment in list(message_event.attachments or []):
        filepath = str(getattr(attachment, "filepath", "") or "").strip()
        if filepath:
            attachment_paths.append(filepath)
    return merge_attachment_paths(attachment_paths)


def build_summarize_prompt(state: PlannerReActLangGraphState) -> str:
    """构建带上下文的总结提示词，降低无关总结风险。"""
    plan = state.get("plan")
    plan_snapshot = plan.model_dump(mode="json") if plan is not None else {}
    final_message = str(state.get("final_message") or "")
    user_message = str(state.get("user_message") or "")
    execution_count = int(state.get("execution_count") or 0)

    return (
        f"{SUMMARIZE_PROMPT}\n\n"
        "请严格基于以下运行上下文输出总结，禁止引入上下文之外的场景或数据：\n"
        f"- 用户原始消息: {user_message}\n"
        f"- 执行轮次: {execution_count}\n"
        f"- 最近一步结果: {final_message}\n"
        f"- 计划快照(JSON): {json.dumps(plan_snapshot, ensure_ascii=False)}\n"
    )


def should_accept_summary_message(
        state: PlannerReActLangGraphState,
        candidate_message: str,
        fallback_message: str,
) -> bool:
    """判断是否接受 summarize 文本，避免单步任务被无关长文覆盖。"""
    candidate = candidate_message.strip()
    if not candidate:
        return False

    execution_count = int(state.get("execution_count") or 0)
    if execution_count <= 1 and fallback_message.strip():
        fallback = fallback_message.strip()
        if fallback not in candidate and candidate not in fallback:
            logger.warning("LangGraph summarize 与单步执行结果无关联，回退到步骤结果")
            return False

        max_allowed_length = len(fallback_message.strip()) * 3 + 120
        if len(candidate) > max_allowed_length:
            logger.warning("LangGraph summarize 产出异常长文本，回退到步骤结果以避免无关回复")
            return False

    return True


def build_step_from_payload(payload: Any, fallback_index: int) -> Step:
    """将模型返回的步骤片段规范化为领域 Step。"""
    if isinstance(payload, dict):
        step_id = str(payload.get("id") or str(uuid.uuid4()))
        description = str(payload.get("description") or f"步骤{fallback_index + 1}")
        return Step(
            id=step_id,
            description=description,
            status=ExecutionStatus.PENDING,
        )

    return Step(
        id=str(uuid.uuid4()),
        description=str(payload).strip() or f"步骤{fallback_index + 1}",
        status=ExecutionStatus.PENDING,
    )


def format_attachments_for_prompt(attachments: List[str]) -> str:
    if not attachments:
        return "无"
    return "\n".join(f"- {item}" for item in attachments)


def should_emit_planner_message(user_message: str, planner_message: str, steps: List[Step]) -> bool:
    """判断是否需要输出规划消息，避免问候/回显噪音。"""
    normalized_planner_message = planner_message.strip()
    if not normalized_planner_message:
        return False

    normalized_user_message = user_message.strip()
    if normalized_user_message and normalized_planner_message == normalized_user_message:
        return False

    if len(steps) == 1:
        first_step_description = str(steps[0].description or "").strip()
        if first_step_description and normalized_planner_message == first_step_description:
            return False

    return True
