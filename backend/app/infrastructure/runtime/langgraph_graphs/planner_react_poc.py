#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/20 13:10
@Author : caixiaorong01@outlook.com
@File   : planner_react_poc.py
"""
import json
import inspect
import logging
import re
import uuid
from contextvars import ContextVar, Token
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, Tuple

from app.domain.external import LLM
from app.domain.models import (
    BaseEvent,
    File,
    Plan,
    Step,
    ExecutionStatus,
    TitleEvent,
    PlanEvent,
    PlanEventStatus,
    StepEvent,
    StepEventStatus,
    MessageEvent,
    ToolEvent,
    ToolEventStatus,
    ToolResult,
    DoneEvent,
)
from app.domain.services.prompts import CREATE_PLAN_PROMPT, EXECUTION_PROMPT, SUMMARIZE_PROMPT, UPDATE_PLAN_PROMPT
from app.domain.services.runtime import SkillGraphRuntime
from app.domain.services.runtime.langgraph_events import append_events
from app.domain.services.runtime.langgraph_state import PlannerReActPOCState
from app.domain.services.tools import BaseTool
from app.infrastructure.runtime.langgraph_graphs.skill_subgraphs import build_default_skill_graph_registry

logger = logging.getLogger(__name__)

PLANNER_EXECUTE_STEP_SKILL_ID = "planner_react.execute_step"
_LIVE_EVENT_SINK: ContextVar[
    Optional[Callable[[BaseEvent], Optional[Awaitable[None]]]]
] = ContextVar("langgraph_poc_live_event_sink", default=None)

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.checkpoint.memory import InMemorySaver

    LANGGRAPH_AVAILABLE = True
    LANGGRAPH_IMPORT_ERROR = None
except ImportError as e:  # pragma: no cover - 依赖缺失时的保护逻辑
    StateGraph = None
    START = "__start__"
    END = "__end__"
    InMemorySaver = None
    LANGGRAPH_AVAILABLE = False
    LANGGRAPH_IMPORT_ERROR = e


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


def _safe_parse_json(content: str | None) -> Dict[str, Any]:
    if not content:
        return {}
    candidates: List[str] = [content]

    # 优先处理 ```json ... ``` 包裹场景。
    fenced_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", content, flags=re.IGNORECASE)
    if fenced_match:
        candidates.insert(0, fenced_match.group(1).strip())

    # 再尝试抽取首个 JSON 对象片段，降低模型前后赘述对解析的影响。
    start_index = content.find("{")
    end_index = content.rfind("}")
    if start_index != -1 and end_index != -1 and end_index > start_index:
        candidates.append(content[start_index : end_index + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue

    logger.warning("LangGraph POC 解析JSON失败，使用回退逻辑")
    return {}


def _normalize_attachments(raw_attachments: Any) -> List[str]:
    if isinstance(raw_attachments, str):
        return [raw_attachments]
    if isinstance(raw_attachments, list):
        return [str(item) for item in raw_attachments if str(item).strip()]
    return []


def _merge_attachment_paths(*path_groups: List[str]) -> List[str]:
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


def _extract_write_file_paths_from_tool_events(tool_events: List[ToolEvent]) -> List[str]:
    """从工具事件中提取写文件产物路径（用于附件兜底回传）。"""
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

    return _merge_attachment_paths(attachment_paths)


def _collect_plan_attachment_paths(plan: Optional[Plan]) -> List[str]:
    """汇总计划步骤中已经产出的附件路径。"""
    if plan is None:
        return []
    collected_paths: List[str] = []
    for step in plan.steps:
        if not step.done:
            continue
        collected_paths.extend(_normalize_attachments(step.attachments))
    return _merge_attachment_paths(collected_paths)


def _normalize_user_text(text: str) -> str:
    """归一化用户消息文本，便于做轻量意图判断。"""
    return re.sub(r"[\s\W_]+", "", text.strip().lower(), flags=re.UNICODE)


def bind_live_event_sink(
        sink: Optional[Callable[[BaseEvent], Optional[Awaitable[None]]]],
) -> Token:
    """绑定当前运行的实时事件输出回调。"""
    return _LIVE_EVENT_SINK.set(sink)


def unbind_live_event_sink(token: Token) -> None:
    """清理实时事件输出回调。"""
    _LIVE_EVENT_SINK.reset(token)


async def _emit_live_events(*events: BaseEvent) -> None:
    """将节点事件实时投递给运行时（best effort，不影响主链路）。"""
    sink = _LIVE_EVENT_SINK.get()
    if sink is None:
        return
    for event in events:
        try:
            emitted = sink(event)
            if inspect.isawaitable(emitted):
                await emitted
        except Exception as e:
            logger.warning("LangGraph POC 实时事件投递失败，继续主流程: %s", e)


def _is_simple_greeting_message(user_message: str) -> bool:
    """识别纯问候类输入，避免进入完整 planner/step 流程。"""
    normalized = _normalize_user_text(user_message)
    return normalized in _SIMPLE_GREETING_NORMALIZED_SET


def _build_simple_greeting_reply(user_message: str) -> str:
    """根据问候语种返回简洁回复。"""
    has_ascii_letters = re.search(r"[a-z]", user_message, flags=re.IGNORECASE) is not None
    has_cjk = re.search(r"[\u4e00-\u9fff]", user_message) is not None
    if has_ascii_letters and not has_cjk:
        return "Hello! I'm your assistant, happy to help."
    return "你好！我是助手，很高兴为您服务。"


def _build_fallback_plan_title(user_message: str) -> str:
    """在规划 JSON 缺失时生成可读标题，避免落到泛化默认值。"""
    normalized = user_message.strip()
    if not normalized:
        return "任务计划"
    title = normalized[:24]
    return title if len(normalized) <= 24 else f"{title}..."


def _get_last_assistant_message_event(events: List[Any]) -> Optional[MessageEvent]:
    """获取当前事件列表中最后一条 assistant message 事件。"""
    for event in reversed(events):
        if isinstance(event, MessageEvent) and event.role == "assistant":
            return event
    return None


def _collect_message_attachment_paths(message_event: Optional[MessageEvent]) -> List[str]:
    """提取消息事件中的附件路径，并按顺序去重。"""
    if message_event is None:
        return []
    attachment_paths: List[str] = []
    for attachment in list(message_event.attachments or []):
        filepath = str(getattr(attachment, "filepath", "") or "").strip()
        if filepath:
            attachment_paths.append(filepath)
    return _merge_attachment_paths(attachment_paths)


def _build_summarize_prompt(state: PlannerReActPOCState) -> str:
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


def _should_accept_summary_message(
        state: PlannerReActPOCState,
        candidate_message: str,
        fallback_message: str,
) -> bool:
    """判断是否接受模型总结文本，避免简单任务被无关长文覆盖。"""
    candidate = candidate_message.strip()
    if not candidate:
        return False

    # 单步任务通常没有必要产出超长总结；若出现明显异常长文，回退到步骤结果。
    execution_count = int(state.get("execution_count") or 0)
    if execution_count <= 1 and fallback_message.strip():
        # 单步任务优先与执行结果保持一致；若总结与步骤结果无明显关联，则拒绝覆盖。
        fallback = fallback_message.strip()
        if fallback not in candidate and candidate not in fallback:
            logger.warning("LangGraph summarize 与单步执行结果无关联，回退到步骤结果")
            return False

        max_allowed_length = len(fallback_message.strip()) * 3 + 120
        if len(candidate) > max_allowed_length:
            logger.warning("LangGraph summarize 产出异常长文本，回退到步骤结果以避免无关回复")
            return False

    return True


def _build_step_from_payload(payload: Any, fallback_index: int) -> Step:
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


def _format_attachments_for_prompt(attachments: List[str]) -> str:
    if not attachments:
        return "无"
    return "\n".join(f"- {item}" for item in attachments)


def _collect_available_tools(runtime_tools: List[BaseTool]) -> List[Dict[str, Any]]:
    """收集当前步骤可用的工具 schema 列表。"""
    available_tools: List[Dict[str, Any]] = []
    for tool in runtime_tools:
        try:
            available_tools.extend(tool.get_tools())
        except Exception as e:
            logger.warning("读取工具[%s] schema 失败，已跳过: %s", getattr(tool, "name", "unknown"), e)

    def _tool_priority(tool_schema: Dict[str, Any]) -> Tuple[int, str]:
        function_name = str(
            (tool_schema.get("function") or {}).get("name")
            if isinstance(tool_schema, dict)
            else ""
        ).strip().lower()
        # 搜索类工具优先，浏览器类工具后置：
        # 对“信息检索/课程推荐”等任务，优先走 search 能显著减少浏览器交互轮次与整体耗时。
        if "search" in function_name:
            return 0, function_name
        if function_name.startswith("browser_"):
            return 80, function_name
        return 20, function_name

    available_tools.sort(key=_tool_priority)
    return available_tools


def _extract_function_name(tool_schema: Dict[str, Any]) -> str:
    """提取工具 schema 的函数名（小写），失败时返回空串。"""
    if not isinstance(tool_schema, dict):
        return ""
    function = tool_schema.get("function")
    if not isinstance(function, dict):
        return ""
    return str(function.get("name") or "").strip().lower()


def _tool_call_priority(function_name: str) -> int:
    normalized_name = function_name.strip().lower()
    if "search" in normalized_name:
        return 0
    if normalized_name.startswith("browser_"):
        return 80
    return 20


def _pick_preferred_tool_call(
        tool_calls: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """从模型返回的多工具调用中挑选本轮优先执行的工具。

    策略：
    1. 仅在同一轮出现多个 tool_call 时启用；
    2. 优先 `search_*`，其次通用工具，最后 `browser_*`；
    3. 若函数名不在可用工具集合中，降级到最低优先级。
    """
    if len(tool_calls) == 0:
        return None
    if len(tool_calls) == 1:
        return tool_calls[0] if isinstance(tool_calls[0], dict) else None

    available_function_names: set[str] = set()
    for tool_schema in available_tools:
        function_name = _extract_function_name(tool_schema)
        if function_name:
            available_function_names.add(function_name)
    ranked_candidates: List[Tuple[int, int, Dict[str, Any]]] = []
    for index, raw_call in enumerate(tool_calls):
        if not isinstance(raw_call, dict):
            continue
        function = raw_call.get("function")
        if not isinstance(function, dict):
            continue
        function_name = str(function.get("name") or "").strip()
        if not function_name:
            continue

        normalized_name = function_name.lower()
        priority = _tool_call_priority(function_name)
        if normalized_name not in available_function_names:
            priority += 1000
        ranked_candidates.append((priority, index, raw_call))

    if len(ranked_candidates) == 0:
        return None

    ranked_candidates.sort(key=lambda item: (item[0], item[1]))
    selected_call = ranked_candidates[0][2]
    selected_function = str((selected_call.get("function") or {}).get("name") or "")
    logger.info(
        "LangGraph POC 多工具候选仲裁: total=%s, selected=%s",
        len(tool_calls),
        selected_function,
    )
    return selected_call


def _resolve_tool_by_function_name(function_name: str, runtime_tools: List[BaseTool]) -> Optional[BaseTool]:
    """根据函数名定位实际工具实例。"""
    for tool in runtime_tools:
        try:
            if tool.has_tool(function_name):
                return tool
        except Exception:
            continue
    return None


def _parse_tool_call_args(raw_arguments: Any) -> Dict[str, Any]:
    """解析 tool_call 参数，兼容 string/dict 两种输入。"""
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        try:
            parsed = json.loads(raw_arguments)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            logger.warning("LangGraph POC tool_call 参数解析失败，按空参数处理")
            return {}
    return {}


async def _execute_step_with_prompt(
        *,
        llm: LLM,
        user_message: str,
        step: Step,
        language: str,
        attachments: List[str],
        runtime_tools: Optional[List[BaseTool]] = None,
        max_tool_iterations: int = 5,
        on_tool_event: Optional[Callable[[ToolEvent], Optional[Awaitable[None]]]] = None,
) -> Tuple[Dict[str, Any], List[ToolEvent]]:
    """使用执行提示词完成步骤执行，支持可选工具调用循环。"""
    prompt = EXECUTION_PROMPT.format(
        message=user_message,
        attachments=_format_attachments_for_prompt(attachments),
        language=language,
        step=step.description,
    )

    if not runtime_tools:
        llm_message = await llm.invoke(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            response_format={"type": "json_object"},
        )
        parsed = _safe_parse_json(llm_message.get("content"))
        return {
            "success": bool(parsed.get("success", True)),
            "result": str(parsed.get("result") or f"已完成步骤：{step.description}"),
            "attachments": _normalize_attachments(parsed.get("attachments")),
        }, []

    available_tools = _collect_available_tools(runtime_tools)
    if len(available_tools) == 0:
        llm_message = await llm.invoke(
            messages=[{"role": "user", "content": prompt}],
            tools=[],
            response_format={"type": "json_object"},
        )
        parsed = _safe_parse_json(llm_message.get("content"))
        return {
            "success": bool(parsed.get("success", True)),
            "result": str(parsed.get("result") or f"已完成步骤：{step.description}"),
            "attachments": _normalize_attachments(parsed.get("attachments")),
        }, []

    messages: List[Dict[str, Any]] = [{"role": "user", "content": prompt}]
    emitted_tool_events: List[ToolEvent] = []
    llm_message: Dict[str, Any] = {}

    async def _notify_tool_event(event: ToolEvent) -> None:
        if on_tool_event is None:
            return
        try:
            maybe_awaitable = on_tool_event(event)
            if inspect.isawaitable(maybe_awaitable):
                await maybe_awaitable
        except Exception as e:
            logger.warning("LangGraph POC 投递实时工具事件失败，继续主流程: %s", e)

    # LangGraph POC 的最小工具循环：模型决定调用工具，执行后再回传继续推理。
    for _ in range(max(1, int(max_tool_iterations))):
        llm_message = await llm.invoke(
            messages=messages,
            tools=available_tools,
            tool_choice="auto",
        )
        tool_calls = llm_message.get("tool_calls") or []
        if len(tool_calls) == 0:
            break

        selected_tool_call = _pick_preferred_tool_call(
            tool_calls=[item for item in tool_calls if isinstance(item, dict)],
            available_tools=available_tools,
        )
        if selected_tool_call is None:
            continue

        messages.append(
            {
                "role": "assistant",
                "content": llm_message.get("content"),
                "tool_calls": [selected_tool_call],
            }
        )
        for raw_call in [selected_tool_call]:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function")
            if not isinstance(function, dict):
                continue

            function_name = str(function.get("name") or "").strip()
            if not function_name:
                continue
            tool_call_id = str(raw_call.get("id") or uuid.uuid4())
            function_args = _parse_tool_call_args(function.get("arguments"))

            matched_tool = _resolve_tool_by_function_name(function_name=function_name, runtime_tools=runtime_tools)
            tool_name = matched_tool.name if matched_tool is not None else "unknown"
            calling_event = ToolEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                function_name=function_name,
                function_args=function_args,
                status=ToolEventStatus.CALLING,
            )
            emitted_tool_events.append(calling_event)
            await _notify_tool_event(calling_event)

            if matched_tool is None:
                tool_result = ToolResult(success=False, message=f"无效工具: {function_name}")
            else:
                try:
                    tool_result = await matched_tool.invoke(function_name, **function_args)
                    if not isinstance(tool_result, ToolResult):
                        # 防御性收敛：兼容个别工具返回原始dict/str的历史写法。
                        tool_result = ToolResult(success=True, data=tool_result)
                except Exception as e:
                    logger.exception("LangGraph POC 调用工具[%s]失败: %s", function_name, e)
                    tool_result = ToolResult(success=False, message=f"调用工具失败: {function_name}")

            called_event = ToolEvent(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                function_name=function_name,
                function_args=function_args,
                function_result=tool_result,
                status=ToolEventStatus.CALLED,
            )
            emitted_tool_events.append(called_event)
            await _notify_tool_event(called_event)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "function_name": function_name,
                    "content": tool_result.model_dump_json(),
                }
            )

    parsed = _safe_parse_json(llm_message.get("content"))
    return {
        "success": bool(parsed.get("success", True)),
        "result": str(parsed.get("result") or f"已完成步骤：{step.description}"),
        "attachments": _normalize_attachments(parsed.get("attachments")),
    }, emitted_tool_events


def _should_emit_planner_message(user_message: str, planner_message: str, steps: List[Step]) -> bool:
    """判断是否需要对外输出规划阶段消息，避免简单问候场景的回显噪音。"""
    normalized_planner_message = planner_message.strip()
    if not normalized_planner_message:
        return False

    normalized_user_message = user_message.strip()
    # 规划消息与用户输入完全一致时，通常是模型回显，不需要额外展示。
    if normalized_user_message and normalized_planner_message == normalized_user_message:
        return False

    # 单步任务下若规划消息仅重复步骤描述，同样不输出，避免和步骤结果形成“多条问候”。
    if len(steps) == 1:
        first_step_description = str(steps[0].description or "").strip()
        if first_step_description and normalized_planner_message == first_step_description:
            return False

    return True


def _route_after_plan(state: PlannerReActPOCState) -> Literal["execute_step", "summarize", "finalize"]:
    plan = state.get("plan")
    if plan is None:
        # 无 plan 且已有最终回复时，直接结束，避免无意义 summarize。
        return "finalize" if str(state.get("final_message") or "").strip() else "summarize"
    if state.get("execution_count", 0) >= state.get("max_execution_steps", 20):
        logger.warning("LangGraph V1 执行次数达到上限，提前进入总结阶段")
        return "summarize"
    return "execute_step" if plan.get_next_step() is not None else "summarize"


def _route_after_replan(state: PlannerReActPOCState) -> Literal["execute_step", "summarize", "finalize"]:
    return _route_after_plan(state)


async def _create_or_reuse_plan_node(state: PlannerReActPOCState, llm: LLM) -> PlannerReActPOCState:
    """创建计划或复用已恢复计划。"""
    plan = state.get("plan")
    if plan is not None and len(plan.steps) > 0 and not plan.done:
        next_step = plan.get_next_step()
        return {
            **state,
            "current_step_id": next_step.id if next_step is not None else None,
        }

    user_message = state.get("user_message", "").strip()
    if _is_simple_greeting_message(user_message):
        # 纯问候直接回复并结束，不生成 plan/step，避免前端出现无意义步骤卡片。
        greeting_reply = _build_simple_greeting_reply(user_message=user_message)
        greeting_event = MessageEvent(role="assistant", message=greeting_reply)
        await _emit_live_events(greeting_event)
        return {
            **state,
            "plan": None,
            "current_step_id": None,
            "final_message": greeting_reply,
            "emitted_events": append_events(
                state.get("emitted_events"),
                greeting_event,
            ),
        }

    prompt = CREATE_PLAN_PROMPT.format(
        message=user_message,
        attachments="",
    )
    llm_message = await llm.invoke(
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        response_format={"type": "json_object"},
    )
    parsed = _safe_parse_json(llm_message.get("content"))

    title = str(parsed.get("title") or _build_fallback_plan_title(user_message))
    language = str(parsed.get("language") or "zh")
    goal = str(parsed.get("goal") or user_message)
    planner_message = str(parsed.get("message") or user_message or "已生成任务计划")
    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list) or len(raw_steps) == 0:
        raw_steps = [user_message or "处理用户任务"]

    steps = [_build_step_from_payload(item, index) for index, item in enumerate(raw_steps)]
    plan = Plan(
        title=title,
        goal=goal,
        language=language,
        message=planner_message,
        steps=steps,
        status=ExecutionStatus.PENDING,
    )
    next_step = plan.get_next_step()
    planner_events: List[Any] = [TitleEvent(title=title)]
    if _should_emit_planner_message(
            user_message=user_message,
            planner_message=planner_message,
            steps=steps,
    ):
        planner_events.append(MessageEvent(role="assistant", message=planner_message))
    planner_events.append(PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.CREATED))
    await _emit_live_events(*planner_events)

    return {
        **state,
        "plan": plan,
        "current_step_id": next_step.id if next_step is not None else None,
        "emitted_events": append_events(
            state.get("emitted_events"),
            *planner_events,
        ),
    }


async def _execute_step_node(
        state: PlannerReActPOCState,
        llm: LLM,
        skill_runtime: Optional[SkillGraphRuntime] = None,
        runtime_tools: Optional[List[BaseTool]] = None,
        max_tool_iterations: int = 5,
) -> PlannerReActPOCState:
    """执行单个步骤，完成后交给 replan 节点更新后续计划。"""
    plan = state.get("plan")
    if plan is None:
        return state

    step = plan.get_next_step()
    if step is None:
        return state

    step.status = ExecutionStatus.RUNNING
    started_event = StepEvent(
        step=step.model_copy(deep=True),
        status=StepEventStatus.STARTED,
    )
    await _emit_live_events(started_event)

    user_message = str(state.get("user_message", ""))
    language = plan.language or "zh"
    attachments: List[str] = []
    execution_payload: Optional[Dict[str, Any]] = None
    tool_events: List[ToolEvent] = []

    # 若运行时已注入工具能力，则优先走“提示词 + 工具循环”路径。
    if runtime_tools:
        execution_payload, tool_events = await _execute_step_with_prompt(
            llm=llm,
            user_message=user_message,
            step=step,
            language=language,
            attachments=attachments,
            runtime_tools=runtime_tools,
            max_tool_iterations=max_tool_iterations,
            on_tool_event=_emit_live_events,
        )

    # 无工具能力时保持原 BE-LG-10 skill 路径。
    if execution_payload is None and skill_runtime is not None:
        try:
            skill_result = await skill_runtime.execute_skill(
                skill_id=PLANNER_EXECUTE_STEP_SKILL_ID,
                payload={
                    "session_id": str(state.get("session_id") or ""),
                    "user_message": user_message,
                    "step_description": step.description,
                    "language": language,
                    "attachments": attachments,
                },
            )
            execution_payload = {
                "success": bool(getattr(skill_result, "success", True)),
                "result": str(getattr(skill_result, "result", "") or f"已完成步骤：{step.description}"),
                "attachments": _normalize_attachments(getattr(skill_result, "attachments", [])),
            }
        except Exception as e:
            logger.warning("执行步骤 Skill 运行失败，回退默认执行链路: %s", e)

    if execution_payload is None:
        execution_payload, tool_events = await _execute_step_with_prompt(
            llm=llm,
            user_message=user_message,
            step=step,
            language=language,
            attachments=attachments,
            on_tool_event=_emit_live_events,
        )

    step.success = bool(execution_payload.get("success", True))
    step.result = str(execution_payload.get("result") or f"已完成步骤：{step.description}")
    model_attachment_paths = _normalize_attachments(execution_payload.get("attachments"))
    inferred_attachment_paths = _extract_write_file_paths_from_tool_events(tool_events)
    # 执行结果若未显式给出附件，兜底从 write_file 工具结果提取产物路径。
    step.attachments = _merge_attachment_paths(model_attachment_paths, inferred_attachment_paths)
    step.status = ExecutionStatus.COMPLETED if step.success else ExecutionStatus.FAILED

    completed_event = StepEvent(
        step=step.model_copy(deep=True),
        status=StepEventStatus.COMPLETED if step.success else StepEventStatus.FAILED,
    )
    final_step_events: List[Any] = [completed_event]
    if step.result:
        final_step_events.append(
            MessageEvent(
                role="assistant",
                message=step.result,
                attachments=[File(filepath=filepath) for filepath in step.attachments],
            )
        )
    await _emit_live_events(*final_step_events)
    events: List[Any] = [started_event, *tool_events, *final_step_events]
    next_step = plan.get_next_step()

    return {
        **state,
        "plan": plan,
        "last_executed_step": step.model_copy(deep=True),
        "execution_count": int(state.get("execution_count", 0)) + 1,
        "current_step_id": next_step.id if next_step is not None else None,
        "final_message": step.result or "",
        "emitted_events": append_events(
            state.get("emitted_events"),
            *events,
        ),
    }


async def _replan_node(state: PlannerReActPOCState, llm: LLM) -> PlannerReActPOCState:
    """根据最新步骤执行结果更新后续未完成步骤。"""
    plan = state.get("plan")
    last_step = state.get("last_executed_step")
    if plan is None or last_step is None:
        return state

    prompt = UPDATE_PLAN_PROMPT.format(
        step=last_step.model_dump_json(),
        plan=plan.model_dump_json(),
    )
    llm_message = await llm.invoke(
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        response_format={"type": "json_object"},
    )
    parsed = _safe_parse_json(llm_message.get("content"))

    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list):
        return state

    new_steps = [_build_step_from_payload(item, index) for index, item in enumerate(raw_steps)]
    first_pending_index: Optional[int] = None
    for index, current_step in enumerate(plan.steps):
        if not current_step.done:
            first_pending_index = index
            break

    if first_pending_index is not None:
        updated_steps = plan.steps[:first_pending_index]
        updated_steps.extend(new_steps)
        plan.steps = updated_steps

    next_step = plan.get_next_step()
    updated_event = PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.UPDATED)
    await _emit_live_events(updated_event)
    return {
        **state,
        "plan": plan,
        "current_step_id": next_step.id if next_step is not None else None,
        "emitted_events": append_events(
            state.get("emitted_events"),
            updated_event,
        ),
    }


async def _summarize_node(state: PlannerReActPOCState, llm: LLM) -> PlannerReActPOCState:
    """在所有步骤完成后汇总结果。"""
    plan = state.get("plan")
    prompt = _build_summarize_prompt(state)
    llm_message = await llm.invoke(
        messages=[{"role": "user", "content": prompt}],
        tools=[],
        response_format={"type": "json_object"},
    )
    parsed = _safe_parse_json(llm_message.get("content"))
    fallback_message = str(state.get("final_message") or "任务已完成")
    candidate_summary_message = str(parsed.get("message") or "")
    accepted_candidate = _should_accept_summary_message(
        state=state,
        candidate_message=candidate_summary_message,
        fallback_message=fallback_message,
    )
    summary_message = candidate_summary_message if accepted_candidate else fallback_message
    summary_attachment_paths = _normalize_attachments(parsed.get("attachments")) if accepted_candidate else []
    # summarize 未返回附件时，兜底使用步骤执行阶段已产出的附件。
    attachment_paths = _merge_attachment_paths(
        summary_attachment_paths,
        _collect_plan_attachment_paths(plan),
    )
    attachments = [File(filepath=filepath) for filepath in attachment_paths]
    previous_assistant_event = _get_last_assistant_message_event(list(state.get("emitted_events") or []))
    previous_assistant_message = str(previous_assistant_event.message or "").strip() if previous_assistant_event else ""
    previous_assistant_attachment_paths = _collect_message_attachment_paths(previous_assistant_event)

    final_events: List[Any] = []
    # summarize 与上一条 assistant 在“文本 + 附件”都一致时，跳过重复 message，
    # 避免前端出现两条完全相同的最终回复。
    if not (
            summary_message.strip()
            and previous_assistant_message
            and summary_message.strip() == previous_assistant_message
            and attachment_paths == previous_assistant_attachment_paths
    ):
        final_events.append(MessageEvent(role="assistant", message=summary_message, attachments=attachments))

    if plan is not None:
        plan.status = ExecutionStatus.COMPLETED
        final_events.append(PlanEvent(plan=plan.model_copy(deep=True), status=PlanEventStatus.COMPLETED))
    await _emit_live_events(*final_events)

    return {
        **state,
        "plan": plan,
        "current_step_id": None,
        "final_message": summary_message,
        "emitted_events": append_events(state.get("emitted_events"), *final_events),
    }


async def _finalize_node(state: PlannerReActPOCState) -> PlannerReActPOCState:
    """结束节点，追加 done 事件。"""
    events = list(state.get("emitted_events") or [])
    if events and isinstance(events[-1], DoneEvent):
        return state

    done_event = DoneEvent()
    await _emit_live_events(done_event)
    return {
        **state,
        "emitted_events": append_events(state.get("emitted_events"), done_event),
    }


def build_planner_react_poc_graph(
        llm: LLM,
        runtime_tools: Optional[List[BaseTool]] = None,
        max_tool_iterations: int = 5,
) -> Any:
    """构建 LangGraph Planner-ReAct V1 图（沿用 POC 编译入口）。"""
    if not LANGGRAPH_AVAILABLE:
        raise RuntimeError(f"LangGraph 未安装，无法构建 POC 图: {LANGGRAPH_IMPORT_ERROR}")

    skill_runtime: Optional[SkillGraphRuntime] = None
    try:
        skill_runtime = build_default_skill_graph_registry().create_runtime(llm=llm)
    except Exception as e:
        logger.warning("初始化默认 Skill 注册表失败，继续使用无 Skill 模式: %s", e)

    # 显式 async wrapper，避免 lambda 返回 coroutine 导致节点返回值非法。
    async def _create_plan_with_llm(state: PlannerReActPOCState) -> PlannerReActPOCState:
        return await _create_or_reuse_plan_node(state, llm)

    async def _execute_step_with_llm(state: PlannerReActPOCState) -> PlannerReActPOCState:
        return await _execute_step_node(
            state,
            llm,
            skill_runtime=skill_runtime,
            runtime_tools=runtime_tools,
            max_tool_iterations=max_tool_iterations,
        )

    async def _replan_with_llm(state: PlannerReActPOCState) -> PlannerReActPOCState:
        return await _replan_node(state, llm)

    async def _summarize_with_llm(state: PlannerReActPOCState) -> PlannerReActPOCState:
        return await _summarize_node(state, llm)

    graph = StateGraph(PlannerReActPOCState)
    graph.add_node("create_plan_or_reuse", _create_plan_with_llm)
    graph.add_node("execute_step", _execute_step_with_llm)
    graph.add_node("replan", _replan_with_llm)
    graph.add_node("summarize", _summarize_with_llm)
    graph.add_node("finalize", _finalize_node)
    graph.add_edge(START, "create_plan_or_reuse")
    graph.add_conditional_edges(
        "create_plan_or_reuse",
        _route_after_plan,
        {
            "execute_step": "execute_step",
            "summarize": "summarize",
            "finalize": "finalize",
        },
    )
    graph.add_edge("execute_step", "replan")
    graph.add_conditional_edges(
        "replan",
        _route_after_replan,
        {
            "execute_step": "execute_step",
            "summarize": "summarize",
            "finalize": "finalize",
        },
    )
    graph.add_edge("summarize", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=InMemorySaver())
