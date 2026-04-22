#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""记忆沉淀提示词模板。

结构正确性由 Ollama structured output + MemoryConsolidationResult Pydantic schema 约束；
本提示词只约束字段业务语义，避免 prompt 与代码合同重复维护。
"""

MEMORY_CONSOLIDATION_PROMPT = """
你正在执行会话记忆沉淀。你的输出会被结构化 schema 校验，只返回符合 schema 的 JSON 对象。

## 任务定位

你不是 planner、executor、summary 或 feedback 节点。你只负责把一次运行结束后的上下文沉淀为后续对话可复用的信息。

## 输入说明

输入 JSON 中可能包含：
- user_message：本轮用户原始输入
- assistant_message：本轮最终回复或轻量最终摘要
- previous_conversation_summary：上一轮会话摘要
- message_window：近期对话窗口
- selected_artifacts：本轮最终交付附件路径
- goal：本轮任务目标
- completed_step_count / total_step_count：本轮执行进度
- facts_in_session：已有会话事实
- user_preferences：已有用户偏好
- pending_memory_writes：上游已经明确产生的长期记忆候选

## 字段语义

conversation_summary：
- 写会话级摘要，用于下一轮理解上下文。
- 只保留主题、用户目标、已确认约束、关键结论和当前进度。
- 不要复制完整最终正文。
- 不要写工具日志、内部路径、临时失败细节或模型推测。
- 如果 previous_conversation_summary 有价值，应在此基础上增量更新，而不是简单重复。

facts_in_session：
- 写当前会话内后续仍可能复用的稳定事实。
- 只写已经明确出现或由执行结果稳定确认的信息。
- 不写临时过程、一次性工具错误、猜测或未确认信息。

user_preferences：
- 只写用户明确表达的长期偏好，例如语言、回答风格、技术栈倾向。
- 不要把本轮任务约束误写成长期偏好。
- 不确定是否长期有效时不要写入。

memory_candidates：
- 只生成适合长期记忆仓储的候选。
- 每条候选必须有 namespace、memory_type、summary、content、confidence。
- memory_type 只能表达 profile、fact、instruction 之一。
- confidence 应反映确定性，不确定内容应降低置信度或不生成。
- 如果没有明确长期价值，返回空数组。

message_window：
- 保留近期 user/assistant 对话窗口。
- 不需要塞入工具轨迹或完整文件内容。
- 如果已有窗口可用，应在其基础上保留关键近期对话。

## 禁止事项

- 禁止编造用户偏好。
- 禁止把完整最终回答当成 conversation_summary。
- 禁止把搜索结果、工具失败、内部 prompt 或敏感路径写成长期记忆。
- 禁止为了填字段而填字段；没有可靠内容时使用空数组或空对象。
- 禁止输出 Markdown、解释文字或 schema 之外的字段。

## 正例

用户说“以后都用中文简洁回答”，可以写入：
- user_preferences: {{"language": "zh", "response_style": "concise"}}
- memory_candidates 中生成 profile 类型候选，summary 为“用户偏好中文简洁回复”。

本轮只是“预算 3000 元以内安排一次旅行”，不要把“预算 3000”写成长期偏好；它只属于当前任务约束，可进入 conversation_summary。

## 输入 JSON

{payload_json}
"""
