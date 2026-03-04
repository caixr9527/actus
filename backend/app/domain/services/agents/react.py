#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/25 15:24
@Author : caixiaorong01@outlook.com
@File   : react.py
"""
import logging
from typing import AsyncGenerator

from app.domain.models import Plan, Step, Message, Event, ExecutionStatus, File
from app.domain.models.event import (
    StepEventStatus,
    StepEvent,
    ToolEvent,
    MessageEvent,
    ErrorEvent,
    ToolEventStatus,
    WaitEvent
)
from app.domain.services.prompts import SYSTEM_PROMPT, REACT_SYSTEM_PROMPT, EXECUTION_PROMPT, SUMMARIZE_PROMPT
from .base import BaseAgent

logger = logging.getLogger(__name__)


class ReActAgent(BaseAgent):
    """
    ReAct Agent
    """
    name: str = "ReAct"
    _system_prompt: str = SYSTEM_PROMPT + REACT_SYSTEM_PROMPT
    _format: str = "json_object"

    async def execute_step(self, plan: Plan, step: Step, message: Message) -> AsyncGenerator[Event, None]:
        """
        执行ReAct Agent的单个步骤
        
        :param plan: 包含完整任务计划的对象
        :param step: 当前需要执行的步骤对象
        :param message: 包含用户输入和上下文信息的消息对象
        :return: 异步生成器，产生执行过程中的事件流
        """
        # 构造执行提示词，包含用户消息、附件、语言和步骤描述
        query = EXECUTION_PROMPT.format(
            message=message.message,
            attachments="\n".join(message.attachments),
            language=plan.language,
            step=step.description
        )

        # 设置步骤状态为运行中，并产出步骤开始事件
        step.status = ExecutionStatus.RUNNING
        yield StepEvent(
            step=step,
            status=StepEventStatus.STARTED,
        )

        # 执行推理调用并处理产生的事件流
        async for event in self.invoke(query):
            # 处理工具调用事件
            if isinstance(event, ToolEvent):
                # 特殊处理询问用户消息的工具调用
                if event.function_name == "message_ask_user":
                    # 工具调用中：产出消息事件
                    if event.status == ToolEventStatus.CALLING:
                        yield MessageEvent(
                            role="assistant",
                            message=event.function_args.get("text", ""),
                        )
                    # 工具已调用：产出等待事件并结束当前步骤
                    elif event.status == ToolEventStatus.CALLED:
                        yield WaitEvent()
                        return
                    continue
            # 处理消息事件（通常是模型返回的结果）
            elif isinstance(event, MessageEvent):
                # 设置步骤状态为已完成
                step.status = ExecutionStatus.COMPLETED
                # 解析模型返回的JSON格式响应
                parsed_obj = await self._json_parser.invoke(event.message)
                # 将解析结果转换为Step对象
                new_step = Step.model_validate(parsed_obj)

                # 更新当前步骤的成功状态、结果和附件
                step.success = new_step.success
                step.result = new_step.result
                step.attachments = new_step.attachments

                # 产出步骤完成事件
                yield StepEvent(
                    step=step,
                    status=StepEventStatus.COMPLETED,
                )

                # 如果步骤有结果，则产出对应的消息事件
                if step.result:
                    yield MessageEvent(
                        role="assistant",
                        message=step.result,
                    )
                continue
            # 处理错误事件
            elif isinstance(event, ErrorEvent):
                # 设置步骤状态为失败，并记录错误信息
                step.status = ExecutionStatus.FAILED
                step.error = event.error
                # 产出步骤失败事件
                yield StepEvent(
                    step=step,
                    status=StepEventStatus.FAILED,
                )
            # 透传其他类型的事件
            yield event

        # 确保步骤最终状态为完成（如果没有提前返回或出错）
        step.status = ExecutionStatus.COMPLETED

    async def summarize(self) -> AsyncGenerator[Event, None]:
        """
        总结ReAct Agent的运行结果

        :return: 异步生成器，产生总结过程中的事件流
        """
        # 构造总结提示词
        query = SUMMARIZE_PROMPT
        # 调用模型执行总结任务
        async for event in self.invoke(query):
            if isinstance(event, MessageEvent):
                # 记录生成的总结内容
                logger.info(f"执行Agent生成汇总内容：{event.message}")
                # 解析模型返回的JSON格式响应
                parsed_obj = await self._json_parser.invoke(event.message)

                # 将解析结果转换为Message对象
                message = Message.model_validate(parsed_obj)

                # 根据附件路径创建File对象列表 todo
                attachments = [File(filepath=filepath) for filepath in message.attachments]
                # 产出包含总结结果的消息事件
                yield MessageEvent(
                    role="assistant",
                    message=message.message,
                    attachments=attachments,
                )
            else:
                # 透传其他类型的事件
                yield event
