#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/24 17:26
@Author : caixiaorong01@outlook.com
@File   : planner.py
"""
import logging
from typing import Optional, AsyncGenerator

from app.domain.models import Message, Event, MessageEvent, Plan, PlanEvent, PlanEventStatus, Step
from app.domain.services.prompts import SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT, CREATE_PLAN_PROMPT, UPDATE_PLAN_PROMPT
from .base import BaseAgent

"""
多Agent系统/flow=PlannerAgent+ReActAgent

顺序:
1. PlannerAgent生成规划;
2. 循环取出规划中的子步骤，让ReActAgent执行，依次迭代;
3. ReActAgent执行完每一个子步骤之后，需要将子步骤结果+Plan传递给PlannerAgent让其更新计划/Plan；
4. 循环取出规划中的子步骤，让ReActAgent执行，依次迭代;
5. ...
6. 直到所有子任务/步骤都完成，这时候将子步骤的所有结果汇总进行总结(ReActAgent);

PlannerAgent:
- 功能: 将用户的需求拆解成多个子任务+根据已完成的子任务更新规划
- 提示词: 创建规划的prompt、更新规划的prompt

ReActAgent:
- 功能: 迭代执行完每一个子任务、汇总所有的子任务进行总结
- 提示词: 执行任务的prompt、汇总总结prompt
"""
logger = logging.getLogger(__name__)


class PlannerAgent(BaseAgent):
    """规划者,用于将用户的任务/需求拆解为多个子任务"""
    name: str = "planner"
    _system_prompt: str = SYSTEM_PROMPT + PLANNER_SYSTEM_PROMPT
    _format: Optional[str] = "json_object"
    _tool_choice: Optional[str] = "none"

    async def create_plan(self, message: Message) -> AsyncGenerator[Event, None]:
        """
        根据用户消息创建执行计划
        :param message: 包含用户请求的Message对象
        :return: 异步生成器，产生规划相关的事件流
        """
        # 构造创建计划的查询提示词，包含用户消息内容和附件信息
        query = CREATE_PLAN_PROMPT.format(
            message=message.message,
            attachments="\n".join(message.attachments),
        )

        # 调用大模型生成计划，并处理返回的事件流
        async for event in self.invoke(query):
            if isinstance(event, MessageEvent):
                # 记录PlannerAgent生成的消息日志
                logger.info(f"PlannerAgent生成消息: {event.message}")
                # 解析模型返回的JSON格式消息
                parsed_obj = await self._json_parser.invoke(event.message)
                # 验证并转换为Plan对象
                plan = Plan.model_validate(parsed_obj)
                # 产出计划创建事件
                yield PlanEvent(
                    plan=plan,
                    status=PlanEventStatus.CREATED
                )
            else:
                # 产出其他类型的事件
                yield event

    async def update_plan(self, plan: Plan, step: Step) -> AsyncGenerator[Event, None]:
        """
        根据步骤执行结果更新计划
        :param plan: 当前计划
        :param step: 当前步骤
        :return: 异步生成器，产生更新计划相关事件流
        """
        # 构造更新计划的查询提示词，包含当前步骤和计划信息
        query = UPDATE_PLAN_PROMPT.format(
            step=step.model_dump_json(),
            plan=plan.model_dump_json(),
        )
        # 调用大模型更新计划，并处理返回的事件流
        async for event in self.invoke(query):
            if isinstance(event, MessageEvent):
                # 记录PlannerAgent生成的消息日志
                logger.info(f"PlannerAgent生成消息: {event.message}")
                # 解析模型返回的JSON格式消息
                parsed_obj = await self._json_parser.invoke(event.message)
                # 验证并转换为Plan对象
                update_plan = Plan.model_validate(parsed_obj)

                # 复制一份，避免造成数据污染
                new_steps = [Step.model_validate(step) for step in update_plan.steps]

                # 查找原计划中第一个未完成步骤的索引
                first_pending_index = None
                for idx, step in enumerate(plan.steps):
                    if not step.done:
                        first_pending_index = idx
                        break

                # 如果找到未完成的步骤，则用新步骤替换原计划中从该步骤开始的所有步骤
                if first_pending_index is not None:
                    updated_steps = plan.steps[:first_pending_index]
                    updated_steps.extend(new_steps)

                    plan.steps = updated_steps

                # 产出计划更新事件
                yield PlanEvent(
                    plan=plan,
                    status=PlanEventStatus.UPDATED
                )
            else:
                # 产出其他类型的事件
                yield event
