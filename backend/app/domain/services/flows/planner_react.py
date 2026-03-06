#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/1/27 10:09
@Author : caixiaorong01@outlook.com
@File   : planner_react.py
"""
import logging
from typing import AsyncGenerator, Optional, Callable

from app.domain.external import Sandbox, Browser, SearchEngine, LLM, JSONParser
from app.domain.models import (
    Message,
    BaseEvent,
    Plan,
    AgentConfig,
    SessionStatus,
    DoneEvent,
    PlanEvent,
    PlanEventStatus,
    TitleEvent,
    MessageEvent,
    ExecutionStatus
)
from app.domain.repositories import IUnitOfWork
from app.domain.services.agents import PlannerAgent, ReActAgent
from app.domain.services.tools import (
    BrowserTool,
    FileTool,
    ShellTool,
    SearchTool,
    MCPTool,
    A2ATool,
    MessageTool,
)
from .base import BaseFlow, FlowStatus

logger = logging.getLogger(__name__)


class PlannerReActFlow(BaseFlow):

    def __init__(
            self,
            llm: LLM,
            agent_config: AgentConfig,
            session_id: str,
            uow_factory: Callable[[], IUnitOfWork],
            json_parser: JSONParser,
            browser: Browser,
            sandbox: Sandbox,
            search_engine: SearchEngine,
            mcp_tool: MCPTool,
            a2a_tool: A2ATool,
    ):
        # 初始化会话ID和会话仓库，用于后续的交互和状态管理
        self._session_id = session_id
        self._uow_factory = uow_factory

        # 设置流程初始状态为待机状态，并初始化计划为空
        self.status = FlowStatus.IDLE
        self.plan: Optional[Plan] = None

        # 构建工具列表，包括文件操作、shell命令执行、浏览器控制、搜索功能、消息处理、MCP和A2A工具
        tools = [
            FileTool(sandbox=sandbox),  # 文件操作工具
            ShellTool(sandbox=sandbox),  # Shell命令执行工具
            BrowserTool(browser=browser),  # 浏览器控制工具
            SearchTool(search_engine=search_engine),  # 搜索引擎工具
            MessageTool(),  # 消息处理工具
            mcp_tool,  # MCP工具
            a2a_tool,  # A2A工具
        ]

        # 创建规划代理实例，负责制定任务计划
        self.planner = PlannerAgent(
            session_id=self._session_id,
            uow_factory=self._uow_factory,
            agent_config=agent_config,
            llm=llm,
            tools=tools,
            json_parser=json_parser,
        )
        logger.debug(f"创建PlannerAgent成功, 会话id: {self._session_id}")

        # 创建ReAct代理实例，负责执行计划并进行推理和行动
        self.react = ReActAgent(
            session_id=self._session_id,
            uow_factory=self._uow_factory,
            agent_config=agent_config,
            llm=llm,
            tools=tools,
            json_parser=json_parser,
        )

        logger.debug(f"创建ReActAgent成功, 会话id: {self._session_id}")

    async def invoke(self, message: Message) -> AsyncGenerator[BaseEvent, None]:
        # 获取当前会话信息，如果会话不存在则抛出异常
        async with self._uow_factory() as uow:
            session = await uow.session.get_by_id(self._session_id)
        if not session:
            raise ValueError(f"会话不存在: {self._session_id}, 请确认会话ID是否正确")

        # 如果会话不是PENDING状态，则回滚planner和react的消息，确保消息列表格式正常
        if session.status != SessionStatus.PENDING:
            logger.debug(f"会话 {self._session_id} 未处于空闲状态, 回滚数据确保消息列表格式正常")
            await self.planner.roll_back(message=message)
            await self.react.roll_back(message=message)

        # 根据会话状态设置当前flow的状态
        if session.status == SessionStatus.RUNNING:
            logger.debug(f"会话 {self._session_id} 正在运行中, 重新规划")
            self.status = FlowStatus.PLANNING

        if session.status == SessionStatus.WAITING:
            logger.debug(f"会话 {self._session_id} 正在等待中, 继续执行")
            self.status = FlowStatus.EXECUTING

        # 更新会话状态为RUNNING
        # 采用独立短事务更新会话状态，避免跨调用复用同一UoW实例。
        async with self._uow_factory() as uow:
            await uow.session.update_status(self._session_id, SessionStatus.RUNNING)

        # 获取最新的计划
        self.plan = session.get_latest_plan()
        logger.info(f"Planner&ReAct流接收消息：{message.message[:50]}...")

        # 初始化step变量
        step = None

        # 主循环：根据flow的不同状态执行相应操作
        while True:
            # IDLE状态：切换到PLANNING状态
            if self.status == FlowStatus.IDLE:
                logger.info(f"Planner&ReAct流状态变更 {FlowStatus.IDLE} -> {FlowStatus.PLANNING}")
                self.status = FlowStatus.PLANNING

            # PLANNING状态：创建计划
            elif self.status == FlowStatus.PLANNING:
                logger.info(f"Planner&ReAct流开始创建计划/Plan")
                async for event in self.planner.create_plan(message=message):
                    # 当计划创建完成后，保存计划并发送标题和消息事件
                    if isinstance(event, PlanEvent) and event.status == PlanEventStatus.CREATED:
                        self.plan = event.plan
                        logger.info(f"Planner&ReAct流创建计划成功, 共计: {len(event.plan.steps)} 步")

                        yield TitleEvent(title=event.plan.title)
                        yield MessageEvent(role="assistant", message=event.plan.message)

                    yield event

                logger.info(f"Planner&ReAct流状态变更 {FlowStatus.PLANNING} -> {FlowStatus.EXECUTING}")
                self.status = FlowStatus.EXECUTING

                # 如果计划或步骤为空，直接进入完成状态
                if not self.plan or len(self.plan.steps) == 0:
                    logger.info(f"Planner&ReAct流计划或子步骤为空")
                    self.status = FlowStatus.COMPLETED

            # EXECUTING状态：执行计划中的下一步
            elif self.status == FlowStatus.EXECUTING:
                self.plan.status = ExecutionStatus.RUNNING

                # 获取下一个待执行的步骤
                step = self.plan.get_next_step()
                if not step:
                    logger.info(
                        f"Planner&ReAct流没有更多步骤,状态变更 {FlowStatus.EXECUTING} -> {FlowStatus.SUMMARIZING}")
                    self.status = FlowStatus.SUMMARIZING
                    continue

                logger.info(f"Planner&ReAct流开始执行步骤 {step.id}: {step.description[:50]}...")
                # 执行当前步骤
                async for event in self.react.execute_step(plan=self.plan, step=step, message=message):
                    yield event

                logger.info(f"压缩{self.react.name} Agent记忆...")
                # 压缩ReAct Agent的记忆，释放资源
                await self.react.compact_memory()

                # 切换到UPDATING状态，准备更新计划
                self.status = FlowStatus.UPDATING

            # UPDATING状态：更新计划
            elif self.status == FlowStatus.UPDATING:
                logger.info(f"Planner&ReAct流开始更新计划")
                # 使用planner更新计划
                async for event in self.planner.update_plan(plan=self.plan, step=step):
                    yield event

                logger.info(f"Planner&ReAct流状态变更 {FlowStatus.UPDATING} -> {FlowStatus.EXECUTING}")
                self.status = FlowStatus.EXECUTING

            # SUMMARIZING状态：总结执行结果
            elif self.status == FlowStatus.SUMMARIZING:
                logger.info(f"Planner&ReAct流开始总结")
                # 调用react代理进行总结
                async for event in self.react.summarize():
                    yield event

                logger.info(f"Planner&ReAct流状态变更 {FlowStatus.SUMMARIZING} -> {FlowStatus.COMPLETED}")
                self.status = FlowStatus.COMPLETED

            # COMPLETED状态：完成流程
            elif self.status == FlowStatus.COMPLETED:
                logger.info(f"Planner&ReAct流完成")
                # 设置计划状态为已完成，并重置flow状态为IDLE
                self.plan.status = ExecutionStatus.COMPLETED
                self.status = FlowStatus.IDLE
                # 发送完成事件
                yield PlanEvent(status=PlanEventStatus.COMPLETED, plan=self.plan)
                break

        # 发送结束事件
        yield DoneEvent()
        logger.info(f"Planner&ReAct流完成")

    @property
    def done(self) -> bool:
        return self.status == FlowStatus.IDLE
