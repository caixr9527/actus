#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/1/26 14:24
@Author : caixiaorong01@outlook.com
@File   : db_session_repository.py
"""
import logging
from datetime import datetime
from typing import Awaitable, Callable, List, Optional

from sqlalchemy import select, delete, update, func, cast
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.models import Session, SessionStatus, BaseEvent, File, Memory
from app.domain.repositories import SessionRepository
from app.infrastructure.models import SessionModel
from app.infrastructure.storage.redis import get_redis_client
from core.realtime import SESSION_LIST_CHANGE_CHANNEL

logger = logging.getLogger(__name__)


class DBSessionRepository(SessionRepository):

    def __init__(self, db_session: AsyncSession) -> None:
        self.db_session = db_session

    def _register_session_list_changed(self, session_id: str) -> None:
        """注册会话列表变更通知，在事务提交成功后再执行发布。"""
        changed_ids = self.db_session.info.setdefault("session_list_changed_ids", set())
        if session_id in changed_ids:
            return
        changed_ids.add(session_id)

        post_commit_hooks: list[Callable[[], Awaitable[None]]] = self.db_session.info.setdefault(
            "post_commit_hooks",
            [],
        )

        async def _publish() -> None:
            try:
                redis_client = get_redis_client()
                await redis_client.client.publish(SESSION_LIST_CHANGE_CHANNEL, session_id)
            except Exception as e:
                # 通知失败不影响主事务结果，SSE 会回退到周期兜底刷新。
                logger.debug(f"发布会话列表变更通知失败（已忽略）: {e}")

        post_commit_hooks.append(_publish)

    async def save(self, session: Session) -> None:
        # 查询数据库中是否存在具有相同ID的会话记录
        stmt = select(SessionModel).where(SessionModel.id == session.id)
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()

        # 如果记录不存在，则创建新的会话模型并添加到数据库会话中
        if not record:
            record = SessionModel.from_domain(session)
            self.db_session.add(record)
        else:
            # 如果记录已存在，则用新的域对象更新现有的数据库记录
            record.update_from_domain(session)

        self._register_session_list_changed(session.id)

    async def get_all(self, user_id: Optional[str] = None) -> List[Session]:
        """获取所有会话列表"""
        # 构建查询语句，按最新消息时间降序排列
        stmt = select(SessionModel)
        if user_id is not None:
            stmt = stmt.where(SessionModel.user_id == user_id)
        stmt = stmt.order_by(SessionModel.latest_message_at.desc())
        # 执行查询
        result = await self.db_session.execute(stmt)
        # 获取所有结果记录
        records = result.scalars().all()

        # 将数据库模型转换为领域模型并返回
        return [record.to_domain() for record in records]

    async def get_by_id(self, session_id: str, user_id: Optional[str] = None) -> Optional[Session]:
        """根据id查询会话"""
        # 构建查询语句，根据session_id查找对应的会话记录
        stmt = select(SessionModel).where(SessionModel.id == session_id)
        if user_id is not None:
            stmt = stmt.where(SessionModel.user_id == user_id)
        # 执行查询
        result = await self.db_session.execute(stmt)
        # 获取查询结果，如果不存在则返回None
        record = result.scalar_one_or_none()

        # 如果找到了记录，则转换为领域模型并返回；否则返回None
        return record.to_domain() if record is not None else None

    async def delete_by_id(self, session_id: str) -> None:
        """根据传递的id删除会话"""
        # 构建删除语句，根据session_id删除对应的会话记录
        stmt = delete(SessionModel).where(SessionModel.id == session_id)

        # 执行删除操作
        result = await self.db_session.execute(stmt)
        if result.rowcount:
            self._register_session_list_changed(session_id)

    async def update_title(self, session_id: str, title: str) -> None:
        """更新会话标题"""
        # 构建更新语句，根据session_id更新对应的会话标题
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(title=title)
        )
        # 执行更新操作
        result = await self.db_session.execute(stmt)

        # 检查是否有行被更新，如果没有则抛出异常
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")
        self._register_session_list_changed(session_id)

    async def update_latest_message(self, session_id: str, message: str, timestamp: datetime) -> None:
        """更新会话最新消息"""
        # 构建更新语句，根据session_id更新对应的会话最新消息内容和时间戳
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                latest_message=message,
                latest_message_at=timestamp,
            )
        )
        # 执行更新操作
        result = await self.db_session.execute(stmt)

        # 检查是否有行被更新，如果没有则抛出异常
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")
        self._register_session_list_changed(session_id)

    async def add_event(self, session_id: str, event: BaseEvent) -> None:
        """往会话中新增事件"""
        # 将事件对象转换为JSON格式的数据
        event_data = event.model_dump(mode="json")

        # 构建更新语句，将会话的events字段更新为原events列表加上新事件数据的新列表
        # 使用coalesce函数处理events字段为空的情况，如果为空则视为空列表
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                events=func.coalesce(SessionModel.events, cast([], JSONB)) + cast([event_data], JSONB),
            )
        )
        # 执行更新操作
        result = await self.db_session.execute(stmt)

        # 检查是否有行被更新，如果没有则抛出异常
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def add_event_if_absent(self, session_id: str, event: BaseEvent) -> bool:
        """按事件ID幂等新增事件"""
        # 这里使用行级锁而不是直接JSONB append，目的是保证“存在性判断+写入”原子化。
        stmt = (
            select(SessionModel)
            .where(SessionModel.id == session_id)
            .with_for_update()
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        if not record:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

        # 读取当前事件列表，缺省时按空列表处理。
        current_events = list(record.events or [])
        event_id = str(event.id)

        # 幂等判断：只要事件ID已存在就视为已处理，直接返回False。
        for current_event in current_events:
            if str(current_event.get("id", "")) == event_id:
                return False

        # 事件不存在才追加，确保同一event_id不会重复入库。
        current_events.append(event.model_dump(mode="json"))
        record.events = current_events
        return True

    async def add_event_with_snapshot_if_absent(
            self,
            session_id: str,
            event: BaseEvent,
            title: Optional[str] = None,
            latest_message: Optional[str] = None,
            latest_message_at: Optional[datetime] = None,
            increment_unread: bool = False,
            status: Optional[SessionStatus] = None,
    ) -> bool:
        """按事件ID幂等新增事件，并在同一事务中更新会话投影"""
        # 与add_event_if_absent一致：使用悲观锁保证并发安全和幂等判断准确性。
        stmt = (
            select(SessionModel)
            .where(SessionModel.id == session_id)
            .with_for_update()
        )
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()
        if not record:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

        # 事件幂等判断与投影更新要分离：
        # 1) 事件历史只允许按event_id写入一次；
        # 2) 会话投影(status/title/latest)在幂等重放时仍应收敛到最新值。
        current_events = list(record.events or [])
        event_id = str(event.id)
        event_inserted = True
        for current_event in current_events:
            if str(current_event.get("id", "")) == event_id:
                event_inserted = False
                break

        # 事件不存在才写入历史，保证event_id幂等。
        if event_inserted:
            current_events.append(event.model_dump(mode="json"))
            record.events = current_events

        # 标题属于会话投影的一部分，按调用方意图进行更新。
        if title is not None:
            record.title = title

        # 最新消息投影用于会话列表展示，只在显式传入时更新。
        if latest_message is not None:
            record.latest_message = latest_message

        # 时间戳与最新消息成对更新，避免出现文本和时间不同步。
        if latest_message_at is not None:
            record.latest_message_at = latest_message_at

        # 未读数只能在事件首次写入时递增，避免重放消息导致重复累计。
        if increment_unread and event_inserted:
            record.unread_message_count = int(record.unread_message_count or 0) + 1

        # 状态投影在同一事务内收敛。即使事件已存在，也允许补齐状态。
        if status is not None:
            record.status = status.value

        # 返回值只表示“事件历史是否新增”，不表示投影是否更新。
        self._register_session_list_changed(session_id)
        return event_inserted

    async def add_file(self, session_id: str, file: File) -> None:
        """往会话中新增文件"""

        """往会话中新增文件"""
        # 将文件对象转换为JSON格式的数据
        file_data = file.model_dump(mode="json")

        # 构建更新语句，将文件添加到会话的files字段中
        # 使用coalesce函数处理files字段为空的情况，如果为空则视为空列表
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                files=func.coalesce(SessionModel.files, cast([], JSONB)) + cast([file_data], JSONB),
            )
        )
        # 执行更新操作
        result = await self.db_session.execute(stmt)

        # 检查是否有行被更新，如果没有则抛出异常
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def remove_file(self, session_id: str, file_id: str) -> None:
        """移除会话中的指定文件"""
        # 查询会话记录并加锁，防止并发修改
        stmt = select(SessionModel).where(SessionModel.id == session_id).with_for_update()
        result = await self.db_session.execute(stmt)
        record = result.scalar_one_or_none()

        # 如果会话不存在，抛出异常
        if not record:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

        # 如果会话没有文件，直接返回
        if not record.files:
            return

        # 记录原始文件数量，用于判断是否真正删除了文件
        original_length = len(record.files)

        # 过滤掉要删除的文件，保留其他文件
        new_files = [file for file in record.files if file.get("id") != file_id]

        # 如果文件数量没有变化，说明要删除的文件不存在，直接返回
        if len(new_files) == original_length:
            return

        # 更新会话记录中的文件列表
        record.files = new_files

    async def get_file_by_path(self, session_id: str, filepath: str) -> Optional[File]:
        """根据文件路径获取文件信息"""
        # 构建查询语句，根据session_id获取会话的所有文件列表
        stmt = select(SessionModel.files).where(SessionModel.id == session_id)
        # 执行查询操作
        result = await self.db_session.execute(stmt)
        # 获取查询结果
        files = result.scalar_one_or_none()

        # 如果没有找到文件列表，返回None
        if not files:
            return None

        # 遍历文件列表，查找匹配指定文件路径的文件
        for file in files:
            if file.get("filepath", "") == filepath:
                # 找到匹配的文件，将其转换为File对象并返回
                return File(**file)

        # 如果没有找到匹配的文件，返回None
        return None

    async def update_status(self, session_id: str, status: SessionStatus) -> None:
        """更新会话状态"""
        # 构建更新语句，根据session_id更新对应的会话状态
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(status=status.value)
        )
        # 执行更新操作
        result = await self.db_session.execute(stmt)

        # 检查是否有行被更新，如果没有则抛出异常
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")
        self._register_session_list_changed(session_id)

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        """更新会话的未读消息数"""
        # 构建更新语句，根据session_id更新对应的会话未读消息数量
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(unread_message_count=count)
        )
        # 执行更新操作
        result = await self.db_session.execute(stmt)

        # 检查是否有行被更新，如果没有则抛出异常
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")
        self._register_session_list_changed(session_id)

    async def increment_unread_message_count(self, session_id: str) -> None:
        """新增会话的未读消息数"""
        # 构建更新语句，根据session_id增加对应的会话未读消息数量
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                unread_message_count=func.coalesce(SessionModel.unread_message_count, 0) + 1,
            )
        )
        # 执行更新操作
        result = await self.db_session.execute(stmt)

        # 检查是否有行被更新，如果没有则抛出异常
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")
        self._register_session_list_changed(session_id)

    async def decrement_unread_message_count(self, session_id: str) -> None:
        """将会话中的未读消息数-1"""
        # 构建更新语句，将会话的未读消息数减1，但不能小于0
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                unread_message_count=func.greatest(
                    func.coalesce(SessionModel.unread_message_count, 0) - 1,
                    0
                )
            )
        )
        # 执行更新操作
        result = await self.db_session.execute(stmt)

        # 检查是否有行被更新，如果没有则抛出异常
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")
        self._register_session_list_changed(session_id)

    async def save_memory(self, session_id: str, agent_name: str, memory: Memory) -> None:
        """存储或者更新会话中的记忆(字典直接覆盖)"""

        # 将记忆对象转换为JSON格式数据
        memory_data = memory.model_dump(mode="json")

        # 创建补丁数据，以代理名称作为键，记忆数据作为值
        patch_data = {agent_name: memory_data}

        # 构建更新语句，将会话的记忆字段更新为原记忆字典与新记忆数据的合并
        # 使用coalesce函数处理memories字段为空的情况，如果为空则视为默认空字典
        stmt = (
            update(SessionModel)
            .where(SessionModel.id == session_id)
            .values(
                memories=func.coalesce(SessionModel.memories, cast({}, JSONB)) + cast(patch_data, JSONB),
            )
        )
        # 执行更新操作
        result = await self.db_session.execute(stmt)

        # 检查是否有行被更新，如果没有则抛出异常
        if result.rowcount == 0:
            raise ValueError(f"会话[{session_id}]不存在，请核实后重试")

    async def get_memory(self, session_id: str, agent_name: str) -> Memory:
        """获取指定会话的agent记忆信息"""

        # 构建查询语句，从会话中获取指定代理的记忆数据
        stmt = (
            select(SessionModel.memories[agent_name])
            .where(SessionModel.id == session_id)
        )
        # 执行查询操作
        result = await self.db_session.execute(stmt)
        # 获取查询结果，如果不存在则返回None
        memory_data = result.scalar_one_or_none()

        # 如果找到了记忆数据，则将其转换为Memory对象并返回
        if memory_data:
            return Memory(**memory_data)

        # 如果没有找到对应的记忆数据，返回一个空的记忆对象
        return Memory(messages=[])
