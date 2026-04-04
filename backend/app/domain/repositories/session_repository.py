#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/1/25 16:25
@Author : caixiaorong01@outlook.com
@File   : session_repository.py
"""
from datetime import datetime
from typing import Protocol, List, Optional

from app.domain.models import BaseEvent, File
from app.domain.models import Session, SessionStatus


class SessionRepository(Protocol):
    """会话仓库协议定义"""

    async def save(self, session: Session) -> None:
        """存储或更新传递进来的会话"""
        ...

    async def get_all(self, user_id: Optional[str] = None) -> List[Session]:
        """获取所有会话列表信息"""
        ...

    async def get_by_id(self, session_id: str, user_id: Optional[str] = None) -> Optional[Session]:
        """根据传递的会话id查询会话"""
        ...

    async def delete_by_id(self, session_id: str) -> None:
        """根据传递的会话id删除会话"""
        ...

    async def update_title(self, session_id: str, title: str) -> None:
        """根据传递的会话id+标题更新会话信息"""
        ...

    async def update_latest_message(self, session_id: str, message: str, timestamp: datetime) -> None:
        """根据传递的信息更新最新消息"""
        ...

    async def update_unread_message_count(self, session_id: str, count: int) -> None:
        """根据传递的信息更新未读消息数"""
        ...

    async def increment_unread_message_count(self, session_id: str) -> None:
        """根据传递的会话id新增未读消息数"""
        ...

    async def decrement_unread_message_count(self, session_id: str) -> None:
        """根据传递的会话id减少未读消息数"""
        ...

    async def update_status(self, session_id: str, status: SessionStatus) -> None:
        """根据传递的会话id更新会话状态"""
        ...

    async def update_current_model_id(self, session_id: str, current_model_id: Optional[str]) -> None:
        """根据传递的会话id更新当前模型id"""
        ...

    async def add_event(self, session_id: str, event: BaseEvent) -> None:
        """往会话中新增事件"""
        ...

    async def add_event_if_absent(self, session_id: str, event: BaseEvent) -> bool:
        """按事件ID幂等地新增事件，返回值表示是否新增成功"""
        ...

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
        """按事件ID幂等新增事件，并在同一事务中更新会话投影字段

        返回值仅表示“事件历史是否新增成功”，不代表投影字段是否被更新。
        """
        ...

    async def add_file(self, session_id: str, file: File) -> None:
        """往会话中新增文件"""
        ...

    async def add_final_files(self, session_id: str, file: File) -> None:
        """往会话中新增文件"""
        ...

    async def remove_file(self, session_id: str, file_id: str) -> None:
        """根据传递的会话id+文件id移除文件"""
        ...

    async def get_file_by_path(self, session_id: str, filepath: str) -> Optional[File]:
        """查询会话中的文件信息"""
        ...
