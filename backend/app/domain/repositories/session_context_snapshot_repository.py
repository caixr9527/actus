#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""会话上下文快照仓储协议。"""
from typing import Optional, Protocol

from app.domain.models import SessionContextSnapshot


class SessionContextSnapshotRepository(Protocol):
    """会话上下文快照仓储协议定义。"""

    async def get_by_session_id(self, session_id: str) -> Optional[SessionContextSnapshot]:
        """根据 session_id 读取会话上下文快照。"""
        ...

    async def upsert(self, snapshot: SessionContextSnapshot) -> SessionContextSnapshot:
        """按 session_id 幂等写入会话上下文快照。"""
        ...
