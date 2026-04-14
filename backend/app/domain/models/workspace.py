#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/4/12 17:30
@Author : caixiaorong01@outlook.com
@File   : workspace.py
"""
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class WorkspaceStatus(str, Enum):
    """工作区状态。"""

    ACTIVE = "active"
    EXPIRED = "expired"
    DESTROYED = "destroyed"


class Workspace(BaseModel):
    """Workspace 环境聚合。"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    current_run_id: Optional[str] = None
    sandbox_id: Optional[str] = None
    task_id: Optional[str] = None
    shell_session_id: Optional[str] = None
    cwd: str = ""
    browser_snapshot: Dict[str, Any] = Field(default_factory=dict)
    environment_summary: Dict[str, Any] = Field(default_factory=dict)
    status: WorkspaceStatus = WorkspaceStatus.ACTIVE
    last_active_at: datetime = Field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
