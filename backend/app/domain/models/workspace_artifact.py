#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/4/12 17:30
@Author : caixiaorong01@outlook.com
@File   : workspace_artifact.py
"""
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class WorkspaceArtifact(BaseModel):
    """Workspace 产物索引。"""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workspace_id: str
    path: str
    artifact_type: str
    summary: str = ""
    source_step_id: Optional[str] = None
    source_capability: Optional[str] = None
    delivery_state: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
