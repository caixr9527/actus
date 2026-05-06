#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Runtime 访问范围领域契约。"""

from __future__ import annotations

from pydantic import BaseModel


class AccessScopeResult(BaseModel):
    """已校验的数据访问范围。"""

    tenant_id: str
    user_id: str
    session_id: str | None = None
    workspace_id: str | None = None
    run_id: str | None = None
    current_step_id: str | None = None
