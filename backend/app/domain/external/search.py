#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/27 18:44
@Author : caixiaorong01@outlook.com
@File   : search.py
"""
from typing import Protocol, Optional

from app.domain.models import ToolResult, SearchResults


class SearchEngine(Protocol):

    async def invoke(self, query: str, date_range: Optional[str] = None) -> ToolResult[SearchResults]:
        ...
