#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
会话相关 DTO
"""

from typing import List

from pydantic import BaseModel, Field


class FileReadResult(BaseModel):
    filepath: str
    content: str


class ConsoleRecordResult(BaseModel):
    ps1: str
    command: str
    output: str


class ShellReadResult(BaseModel):
    session_id: str
    output: str
    console_records: List[ConsoleRecordResult] = Field(default_factory=list)
