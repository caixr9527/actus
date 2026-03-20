#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Application 层 DTO 定义
"""

from .app_config import A2AServerItemResult, MCPServerItemResult
from .auth import LoginResult, RefreshResult, RegisterVerificationCodeResult
from .session import ConsoleRecordResult, FileReadResult, ShellReadResult

__all__ = [
    "A2AServerItemResult",
    "MCPServerItemResult",
    "LoginResult",
    "RefreshResult",
    "RegisterVerificationCodeResult",
    "ConsoleRecordResult",
    "FileReadResult",
    "ShellReadResult",
]
