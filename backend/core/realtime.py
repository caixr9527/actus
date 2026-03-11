#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/11
@Author : caixiaorong01@outlook.com
@File   : realtime.py
"""

# 会话列表实时变更通知通道
SESSION_LIST_CHANGE_CHANNEL = "session:list:changed"

# SSE 订阅端在无通知时的兜底校准间隔（秒）
SESSION_LIST_FALLBACK_REFRESH_SECONDS = 30.0
