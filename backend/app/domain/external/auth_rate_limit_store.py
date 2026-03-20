#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/16 15:20
@Author : caixiaorong01@outlook.com
@File   : auth_rate_limit_store.py
"""
from typing import Protocol


class AuthRateLimitStore(Protocol):
    """认证限流存储协议"""

    async def get_login_attempt_count_by_ip(self, ip: str) -> int:
        """获取登录失败次数（IP 维度）"""
        ...

    async def get_login_attempt_count_by_email(self, email: str) -> int:
        """获取登录失败次数（邮箱维度）"""
        ...

    async def increase_login_attempt_count(
            self,
            ip: str | None,
            email: str,
            expires_in_seconds: int,
    ) -> None:
        """增加登录失败次数（同时更新 IP + 邮箱维度）"""
        ...

    async def clear_login_attempt_count(self, ip: str | None, email: str) -> None:
        """清空登录失败次数（登录成功后）"""
        ...

    async def increase_register_send_code_attempt_count_by_ip(
            self,
            ip: str,
            expires_in_seconds: int,
    ) -> int:
        """增加发送注册验证码次数（IP 维度）"""
        ...
