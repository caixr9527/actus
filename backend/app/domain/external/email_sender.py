#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 23:04
@Author : caixiaorong01@outlook.com
@File   : email_sender.py
"""
from abc import ABC, abstractmethod


class EmailSender(ABC):
    """邮件发送接口"""

    @abstractmethod
    async def send_register_verification_code(
            self,
            to_email: str,
            verification_code: str,
            expires_in_seconds: int,
    ) -> None:
        """发送注册验证码邮件"""
        ...
