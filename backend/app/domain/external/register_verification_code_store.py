#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 23:03
@Author : caixiaorong01@outlook.com
@File   : register_verification_code_store.py
"""
from abc import ABC, abstractmethod


class RegisterVerificationCodeStore(ABC):
    """注册验证码存储接口"""

    @abstractmethod
    async def save_verification_code(
            self,
            email: str,
            verification_code: str,
            expires_in_seconds: int,
    ) -> None:
        """保存邮箱验证码"""
        ...

    @abstractmethod
    async def verify_and_consume_verification_code(
            self,
            email: str,
            verification_code: str,
    ) -> bool:
        """校验并消费验证码（单次有效）"""
        ...
