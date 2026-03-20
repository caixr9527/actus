#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 23:09
@Author : caixiaorong01@outlook.com
@File   : verification_code_manager.py
"""
import secrets


class VerificationCodeManager:
    """验证码工具类"""

    @classmethod
    def generate_numeric_code(cls, length: int = 6) -> str:
        """生成固定长度数字验证码"""
        if length <= 0:
            raise ValueError("验证码长度必须大于0")
        start = 10 ** (length - 1)
        end = (10 ** length) - 1
        return str(secrets.randbelow(end - start + 1) + start)
