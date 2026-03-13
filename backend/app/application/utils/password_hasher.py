#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 20:12
@Author : caixiaorong01@outlook.com
@File   : password_hasher.py
"""
import hashlib
import secrets


class PasswordHasher:
    """密码处理工具类（PBKDF2-HMAC-SHA256）"""

    _PBKDF2_ITERATIONS = 200_000
    _SALT_BYTES = 16

    @classmethod
    def generate_password_salt(cls) -> str:
        """生成密码盐值（十六进制字符串）"""
        return secrets.token_hex(cls._SALT_BYTES)

    @classmethod
    def hash_password_with_salt(cls, password: str, salt: str) -> str:
        """基于PBKDF2-HMAC-SHA256计算密码哈希"""
        hashed = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            cls._PBKDF2_ITERATIONS,
        )
        return hashed.hex()
