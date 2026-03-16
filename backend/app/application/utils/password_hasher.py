#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 20:12
@Author : caixiaorong01@outlook.com
@File   : password_hasher.py
"""
from argon2 import PasswordHasher as Argon2PasswordHasher
from argon2.exceptions import VerificationError


class PasswordHasher:
    """密码处理工具类（Argon2id）"""

    _HASHER = Argon2PasswordHasher(
        # Argon2id 推荐参数：兼顾开发环境可用性与基础抗 GPU 破解能力。
        time_cost=3,
        memory_cost=65536,
        parallelism=4,
        hash_len=32,
        salt_len=16,
    )

    @classmethod
    def hash_password(cls, password: str) -> str:
        """计算密码哈希（返回 Argon2 编码串，包含算法参数与盐值）"""
        return cls._HASHER.hash(password)

    @classmethod
    def verify_password(cls, password: str, password_hash: str) -> bool:
        """校验明文密码与 Argon2 哈希是否匹配"""
        try:
            return cls._HASHER.verify(password_hash, password)
        except (VerificationError, ValueError):
            return False
