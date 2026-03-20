#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 21:32
@Author : caixiaorong01@outlook.com
@File   : auth_token_manager.py
"""
import uuid
from datetime import datetime, timedelta, timezone

import jwt


class AuthTokenManager:
    """认证令牌工具类"""

    @classmethod
    def generate_access_token(
            cls,
            user_id: str,
            email: str,
            secret_key: str,
            algorithm: str,
            expires_in_seconds: int,
    ) -> str:
        """生成 Access Token（JWT）"""
        now = datetime.now(timezone.utc)
        payload = {
            "sub": user_id,
            "user_id": user_id,
            "email": email,
            "type": "access",
            "iat": int(now.timestamp()),
            "exp": int((now + timedelta(seconds=expires_in_seconds)).timestamp()),
        }
        return jwt.encode(payload, secret_key, algorithm=algorithm)

    @classmethod
    def generate_refresh_token(cls) -> str:
        """生成 Refresh Token（UUID）"""
        return str(uuid.uuid4())
