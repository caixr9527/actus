#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
认证相关 DTO
"""

from pydantic import BaseModel

from app.domain.models import User, UserProfile


class LoginResult(BaseModel):
    user: User
    profile: UserProfile
    access_token: str
    refresh_token: str
    access_token_expires_in: int
    refresh_token_expires_in: int


class RegisterVerificationCodeResult(BaseModel):
    verification_required: bool
    expires_in_seconds: int


class RefreshResult(BaseModel):
    access_token: str
    refresh_token: str
    access_token_expires_in: int
    refresh_token_expires_in: int

