#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/13 23:07
@Author : caixiaorong01@outlook.com
@File   : smtp_email_sender.py
"""
import asyncio
import smtplib
from email.message import EmailMessage

from app.domain.external import EmailSender
from core.config import get_settings


class SMTPEmailSender(EmailSender):
    """基于 SMTP 的邮件发送实现"""

    def __init__(self) -> None:
        self._setting = get_settings()

    async def send_register_verification_code(
            self,
            to_email: str,
            verification_code: str,
            expires_in_seconds: int,
    ) -> None:
        self._validate_config()

        message = EmailMessage()
        message["Subject"] = "Actus 注册验证码"
        message["From"] = self._setting.smtp_from_email
        message["To"] = to_email
        message.set_content(
            f"您的注册验证码为：{verification_code}\n"
            f"该验证码将在 {expires_in_seconds // 60} 分钟后失效。\n"
            "若非本人操作，请忽略此邮件。"
        )

        await asyncio.to_thread(self._send_message, message)

    def _validate_config(self) -> None:
        if not self._setting.smtp_host or not self._setting.smtp_port or not self._setting.smtp_from_email:
            raise RuntimeError("邮件服务未配置，请先配置 SMTP_HOST/SMTP_PORT/SMTP_FROM_EMAIL")

    def _send_message(self, message: EmailMessage) -> None:
        with smtplib.SMTP(self._setting.smtp_host, self._setting.smtp_port, timeout=10) as smtp:
            if self._setting.smtp_use_tls:
                smtp.starttls()
            if self._setting.smtp_from_email and self._setting.smtp_password:
                smtp.login(self._setting.smtp_from_email, self._setting.smtp_password)
            smtp.send_message(message)
