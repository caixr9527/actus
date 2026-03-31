#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/17 23:40
@Author : caixiaorong01@outlook.com
@File   : error_keys.py
"""

# 认证链路
AUTH_MISSING_CREDENTIALS = "error.auth.missing_credentials"
AUTH_INVALID_AUTHORIZATION_HEADER = "error.auth.invalid_authorization_header"
AUTH_ACCESS_TOKEN_REQUIRED = "error.auth.access_token_required"
AUTH_ACCESS_TOKEN_EXPIRED = "error.auth.access_token_expired"
AUTH_ACCESS_TOKEN_INVALID = "error.auth.access_token_invalid"
AUTH_TOKEN_TYPE_INVALID = "error.auth.token_type_invalid"
AUTH_TOKEN_USER_MISSING = "error.auth.token_user_missing"
AUTH_SESSION_INVALIDATED = "error.auth.session_invalidated"
AUTH_SERVICE_UNAVAILABLE = "error.auth.service_unavailable"
AUTH_USER_NOT_FOUND = "error.auth.user_not_found"
AUTH_USER_STATUS_INVALID = "error.auth.user_status_invalid"
AUTH_REFRESH_SESSION_MISSING = "error.auth.refresh_session_missing"
AUTH_PASSWORD_MISMATCH = "error.auth.password_mismatch"
AUTH_EMAIL_ALREADY_REGISTERED = "error.auth.email_already_registered"
AUTH_REGISTER_CODE_REQUIRED = "error.auth.register_verification_code_required"
AUTH_REGISTER_CODE_INVALID = "error.auth.register_verification_code_invalid"
AUTH_SEND_CODE_FAILED = "error.auth.send_code_failed"
AUTH_LOGIN_INVALID_CREDENTIALS = "error.auth.invalid_credentials"
AUTH_LOGIN_FAILED = "error.auth.login_failed"
AUTH_REFRESH_TOKEN_REQUIRED = "error.auth.refresh_token_required"
AUTH_REFRESH_FAILED = "error.auth.refresh_failed"
AUTH_REFRESH_TOKEN_INVALID = "error.auth.refresh_token_invalid"
AUTH_REFRESH_REPLAYED = "error.auth.refresh_replayed"
AUTH_LOGOUT_FAILED = "error.auth.logout_failed"
AUTH_SERVICE_NOT_CONFIGURED = "error.auth.service_not_configured"
AUTH_REGISTER_CODE_SERVICE_NOT_CONFIGURED = "error.auth.register_code_service_not_configured"
AUTH_EMAIL_SERVICE_NOT_CONFIGURED = "error.auth.email_service_not_configured"
AUTH_LOGIN_RATE_LIMITED = "error.auth.login_rate_limited"
AUTH_SEND_CODE_RATE_LIMITED = "error.auth.send_code_rate_limited"
AUTH_HTTPS_REQUIRED = "error.auth.https_required"

# 用户链路
USER_NOT_FOUND = "error.user.not_found"
USER_PROFILE_UPDATE_EMPTY = "error.user.profile_update_empty"
USER_LOCALE_UNSUPPORTED = "error.user.locale_unsupported"
USER_PASSWORD_MISMATCH = "error.user.new_password_mismatch"
USER_CURRENT_PASSWORD_INCORRECT = "error.user.current_password_incorrect"
USER_SESSION_CLEANUP_FAILED = "error.user.session_cleanup_failed"

# 会话链路
SESSION_NOT_FOUND = "error.session.not_found"
SESSION_MODEL_ID_INVALID = "error.session.model_id_invalid"
SESSION_SANDBOX_NOT_BOUND = "error.session.sandbox_not_bound"
SESSION_SANDBOX_UNAVAILABLE = "error.session.sandbox_unavailable"
SESSION_FILE_READ_FAILED = "error.session.file_read_failed"
SESSION_SHELL_READ_FAILED = "error.session.shell_read_failed"
SESSION_RESUME_REQUIRED = "error.session.resume_required"
SESSION_NOT_WAITING = "error.session.not_waiting"
SESSION_RESUME_CHECKPOINT_INVALID = "error.session.resume_checkpoint_invalid"

# 文件链路
FILE_NOT_FOUND = "error.file.not_found"

# 应用配置链路
APP_CONFIG_LOAD_FAILED = "error.app_config.load_failed"
APP_CONFIG_SAVE_FAILED = "error.app_config.save_failed"
APP_CONFIG_MODEL_NOT_FOUND = "error.app_config.model_not_found"
APP_CONFIG_MODEL_INVALID = "error.app_config.model_invalid"
APP_CONFIG_DEFAULT_MODEL_UNAVAILABLE = "error.app_config.default_model_unavailable"
APP_CONFIG_MCP_SERVER_NOT_FOUND = "error.app_config.mcp_server_not_found"
APP_CONFIG_A2A_SERVER_NOT_FOUND = "error.app_config.a2a_server_not_found"
APP_CONFIG_MCP_SERVERS_LOAD_FAILED = "error.app_config.mcp_servers_load_failed"
APP_CONFIG_A2A_SERVERS_LOAD_FAILED = "error.app_config.a2a_servers_load_failed"

# 其他接口链路
STATUS_UNHEALTHY = "error.status.unhealthy"
