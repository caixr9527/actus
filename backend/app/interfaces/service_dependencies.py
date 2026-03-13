#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/17 10:45
@Author : caixiaorong01@outlook.com
@File   : service_dependencies.py
"""
import logging
from functools import lru_cache
from typing import cast

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession
from starlette.datastructures import State

from app.application.service import AppConfigService, FileService, StatusService, AgentService, AuthService
from app.application.service.session_service import SessionService
from app.infrastructure.external.email_sender import SMTPEmailSender
from app.infrastructure.external.file_storage import CosFileStorage
from app.infrastructure.external.health_checker import PostgresHealthChecker, RedisHealthChecker
from app.infrastructure.external.json_parser import RepairJsonParser
from app.infrastructure.external.llm import OpenAILLM
from app.infrastructure.external.search import BingSearchEngine
from app.infrastructure.external.task import RedisStreamTask
from app.infrastructure.external.token_store import RedisRefreshTokenStore
from app.infrastructure.external.verification_code_store import RedisRegisterVerificationCodeStore
from app.infrastructure.repositories import FileAppConfigRepository
from app.infrastructure.sandbox.docker_sandbox import DockerSandbox
from app.infrastructure.storage import get_db_session, RedisClient, get_redis_client, Cos, get_cos, get_uow
from core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


def _get_app_state(app: object) -> State:
    """获取应用 state 对象，便于静态检查工具识别。"""
    state = getattr(app, "state", None)
    if state is None:
        raise RuntimeError("应用对象缺少 state 属性，无法获取 AgentService")
    return cast(State, state)


@lru_cache()
def get_app_config_service() -> AppConfigService:
    """获取应用配置服务"""
    logger.info("加载获取AppConfigService")
    return AppConfigService(app_config_repository=FileAppConfigRepository(settings.app_config_filepath))


def get_status_service(
        db_session: AsyncSession = Depends(get_db_session),
        redis_client: RedisClient = Depends(get_redis_client)
) -> StatusService:
    """获取应用状态服务"""
    logger.info("加载获取StatusService")
    postgres_checker = PostgresHealthChecker(db_session=db_session)
    redis_checker = RedisHealthChecker(redis_client=redis_client)
    return StatusService(checkers=[postgres_checker, redis_checker])


@lru_cache()
def get_file_service(
        cos: Cos = Depends(get_cos),
) -> FileService:
    # 初始化文件仓库和文件存储桶
    file_storage = CosFileStorage(
        bucket=settings.cos_bucket,
        cos=cos,
        uow_factory=get_uow,
    )

    # 构建服务并返回
    return FileService(
        uow_factory=get_uow,
        file_storage=file_storage,
    )


@lru_cache()
def get_session_service() -> SessionService:
    """获取会话服务"""
    logger.info("加载获取SessionService")
    return SessionService(uow_factory=get_uow, sandbox_cls=DockerSandbox)


@lru_cache()
def get_auth_service(redis_client: RedisClient = Depends(get_redis_client)) -> AuthService:
    """获取认证服务"""
    logger.info("加载获取AuthService")
    refresh_token_store = RedisRefreshTokenStore(redis_client=redis_client)
    register_verification_code_store = RedisRegisterVerificationCodeStore(redis_client=redis_client)
    email_sender = SMTPEmailSender()
    return AuthService(
        uow_factory=get_uow,
        refresh_token_store=refresh_token_store,
        register_verification_code_store=register_verification_code_store,
        email_sender=email_sender,
    )


def build_agent_service(cos: Cos) -> AgentService:
    """纯构造函数：根据显式传入依赖构建 AgentService。"""
    app_config_repository = FileAppConfigRepository(config_path=settings.app_config_filepath)
    app_config = app_config_repository.load()

    llm = OpenAILLM(app_config.llm_config)
    file_storage = CosFileStorage(
        bucket=settings.cos_bucket,
        cos=cos,
        uow_factory=get_uow,
    )

    return AgentService(
        llm=llm,
        agent_config=app_config.agent_config,
        mcp_config=app_config.mcp_config,
        a2a_config=app_config.a2a_config,
        sandbox_cls=DockerSandbox,
        task_cls=RedisStreamTask,
        json_parser=RepairJsonParser(),
        search_engine=BingSearchEngine(),
        file_storage=file_storage,
        uow_factory=get_uow,
    )


@lru_cache()
def get_agent_service_for_lifespan() -> AgentService:
    """生命周期使用：构建并缓存单例 AgentService。"""
    logger.info("加载获取AgentService")
    return build_agent_service(cos=get_cos())


def clear_agent_service_for_lifespan_cache() -> None:
    """清理生命周期 AgentService 缓存。"""
    get_agent_service_for_lifespan.cache_clear()


def get_agent_service(request: Request) -> AgentService:
    """路由依赖适配：优先读取 app.state，其次回退到生命周期单例。"""
    app_service = getattr(_get_app_state(request.app), "agent_service", None)
    if app_service is not None:
        return app_service
    return get_agent_service_for_lifespan()
