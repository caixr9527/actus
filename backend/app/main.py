#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/12 16:56
@Author : caixiaorong01@outlook.com
@File   : main.py
"""
import asyncio
import logging
from contextlib import asynccontextmanager, suppress
from typing import cast

from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.datastructures import State

from app.infrastructure.logging import setup_logging
from app.infrastructure.runtime.langgraph import get_langgraph_checkpointer
from app.infrastructure.storage import get_redis_client, get_postgres, get_cos
from app.interfaces.dependencies.auth_guard import validate_api_auth_coverage
from app.interfaces.dependencies.services import (
    get_agent_service_for_lifespan,
    clear_agent_service_for_lifespan_cache,
)
from app.interfaces.endpoints.routes import router
from app.interfaces.errors.exception_handlers import register_exception_handlers
from core.config import get_settings

settings = get_settings()
is_production_env = settings.env.lower() in {"production", "prod"}
enable_api_docs = not is_production_env


def _parse_cors_allowed_origins(origins: str) -> list[str]:
    items = [item.strip() for item in origins.split(",") if item.strip()]
    return items


cors_allowed_origins = _parse_cors_allowed_origins(settings.cors_allowed_origins)
if not cors_allowed_origins:
    raise RuntimeError("CORS_ALLOWED_ORIGINS 不能为空")
if settings.cors_allow_credentials and "*" in cors_allowed_origins:
    raise RuntimeError("CORS_ALLOW_CREDENTIALS=true 时，CORS_ALLOWED_ORIGINS 不能包含 '*'")
if is_production_env and "*" in cors_allowed_origins:
    raise RuntimeError("生产环境下 CORS_ALLOWED_ORIGINS 禁止使用 '*' 通配符")

setup_logging()
logger = logging.getLogger(__name__)

openapi_tags = [
    {
        "name": "状态模块",
        "description": "包含 **状态监测** 等API接口，用于监测系统的运行状态。"
    }
]


def _get_app_state(app: FastAPI) -> State:
    """获取应用 state 对象，规避静态检查误报。"""
    return cast(State, getattr(app, "state"))


async def _log_migration_heartbeat(interval_seconds: float) -> None:
    while True:
        await asyncio.sleep(interval_seconds)
        logger.info("数据库迁移进行中, 服务将在迁移完成后继续启动")


async def run_startup_migrations() -> None:
    if not settings.auto_run_db_migrations:
        logger.info("已跳过启动数据库迁移(AUTO_RUN_DB_MIGRATIONS=false)")
        return

    logger.info("开始执行数据库迁移")
    alembic_cfg = Config("alembic.ini")
    alembic_cfg.attributes["configure_logger"] = False

    heartbeat_interval = max(settings.db_migration_log_interval_seconds, 0.5)
    heartbeat_task = asyncio.create_task(_log_migration_heartbeat(heartbeat_interval))

    try:
        await asyncio.to_thread(command.upgrade, alembic_cfg, "head")
        logger.info("数据库迁移完成")
    except Exception:
        logger.exception("数据库迁移失败, 服务启动终止")
        raise
    finally:
        heartbeat_task.cancel()
        with suppress(asyncio.CancelledError):
            await heartbeat_task


@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期上下文管理"""
    logger.info(f"{settings.env} 模式下启动服务")
    logger.info("Swagger文档已%s", "关闭" if not enable_api_docs else "开启")
    # 启动期校验鉴权覆盖，避免新增 API 路由漏挂鉴权依赖后直接上线。
    validate_api_auth_coverage(app)
    await run_startup_migrations()

    # 初始化数据库连接
    await get_redis_client().init()
    await get_postgres().init()
    await get_cos().init()
    checkpointer = get_langgraph_checkpointer()
    await checkpointer.init()
    # 启动阶段显式执行 checkpoint schema 初始化，避免把 DDL 隐藏在 init() 内部。
    # 这里要求应用数据库账号具备相应权限；同样逻辑也保留在独立 bootstrap 脚本中，便于部署前单独执行。
    await checkpointer.ensure_schema()
    app_state = _get_app_state(app)
    app_state.agent_service = get_agent_service_for_lifespan()

    try:
        # lifespan分界点
        yield
    finally:
        try:
            # 等待agent服务关闭
            logger.info("正在关闭Agent服务")
            app_state = _get_app_state(app)
            agent_service = getattr(app_state, "agent_service", None)
            if agent_service is None:
                agent_service = get_agent_service_for_lifespan()
            await asyncio.wait_for(agent_service.shutdown(), timeout=30.0)
            logger.info("Agent服务成功关闭")
        except asyncio.TimeoutError:
            logger.warning("Agent服务关闭超时, 强制关闭, 部分任务将被释放")
        except Exception as e:
            logger.error(f"Agent服务关闭期间出现错误: {str(e)}")
        finally:
            app_state = _get_app_state(app)
            if hasattr(app_state, "agent_service"):
                delattr(app_state, "agent_service")
            clear_agent_service_for_lifespan_cache()

        # 关闭其他应用
        await checkpointer.close()
        await get_redis_client().close()
        await get_postgres().close()
        await get_cos().close()

        logger.info("应用服务关闭成功")


app = FastAPI(
    title="Actus通用智能体",
    description="Actus是一个通用的AI Agent系统,可以完全私有化部署,使用A2A+MCP连接Agent/Tool。",
    lifespan=lifespan,
    openapi_tags=openapi_tags,
    version="0.1.0",
    docs_url="/docs" if enable_api_docs else None,
    redoc_url="/redoc" if enable_api_docs else None,
    openapi_url="/openapi.json" if enable_api_docs else None,
)

# 跨域处理
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 异常处理
register_exception_handlers(app=app)

# 路由集成
app.include_router(router=router, prefix="/api")
