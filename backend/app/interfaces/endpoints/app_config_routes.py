#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/17 10:13
@Author : caixiaorong01@outlook.com
@File   : app_config_routes.py
"""
import logging
from typing import Dict, Optional

from fastapi import APIRouter, Depends, Body

from app.application.service import AppConfigService, ModelConfigService
from app.domain.models import AgentConfig, MCPConfig
from app.interfaces.dependencies.services import get_app_config_service, get_model_config_service
from app.interfaces.schemas import (
    Response,
    ListMCPServerResponse,
    ListA2AServerResponse,
    ListModelResponse,
    ListModelItem,
    PublicModelConfig,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/app-config", tags=["设置模块"])


@router.get(
    path="/models",
    response_model=Response[ListModelResponse],
    summary="获取可选模型列表",
    description="返回当前启用的模型列表与默认模型 ID",
)
async def get_models(
        model_config_service: ModelConfigService = Depends(get_model_config_service)
) -> Response[ListModelResponse]:
    """获取供前端展示的模型列表"""
    default_model_id, models = await model_config_service.get_public_models()
    return Response.success(
        data=ListModelResponse(
            default_model_id=default_model_id,
            models=[
                ListModelItem(
                    id=model.id,
                    display_name=model.display_name,
                    provider=model.provider,
                    enabled=model.enabled,
                    sort_order=model.sort_order,
                    config=PublicModelConfig.model_validate(model.config),
                )
                for model in models
            ],
        )
    )
@router.get(
    path="/agent",
    response_model=Response[AgentConfig],
    summary="获取Agent配置信息",
    description="包含最大迭代次数、最大重试次数、最大搜索结果数"
)
async def get_agent_config(
        app_config_service: AppConfigService = Depends(get_app_config_service)
) -> Response[AgentConfig]:
    """
    获取Agent配置信息
    """
    agent_config = await app_config_service.get_agent_config()
    return Response.success(data=agent_config.model_dump())


@router.post(
    path="/agent",
    response_model=Response[AgentConfig],
    summary="更新Agent配置信息",
    description="更新Agent配置信息"
)
async def update_agent_config(
        new_agent_config: AgentConfig,
        app_config_service: AppConfigService = Depends(get_app_config_service)
) -> Response[AgentConfig]:
    """
    更新LLM配置信息
    """
    updated_agent_config = await app_config_service.update_agent_config(new_agent_config)
    return Response.success(
        msg="更新Agent配置信息成功",
        data=updated_agent_config.model_dump()
    )


@router.get(
    path="/mcp-servers",
    response_model=Response[ListMCPServerResponse],
    summary="获取MCP服务器配置信息",
    description="包含MCP服务器配置信息"
)
async def get_mcp_servers(
        app_config_service: AppConfigService = Depends(get_app_config_service)
) -> Response[ListMCPServerResponse]:
    """
    获取MCP服务器配置信息
    """
    mcp_servers = await app_config_service.get_mcp_servers()
    return Response.success(
        msg="获取MCP服务器配置信息成功",
        data=ListMCPServerResponse(mcp_servers=mcp_servers)
    )


@router.post(
    path="/mcp-servers",
    response_model=Response[Optional[Dict]],
    summary="新增MCP服务器配置信息",
    description="新增MCP服务器配置信息"
)
async def create_mcp_servers(
        mcp_config: MCPConfig,
        app_config_service: AppConfigService = Depends(get_app_config_service)
) -> Response[Optional[Dict]]:
    """
    新增MCP服务器配置信息
    """
    await app_config_service.update_and_create_mcp_servers(mcp_config)
    return Response.success(msg="新增MCP服务器配置信息成功")


@router.post(
    path="/mcp-servers/{server_name}/delete",
    response_model=Response[Optional[Dict]],
    summary="删除MCP服务器配置信息",
    description="删除MCP服务器配置信息"
)
async def delete_mcp_servers(
        server_name: str,
        app_config_service: AppConfigService = Depends(get_app_config_service)
) -> Response[Optional[Dict]]:
    """
    删除MCP服务器配置信息
    """
    await app_config_service.delete_mcp_server(server_name)
    return Response.success(msg="删除MCP服务器配置信息成功")


@router.post(
    path="/mcp-servers/{server_name}/enabled",
    response_model=Response[Optional[Dict]],
    summary="启用MCP服务器配置",
    description="启用MCP服务器配置"
)
async def set_mcp_server_enabled(
        server_name: str,
        enabled: bool = Body(..., embed=True),
        app_config_service: AppConfigService = Depends(get_app_config_service)
) -> Response[Optional[Dict]]:
    """
    启用MCP服务器配置
    """
    await app_config_service.set_mcp_server_enabled(server_name, enabled)
    return Response.success(msg="更新MCP服务器配置启用状态成功")


@router.get(
    path="/a2a-servers",
    response_model=Response[ListA2AServerResponse],
    summary="获取A2A服务器配置信息",
    description="获取A2A服务器配置信息"
)
async def get_a2a_servers(
        app_config_service: AppConfigService = Depends(get_app_config_service)
) -> Response[ListA2AServerResponse]:
    """
    获取A2A服务器配置信息
    """
    a2a_servers = await app_config_service.get_a2a_servers()
    return Response.success(
        msg="获取A2A服务器配置信息成功",
        data=ListA2AServerResponse(a2a_servers=a2a_servers)
    )


@router.post(
    path="/a2a-servers",
    response_model=Response[Optional[Dict]],
    summary="新增A2A服务器配置信息",
    description="新增A2A服务器配置信息"
)
async def create_a2a_servers(
        base_url: str = Body(..., embed=True),
        app_config_service: AppConfigService = Depends(get_app_config_service)
) -> Response[Optional[Dict]]:
    """
    新增A2A服务器配置信息
    """
    await app_config_service.create_a2a_server(base_url=base_url)
    return Response.success(msg="新增A2A服务器配置信息成功")


@router.post(
    path="/a2a-servers/{a2a_id}/delete",
    response_model=Response[Optional[Dict]],
    summary="删除A2A服务器配置信息",
    description="删除A2A服务器配置信息"
)
async def delete_a2a_servers(
        a2a_id: str,
        app_config_service: AppConfigService = Depends(get_app_config_service)
) -> Response[Optional[Dict]]:
    """
    删除A2A服务器配置信息
    """
    await app_config_service.delete_a2a_server(a2a_id=a2a_id)
    return Response.success(msg="删除A2A服务器配置信息成功")


@router.post(
    path="/a2a-servers/{a2a_id}/enabled",
    response_model=Response[Optional[Dict]],
    summary="启用A2A服务器配置",
    description="启用A2A服务器配置"
)
async def set_a2a_server_enabled(
        a2a_id: str,
        enabled: bool = Body(..., embed=True),
        app_config_service: AppConfigService = Depends(get_app_config_service)
) -> Response[Optional[Dict]]:
    """
    启用A2A服务器配置
    """
    await app_config_service.set_a2a_server_enabled(a2a_id=a2a_id, enabled=enabled)
    return Response.success(msg="更新A2A服务器配置启用状态成功")
