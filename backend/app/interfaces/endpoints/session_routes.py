#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/2/8 15:38
@Author : caixiaorong01@outlook.com
@File   : session_routes.py
"""
import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, AsyncGenerator

import websockets
from fastapi import APIRouter, Depends
from redis.exceptions import RedisError
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.websockets import WebSocket, WebSocketDisconnect
from websockets import ConnectionClosed

from app.application.errors import NotFoundError
from app.application.errors import error_keys
from app.application.service import SessionService, AgentService
from app.infrastructure.storage import get_redis_client
from app.interfaces.schemas import (
    CreateSessionResponse,
    ListSessionResponse,
    ListSessionItem,
    ChatRequest,
    EventMapper,
    GetSessionResponse,
    GetSessionFilesResponse,
    FileReadResponse,
    FileReadRequest,
    ShellReadResponse,
    ShellReadRequest
)
from app.interfaces.schemas import Response
from app.interfaces.dependencies.services import get_session_service, get_agent_service
from core.realtime import SESSION_LIST_CHANGE_CHANNEL, SESSION_LIST_FALLBACK_REFRESH_SECONDS

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/sessions", tags=["会话模块"])


async def _build_session_list_payload(session_service: SessionService) -> str:
    sessions = await session_service.get_all_sessions()
    session_items = [
        ListSessionItem(
            session_id=session.id,
            title=session.title,
            latest_message=session.latest_message,
            latest_message_at=session.latest_message_at,
            status=session.status,
            unread_message_count=session.unread_message_count,
        )
        for session in sessions
    ]
    return ListSessionResponse(sessions=session_items).model_dump_json()


@router.post(
    path="",
    response_model=Response[CreateSessionResponse],
    summary="创建新任务会话",
    description="创建一个空白的新任务会话",
)
async def create_session(
        session_service: SessionService = Depends(get_session_service),
) -> Response[CreateSessionResponse]:
    """创建一个空白的新任务会话"""
    session = await session_service.create_session()
    return Response.success(
        msg="创建任务会话成功",
        data=CreateSessionResponse(session_id=session.id)
    )


@router.post(
    path="/stream",
    summary="流式获取所有会话基础信息列表",
    description="基于 Redis Pub/Sub 增量推送会话列表，并定期兜底校准",
)
async def stream_sessions(
        session_service: SessionService = Depends(get_session_service),
) -> EventSourceResponse:
    """基于事件驱动流式获取会话列表，减少固定轮询带来的数据库压力"""

    async def event_generator() -> AsyncGenerator[ServerSentEvent, None]:
        previous_payload: str | None = None
        pubsub = None

        # 连接建立后先返回一次完整快照，避免客户端首屏等待。
        initial_payload = await _build_session_list_payload(session_service)
        previous_payload = initial_payload
        yield ServerSentEvent(event="sessions", data=initial_payload)

        try:
            pubsub = get_redis_client().client.pubsub()
            await pubsub.subscribe(SESSION_LIST_CHANGE_CHANNEL)
        except (RuntimeError, RedisError, Exception) as e:
            # Redis 异常时降级到周期兜底，保证能力可用。
            logger.warning(f"订阅会话列表变更通道失败，降级为周期校准: {e}")
            pubsub = None

        try:
            while True:
                if pubsub is not None:
                    # 事件触发增量刷新；超时后执行兜底校准，防止漏通知。
                    await pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=SESSION_LIST_FALLBACK_REFRESH_SECONDS,
                    )
                else:
                    await asyncio.sleep(SESSION_LIST_FALLBACK_REFRESH_SECONDS)

                payload = await _build_session_list_payload(session_service)
                if payload == previous_payload:
                    continue
                previous_payload = payload
                yield ServerSentEvent(event="sessions", data=payload)
        finally:
            if pubsub is not None:
                try:
                    await pubsub.unsubscribe(SESSION_LIST_CHANGE_CHANNEL)
                finally:
                    await pubsub.aclose()

    # 返回EventSourceResponse对象，用于流式传输会话列表数据
    return EventSourceResponse(event_generator())


@router.get(
    path="",
    response_model=Response[ListSessionResponse],
    summary="获取会话列表基础信息",
    description="获取所有任务会话基础信息列表",
)
async def get_all_sessions(
        session_service: SessionService = Depends(get_session_service),
) -> Response[ListSessionResponse]:
    """获取所有任务会话基础信息列表"""
    sessions = await session_service.get_all_sessions()
    session_items = [
        ListSessionItem(
            session_id=session.id,
            title=session.title,
            latest_message=session.latest_message,
            latest_message_at=session.latest_message_at,
            status=session.status,
            unread_message_count=session.unread_message_count,
        )
        for session in sessions
    ]
    return Response.success(
        msg="获取任务会话列表成功",
        data=ListSessionResponse(sessions=session_items)
    )


@router.post(
    path="/{session_id}/clear-unread-message-count",
    response_model=Response[Optional[Dict]],
    summary="清除指定任务会话未读消息数",
    description="清除指定任务会话未读消息数",
)
async def clear_unread_message_count(
        session_id: str,
        session_service: SessionService = Depends(get_session_service),
) -> Response[Optional[Dict]]:
    """根据传递的会话id清空未读消息数"""
    await session_service.clear_unread_message_count(session_id)
    return Response.success(msg="清除未读消息数成功")


@router.post(
    path="/{session_id}/delete",
    response_model=Response[Optional[Dict]],
    summary="删除指定任务会话",
    description="根据传递的会话id删除指定任务会话",
)
async def delete_session(
        session_id: str,
        session_service: SessionService = Depends(get_session_service),
) -> Response[Optional[Dict]]:
    """根据传递的会话id删除指定任务会话"""
    await session_service.delete_session(session_id)
    return Response.success(msg="删除任务会话成功")


@router.post(
    path="/{session_id}/chat",
    summary="向指定任务会话发起聊天请求",
    description="向指定任务会话发起聊天请求"
)
async def chat(
        session_id: str,
        request: ChatRequest,
        session_service: SessionService = Depends(get_session_service),
        agent_service: AgentService = Depends(get_agent_service),
) -> EventSourceResponse:
    """根据传递的会话id+chat请求数据向指定会话发起聊天请求"""
    session = await session_service.get_session(session_id=session_id)
    if not session:
        raise NotFoundError(
            msg="该会话不存在，请核实后重试",
            error_key=error_keys.SESSION_NOT_FOUND,
            error_params={"session_id": session_id},
        )

    async def event_generator() -> AsyncGenerator[ServerSentEvent, None]:
        """定义事件生成器，用于配合EventSourceResponse生成流式响应数据"""
        # 调用Agent服务发起聊天
        async for event in agent_service.chat(
                session_id=session_id,
                message=request.message,
                attachments=request.attachments,
                latest_event_id=request.event_id,
                timestamp=datetime.fromtimestamp(request.timestamp) if request.timestamp else None,
        ):
            # 将Agent事件转换为sse数据(因为普通的event没法通过流式事件传输)
            sse_event = EventMapper.event_to_sse_event(event)
            if sse_event:
                yield ServerSentEvent(
                    event=sse_event.event,
                    data=sse_event.data.model_dump_json(),
                )

    return EventSourceResponse(event_generator())


@router.get(
    path="/{session_id}",
    response_model=Response[GetSessionResponse],
    summary="获取指定会话详情信息",
    description="根据传递的会话id获取该会话的对话详情",
)
async def get_session(
        session_id: str,
        session_service: SessionService = Depends(get_session_service),
) -> Response[GetSessionResponse]:
    """传递指定会话id获取该会话的对话详情"""
    session = await session_service.get_session(session_id=session_id)
    if not session:
        raise NotFoundError(
            msg="该会话不存在，请核实后重试",
            error_key=error_keys.SESSION_NOT_FOUND,
            error_params={"session_id": session_id},
        )
    return Response.success(
        msg="获取会话详情成功",
        data=GetSessionResponse(
            session_id=session.id,
            title=session.title,
            status=session.status,
            events=EventMapper.events_to_sse_events(session.events),
        )
    )


@router.post(
    path="/{session_id}/stop",
    response_model=Response[Optional[Dict]],
    summary="停止指定任务会话",
    description="根据传递的指定会话id停止对应任务会话",
)
async def stop_session(
        session_id: str,
        agent_service: AgentService = Depends(get_agent_service),
) -> Response[Optional[Dict]]:
    """根据传递的指定会话id停止对应任务会话"""
    await agent_service.stop_session(session_id=session_id)
    return Response.success(msg="停止任务会话成功")


@router.get(
    path="/{session_id}/files",
    response_model=Response[GetSessionFilesResponse],
    summary="获取指定任务会话文件列表信息",
    description="获取指定任务会话文件列表信息",
)
async def get_session_files(
        session_id: str,
        session_service: SessionService = Depends(get_session_service),
) -> Response[GetSessionFilesResponse]:
    """获取指定任务会话文件列表信息"""
    files = await session_service.get_session_files(session_id=session_id)
    return Response.success(
        msg="获取会话文件列表成功",
        data=GetSessionFilesResponse(files=files)
    )


@router.post(
    path="/{session_id}/file",
    response_model=Response[FileReadResponse],
    summary="查看会话沙箱中指定文件的内容",
    description="根据传递的会话id+文件路径查看沙箱中文件的内容信息"
)
async def read_file(
        session_id: str,
        request: FileReadRequest,
        session_service: SessionService = Depends(get_session_service),
) -> Response[FileReadResponse]:
    """根据传递的会话id+文件路径查看沙箱中文件的内容信息"""
    result = await session_service.read_file(session_id=session_id, filepath=request.filepath)
    return Response.success(
        msg="获取会话文件内容成功",
        data=result
    )


@router.post(
    path="/{session_id}/shell",
    response_model=Response[ShellReadResponse],
    summary="查看会话的shell内容输出",
    description="传递指定会话id与shell会话标识，查看shell内容输出",
)
async def read_shell_output(
        session_id: str,
        request: ShellReadRequest,
        session_service: SessionService = Depends(get_session_service),
) -> Response[ShellReadResponse]:
    """查看会话的shell内容输出"""
    result = await session_service.read_shell_output(session_id=session_id, shell_session_id=request.session_id)
    return Response.success(
        msg="获取Shell内容输出结果成功",
        data=result,
    )


@router.websocket(
    path="/{session_id}/vnc",
)
async def vnc_websocket(
        websocket: WebSocket,
        session_id: str,
        session_service: SessionService = Depends(get_session_service),
) -> None:
    """VNC Websocket端点，用于建立与沙箱环境的vnc连接，并双向转发数据"""
    # 解析WebSocket子协议，支持binary和base64两种格式
    protocols_str = websocket.headers.get("sec-websocket-protocol", "")
    protocols = [p.strip() for p in protocols_str.split(",")]

    # 根据客户端支持的协议选择合适的子协议
    selected_protocol = None
    if "binary" in protocols:
        selected_protocol = "binary"
    elif "base64" in protocols:
        selected_protocol = "base64"

    # 记录日志并接受WebSocket连接
    logger.info(f"为会话[{session_id}]开启WebSocket连接")
    await websocket.accept(subprotocol=selected_protocol)

    try:
        # 获取沙箱环境的VNC URL并建立连接
        sandbox_vnc_url = await session_service.get_vnc_url(session_id=session_id)
        logger.info(f"连接WebSocket VNC： {sandbox_vnc_url}")

        # 建立到沙箱VNC服务器的WebSocket连接
        async with websockets.connect(sandbox_vnc_url) as sandbox_ws:

            # 定义从客户端向沙箱转发数据的协程函数
            async def forward_to_sandbox():
                try:
                    while True:
                        # 接收客户端发送的二进制数据并转发到沙箱
                        data = await websocket.receive_bytes()
                        await sandbox_ws.send(data)
                except WebSocketDisconnect:
                    # 客户端断开连接时记录日志
                    logger.info(f"Web->VNC连接终端")
                except Exception as forward_e:
                    # 处理转发过程中的异常
                    logger.error(f"forward_to_sandbox出错: {str(forward_e)}")

            # 定义从沙箱向客户端转发数据的协程函数
            async def forward_from_sandbox():
                try:
                    while True:
                        # 接收沙箱发送的数据并转发给客户端
                        data = await sandbox_ws.recv()
                        await websocket.send_bytes(data)
                except ConnectionClosed:
                    # 沙箱连接关闭时记录日志
                    logger.info("VNC->Web连接关闭")
                except Exception as forward_e:
                    # 处理转发过程中的异常
                    logger.error(f"forward_from_sandbox出错: {str(forward_e)}")

            # 创建两个转发任务并等待其中任一任务完成
            forward_task1 = asyncio.create_task(forward_to_sandbox())
            forward_task2 = asyncio.create_task(forward_from_sandbox())

            # 等待任一转发任务完成（通常意味着连接中断）
            done, pending = await asyncio.wait(
                [forward_task1, forward_task2],
                return_when=asyncio.FIRST_COMPLETED,
            )
            logger.info("WebSocket连接已关闭")

            # 取消仍在运行的任务
            for task in pending:
                task.cancel()

    except ConnectionError as connection_e:
        # 处理连接沙箱环境失败的情况
        logger.error(f"连接沙箱环境失败: {str(connection_e)}")
        await websocket.close(code=1011, reason=f"连接沙箱环境失败: {str(connection_e)}")
    except Exception as e:
        # 处理其他WebSocket异常
        logger.error(f"WebSocket异常: {str(e)}")
        await websocket.close(code=1011, reason=f"WebSocket异常: {str(e)}")
