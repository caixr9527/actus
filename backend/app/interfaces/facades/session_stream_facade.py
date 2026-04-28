#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/24 20:40
@Author : caixiaorong01@outlook.com
@File   : session_stream_facade.py
"""
import asyncio
import logging
from datetime import datetime
from typing import AsyncGenerator

from redis.exceptions import RedisError
from sse_starlette import ServerSentEvent

from app.application.errors import NotFoundError, error_keys
from app.application.service import AgentService, RuntimeObservationService, SessionService
from app.application.service.runtime_observation_service import RuntimeObservationContextResult
from app.domain.models import SessionStatus
from app.domain.models import WorkflowRunEventRecord
from app.infrastructure.storage import get_redis_client
from app.interfaces.schemas import (
    ChatRequest,
    EventMapper,
    ListSessionItem,
    ListSessionResponse,
)
from core.realtime import SESSION_LIST_CHANGE_CHANNEL, SESSION_LIST_FALLBACK_REFRESH_SECONDS

logger = logging.getLogger(__name__)


class SessionStreamFacade:
    """会话流式 facade：收敛 sessions/chat 的 SSE 编排，路由层仅保留入参与响应封装。"""

    async def _build_session_list_payload(self, session_service: SessionService, user_id: str) -> str:
        sessions = await session_service.get_all_sessions(user_id=user_id)
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

    async def stream_sessions(
            self,
            *,
            user_id: str,
            session_service: SessionService,
    ) -> AsyncGenerator[ServerSentEvent, None]:
        """基于 Redis Pub/Sub + fallback 校准推送会话列表。"""
        previous_payload: str | None = None
        pubsub = None

        initial_payload = await self._build_session_list_payload(session_service=session_service, user_id=user_id)
        previous_payload = initial_payload
        yield ServerSentEvent(event="sessions", data=initial_payload)

        try:
            pubsub = get_redis_client().client.pubsub()
            await pubsub.subscribe(SESSION_LIST_CHANGE_CHANNEL)
        except (RuntimeError, RedisError, Exception) as e:
            logger.warning(f"订阅会话列表变更通道失败，降级为周期校准: {e}")
            pubsub = None

        try:
            while True:
                if pubsub is not None:
                    await pubsub.get_message(
                        ignore_subscribe_messages=True,
                        timeout=SESSION_LIST_FALLBACK_REFRESH_SECONDS,
                    )
                else:
                    await asyncio.sleep(SESSION_LIST_FALLBACK_REFRESH_SECONDS)

                payload = await self._build_session_list_payload(session_service=session_service, user_id=user_id)
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

    async def stream_chat(
            self,
            *,
            user_id: str,
            session_id: str,
            request: ChatRequest,
            session_service: SessionService,
            agent_service: AgentService,
            runtime_observation_service: RuntimeObservationService,
    ) -> AsyncGenerator[ServerSentEvent, None]:
        """收敛 chat SSE 映射编排（会话校验 + runtime context + EventMapper）。"""
        # Context: FastAPI 只有在构造 EventSourceResponse 前抛错，才能稳定返回 HTTP 404。
        # Decision: stream_chat 先执行会话存在性校验，再返回真正的事件迭代器。
        # Trade-off: 比“纯惰性生成器”多一次前置查询，但可保持既有错误语义不变。
        # Removal Plan: N/A（该语义是 SSE 错误处理的长期约束）。
        session = await session_service.get_session(user_id=user_id, session_id=session_id)
        if not session:
            raise NotFoundError(
                msg="该会话不存在，请核实后重试",
                error_key=error_keys.SESSION_NOT_FOUND,
                error_params={"session_id": session_id},
            )
        is_empty_stream_request = (
                request.message is None
                and request.resume is None
                and request.command is None
        )

        async def _event_generator() -> AsyncGenerator[ServerSentEvent, None]:
            observation_context = None
            live_attach_after_event_id = request.event_id
            if is_empty_stream_request:
                observation_context = await runtime_observation_service.build_event_context(
                    user_id=user_id,
                    session_id=session_id,
                    reconcile_reason="before_chat",
                )
                replay = await runtime_observation_service.list_persistent_events_after_cursor(
                    user_id=user_id,
                    session_id=session_id,
                    cursor_event_id=request.event_id,
                )
                live_attach_after_event_id = replay.live_attach_after_event_id
                replay_context = observation_context.model_copy(update={"current_step_id": None})
                replayed_any = False
                async for sse_event, replay_context in self._iter_observable_records(
                        runtime_observation_service=runtime_observation_service,
                        session_id=session_id,
                        records=replay.records,
                        observation_context=replay_context,
                ):
                    replayed_any = True
                    yield sse_event
                if replayed_any:
                    observation_context = replay_context
                if observation_context.status not in {SessionStatus.RUNNING, SessionStatus.WAITING}:
                    return

            async for event in agent_service.chat(
                    session_id=session_id,
                    user_id=user_id,
                    message=request.message,
                    attachments=request.attachments,
                    resume=request.resume.value if request.resume is not None else None,
                    command=request.command.model_dump(mode="json") if request.command is not None else None,
                    latest_event_id=live_attach_after_event_id,
                    timestamp=datetime.fromtimestamp(request.timestamp) if request.timestamp else None,
            ):
                if observation_context is None:
                    observation_context = await runtime_observation_service.build_event_context(
                        user_id=user_id,
                        session_id=session_id,
                    )
                envelope = await runtime_observation_service.build_observable_event(
                    session_id=session_id,
                    event=event,
                    run_id=observation_context.run_id,
                    source_event_id=event.id,
                    cursor_event_id=event.id,
                    source="sse",
                    context=observation_context,
                )
                observation_context = runtime_observation_service.advance_event_context(
                    observation_context,
                    event,
                )
                sse_event = EventMapper.observable_event_to_sse_event(envelope)
                if sse_event:
                    yield ServerSentEvent(
                        event=sse_event.event,
                        data=sse_event.data.model_dump_json(),
                    )

        return _event_generator()

    async def _iter_observable_records(
            self,
            *,
            runtime_observation_service: RuntimeObservationService,
            session_id: str,
            records: list[WorkflowRunEventRecord],
            observation_context: RuntimeObservationContextResult,
    ) -> AsyncGenerator[tuple[ServerSentEvent, RuntimeObservationContextResult], None]:
        """将 DB persistent records 映射为 SSE；调用方负责维护连接级 context。"""
        for record in records:
            envelope = await runtime_observation_service.build_observable_event(
                session_id=session_id,
                event=record.event_payload,
                run_id=record.run_id,
                source_event_id=record.event_id,
                cursor_event_id=record.event_id,
                source="sse",
                context=observation_context,
            )
            next_context = runtime_observation_service.advance_event_context(
                observation_context,
                record.event_payload,
            )
            sse_event = EventMapper.observable_event_to_sse_event(envelope)
            if sse_event:
                yield (
                    ServerSentEvent(
                        event=sse_event.event,
                        data=sse_event.data.model_dump_json(),
                    ),
                    next_context,
                )
            observation_context = next_context
