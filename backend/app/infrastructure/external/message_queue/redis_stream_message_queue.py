#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/11/18 10:22
@Author : caixiaorong01@outlook.com
@File   : redis_stream_message_queue.py
"""
import asyncio
import logging
import uuid
from typing import Any, Tuple, Optional, AsyncGenerator

from app.domain.external import MessageQueue
from app.infrastructure.storage import get_redis_client

logger = logging.getLogger(__name__)


class RedisStreamMessageQueue(MessageQueue):
    """Redis Stream消息队列实现"""

    def __init__(self, stream_name: str) -> None:
        self._stream_name = stream_name
        self._redis = get_redis_client()
        self._lock_expire_seconds = 10

    async def _acquire_lock(self, lock_key: str, timeout_seconds: int = 5) -> Optional[str]:
        # 创建锁对应的值
        lock_value = str(uuid.uuid4())
        end_time = timeout_seconds

        # 使用end_time构建一个循环
        while end_time > 0:
            # 使用redis的set方法，将lock_key和lock_value存储到redis中，并且设置过期时间
            result = await self._redis.client.set(
                lock_key,
                lock_value,
                nx=True,  # 如果值存在则不设置，否则进行设置
                ex=self._lock_expire_seconds,
            )

            # 如果设置成功，则返回锁的值
            if result:
                return lock_value

            # 睡眠指定时间并且将end_time递减
            await asyncio.sleep(0.1)
            end_time -= 0.1

        return None

    async def _release_lock(self, lock_key: str, lock_value: str) -> bool:
        """根据传递的lock_key+lock_value释放分布式锁"""
        # 构建一段redis的脚本用于释放分布式锁
        release_script = """
        if redis.call("GET", KEYS[1]) == ARGV[1] then
            return redis.call("DEL", KEYS[1])
        else
            return 0
        end
        """

        try:
            # 注册脚本
            script = self._redis.client.register_script(release_script)

            # 执行脚本并传递keys+args释放分布式锁
            result = await script(keys=[lock_key], args=[lock_value])

            return result == 1
        except Exception:
            return False

    async def put(self, message: Any) -> str:
        """
        将消息放入队列
        """
        logger.debug(f"添加消息,队列名称: {self._stream_name}, 消息: {message}")
        return await self._redis.client.xadd(self._stream_name, {"data": message})

    async def get(self, start_id: str = None, block_ms: int = None) -> Tuple[str, Any]:
        """
        从队列中取出消息
        """
        logger.debug(f"获取消息,队列名称: {self._stream_name}, 开始ID: {start_id}, 阻塞时间: {block_ms}")
        if start_id is None:
            start_id = '0'
        message = await self._redis.client.xread(
            {self._stream_name: start_id},
            block=block_ms,
            count=1,
        )
        if not message:
            return None, None
        stream_message = message[0][1]
        if not stream_message:
            return None, None
        message_id, message = stream_message[0]
        try:
            return message_id, message.get("data")
        except Exception as e:
            logger.error(f"获取消息失败,队列名称: {self._stream_name}, 错误信息: {e}")
            return None, None

    async def pop(self) -> Tuple[str, Any]:
        """从消息队列中获取第一条消息并删除"""
        # 记录日志
        logger.debug(f"从消息队列中获取第一条消息并删除,队列名称: {self._stream_name}")
        lock_key = f"lock:{self._stream_name}:pop"

        # 构建分布式锁，如果分布式锁创建失败则返回None
        lock_value = await self._acquire_lock(lock_key)
        if not lock_value:
            return None, None

        try:
            # 从redis流中获取第一条消息
            messages = await self._redis.client.xrange(self._stream_name, "-", "+", count=1)
            if not messages:
                return None, None

            # 取出消息id和消息
            message_id, message_data = messages[0]

            # 删除消息队列中的message数据
            await self._redis.client.xdel(self._stream_name, message_id)

            return message_id, message_data.get("data")
        except Exception as e:
            logger.error(f"解析消息队列[{self._stream_name}]出错: {str(e)}")
            return None, None
        finally:
            await self._release_lock(lock_key, lock_value)

    async def clear(self) -> None:
        """清除redis-stream中的所有消息"""
        await self._redis.client.xtrim(self._stream_name, 0)

    async def delete_stream(self) -> bool:
        """删除整个stream key，释放任务生命周期资源"""
        try:
            await self._redis.client.delete(self._stream_name)
            return True
        except Exception:
            return False

    async def is_empty(self) -> bool:
        """检查redis-stream是否为空"""
        return await self.size() == 0

    async def size(self) -> int:
        """获取redis-stream的长度"""
        return await self._redis.client.xlen(self._stream_name)

    async def delete_message(self, message_id: str) -> bool:
        """根据传递的消息id从redis-stream删除数据"""
        try:
            await self._redis.client.xdel(self._stream_name, message_id)
            return True
        except Exception:
            return False

    async def get_range(
            self,
            start_id: str = "-",
            end_id: str = "+",
            count: int = 100,
    ) -> AsyncGenerator[Tuple[str, Any], None]:
        """根据传递的起点、终点id、数量，获取异步迭代器得到消息数据"""
        # 获取所有的消息
        messages = await self._redis.client.xrange(self._stream_name, start_id, end_id, count=count)

        # 如果消息不存在则中断程序
        if not messages:
            return

        # 循环遍历所有的消息列表并取出消息id+消息数据
        for message_id, message_data in messages:
            try:
                yield message_id, message_data.get("data")
            except Exception:
                continue

    async def get_latest_id(self) -> str:
        """获取消息队列中最新的id"""
        # 取出倒序的消息列表，并且设置count=1
        messages = await self._redis.client.xrevrange(self._stream_name, "+", "-", count=1)
        if not messages:
            return "0"

        # 否则取出消息id并返回
        return messages[0][0]
