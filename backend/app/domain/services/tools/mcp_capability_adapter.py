#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2026/3/23
@Author : caixiaorong01@outlook.com
@File   : mcp_capability_adapter.py
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from app.domain.models import MCPConfig, ToolResult
from app.domain.services.tools.base import BaseTool

logger = logging.getLogger(__name__)


class MCPCapabilityAdapter(BaseTool):
    """MCP capability provider 适配器。

    设计目标：
    1. 将 MCP 作为标准 capability 注入工具集合；
    2. 统一调用超时、审计日志与错误收敛语义；
    3. 保持原有 MCP tool schema 与调用协议兼容。
    """

    name: str = "mcp"
    DEFAULT_INVOKE_TIMEOUT_SECONDS: float = 60.0

    def __init__(
            self,
            mcp_tool: Optional[BaseTool],
            mcp_config: Optional[MCPConfig] = None,
            session_id: Optional[str] = None,
            user_id: Optional[str] = None,
            invoke_timeout_seconds: float = DEFAULT_INVOKE_TIMEOUT_SECONDS,
    ) -> None:
        super().__init__()
        self._mcp_tool = mcp_tool
        self._mcp_config = mcp_config
        self._session_id = session_id
        self._user_id = user_id
        # 允许在测试和灰度场景使用短超时，最低限制到毫秒级，避免出现0或负数。
        self._invoke_timeout_seconds = max(float(invoke_timeout_seconds), 0.001)

    def _is_server_enabled(self, tool_name: str) -> bool:
        if self._mcp_config is None:
            return True

        # 工具名规范：mcp_{server_name}_{tool_name}
        # 这里按 server 配置做显式启停控制，避免调用被禁用的 MCP 端点。
        for server_name, server_config in self._mcp_config.mcpServers.items():
            expected_prefix = server_name if server_name.startswith("mcp_") else f"mcp_{server_name}"
            if tool_name.startswith(f"{expected_prefix}_"):
                return bool(server_config.enabled)
        return False

    def get_tools(self) -> List[Dict[str, Any]]:
        if self._mcp_tool is None:
            return []
        return self._mcp_tool.get_tools()

    @staticmethod
    def _normalize_tool_name_key(tool_name: str) -> str:
        return tool_name.replace("-", "_")

    def _resolve_tool_name(self, tool_name: str) -> Optional[str]:
        """解析工具名，兼容 `-/_` 混用导致的命名漂移。"""
        if self._mcp_tool is None:
            return None

        if self._mcp_tool.has_tool(tool_name):
            return tool_name

        normalized_target = self._normalize_tool_name_key(tool_name)
        candidates: List[str] = []
        for tool in self.get_tools():
            name = (
                tool.get("function", {}).get("name")
                if isinstance(tool, dict)
                else None
            )
            if not isinstance(name, str) or not name:
                continue
            if self._normalize_tool_name_key(name) == normalized_target:
                candidates.append(name)

        if len(candidates) == 1:
            return candidates[0]
        return None

    def has_tool(self, tool_name: str) -> bool:
        if self._mcp_tool is None:
            return False
        resolved_tool_name = self._resolve_tool_name(tool_name)
        if not resolved_tool_name:
            return False
        if not self._is_server_enabled(resolved_tool_name):
            return False
        return True

    async def invoke(self, tool_name: str, **kwargs) -> ToolResult:
        if self._mcp_tool is None:
            return ToolResult(success=False, message="MCP能力未初始化")

        resolved_tool_name = self._resolve_tool_name(tool_name)
        if not resolved_tool_name:
            return ToolResult(success=False, message=f"MCP工具不可用或未启用: {tool_name}")
        if not self._is_server_enabled(resolved_tool_name):
            return ToolResult(success=False, message=f"MCP工具不可用或未启用: {tool_name}")

        logger.info(
            "MCP capability调用开始: session_id=%s user_id=%s tool=%s",
            self._session_id,
            self._user_id,
            tool_name,
        )
        try:
            # 统一调用超时，避免远端MCP长时间阻塞主任务流程。
            result = await asyncio.wait_for(
                self._mcp_tool.invoke(resolved_tool_name, **kwargs),
                timeout=self._invoke_timeout_seconds,
            )
        except TimeoutError:
            logger.warning(
                "MCP capability调用超时: session_id=%s user_id=%s tool=%s timeout=%.1fs",
                self._session_id,
                self._user_id,
                tool_name,
                self._invoke_timeout_seconds,
            )
            return ToolResult(
                success=False,
                message=f"MCP工具调用超时({self._invoke_timeout_seconds:.1f}s): {tool_name}",
            )
        except Exception as e:
            logger.exception(
                "MCP capability调用异常: session_id=%s user_id=%s tool=%s err=%s",
                self._session_id,
                self._user_id,
                tool_name,
                e,
            )
            return ToolResult(
                success=False,
                message=f"MCP工具调用失败: {tool_name}",
            )

        if result.success:
            logger.info(
                "MCP capability调用成功: session_id=%s user_id=%s tool=%s",
                self._session_id,
                self._user_id,
                tool_name,
            )
            return result

        # 统一错误收敛口径：下游可稳定消费 message 字段，不依赖 provider 私有报错格式。
        normalized_message = str(result.message or f"MCP工具调用失败: {tool_name}")
        logger.warning(
            "MCP capability调用失败: session_id=%s user_id=%s tool=%s msg=%s",
            self._session_id,
            self._user_id,
            tool_name,
            normalized_message,
        )
        return ToolResult(
            success=False,
            message=normalized_message,
            data=result.data,
        )
