#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Workspace shell capability。"""

from typing import Optional

from app.domain.external import Sandbox
from app.domain.models import ToolResult
from app.domain.services.tools.base import BaseTool, tool
from ..service import WorkspaceRuntimeService


class WorkspaceShellCapability(BaseTool):
    name: str = "shell"

    def __init__(
            self,
            sandbox: Sandbox,
            workspace_runtime_service: WorkspaceRuntimeService,
    ) -> None:
        super().__init__()
        self._sandbox = sandbox
        self._workspace_runtime_service = workspace_runtime_service

    @staticmethod
    def _extract_output_text(result: ToolResult) -> str:
        data = result.data
        if isinstance(data, dict):
            return str(data.get("output") or "").strip()
        return ""

    async def _get_default_shell_session_id(self) -> str:
        return await self._workspace_runtime_service.ensure_shell_session_id()

    async def _record_shell_state(
            self,
            *,
            function_name: str,
            result: ToolResult,
            cwd: Optional[str] = None,
    ) -> None:
        shell_session_status = ""
        if function_name == "shell_execute":
            shell_session_status = "active"
        elif function_name == "shell_kill_process":
            shell_session_status = "terminated" if result.success else ""
        elif function_name == "shell_wait_process" and result.success:
            shell_session_status = "idle"

        result_data = result.data if isinstance(result.data, dict) else {}
        await self._workspace_runtime_service.record_shell_state(
            cwd=cwd,
            shell_session_status=shell_session_status or None,
            latest_shell_result={
                "function_name": function_name,
                "message": str(result.message or "").strip(),
                "console": self._extract_output_text(result),
                "output": str(result_data.get("output") or "").strip(),
                "console_records": list(result_data.get("console_records") or []),
            },
        )

    @tool(
        name="shell_execute",
        description="在当前工作区默认 Shell 会话中执行命令。可用于运行代码、安装依赖包或文件管理。",
        parameters={
            "exec_dir": {
                "type": "string",
                "description": "执行命令的工作目录（必须使用绝对路径）",
            },
            "command": {
                "type": "string",
                "description": "要执行的 Shell 命令",
            },
        },
        required=["exec_dir", "command"],
    )
    async def shell_execute(
            self,
            exec_dir: str,
            command: str,
    ) -> ToolResult:
        shell_session_id = await self._get_default_shell_session_id()
        result = await self._sandbox.exec_command(shell_session_id, exec_dir, command)
        await self._record_shell_state(
            function_name="shell_execute",
            result=result,
            cwd=exec_dir,
        )
        return result

    @tool(
        name="read_shell_output",
        description="查看当前工作区默认 Shell 会话的内容。用于检查命令执行结果或监控输出。",
        parameters={},
        required=[],
    )
    async def read_shell_output(self) -> ToolResult:
        shell_session_id = await self._get_default_shell_session_id()
        result = await self._sandbox.read_shell_output(shell_session_id)
        await self._record_shell_state(function_name="read_shell_output", result=result)
        return result

    @tool(
        name="shell_wait_process",
        description="等待当前工作区默认 Shell 会话中正在运行的进程返回。在运行耗时较长的命令后使用。",
        parameters={
            "seconds": {
                "type": "integer",
                "description": "可选参数, 等待时长（秒）",
            }
        },
        required=[],
    )
    async def shell_wait_process(self, seconds: Optional[int] = None) -> ToolResult:
        shell_session_id = await self._get_default_shell_session_id()
        result = await self._sandbox.wait_process(shell_session_id, seconds)
        await self._record_shell_state(function_name="shell_wait_process", result=result)
        return result

    @tool(
        name="shell_write_input",
        description="向当前工作区默认 Shell 会话中正在运行的进程写入输入。用于响应交互式命令提示符。",
        parameters={
            "input_text": {
                "type": "string",
                "description": "要写入进程的输入内容",
            },
            "press_enter": {
                "type": "boolean",
                "description": "输入后是否按下回车键",
            }
        },
        required=["input_text", "press_enter"],
    )
    async def shell_write_input(
            self,
            input_text: str,
            press_enter: bool,
    ) -> ToolResult:
        shell_session_id = await self._get_default_shell_session_id()
        result = await self._sandbox.write_shell_input(shell_session_id, input_text, press_enter)
        await self._record_shell_state(function_name="shell_write_input", result=result)
        return result

    @tool(
        name="shell_kill_process",
        description="在当前工作区默认 Shell 会话中终止正在运行的进程。用于停止长时间运行的进程或处理卡死的命令。",
        parameters={},
        required=[],
    )
    async def shell_kill_process(self) -> ToolResult:
        shell_session_id = await self._get_default_shell_session_id()
        result = await self._sandbox.kill_process(shell_session_id)
        await self._record_shell_state(function_name="shell_kill_process", result=result)
        return result
