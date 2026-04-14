#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Workspace file capability。"""

from typing import Optional

from app.domain.external import Sandbox
from app.domain.models import ToolResult
from app.domain.services.tools.base import BaseTool, tool
from ..service import WorkspaceRuntimeService


class WorkspaceFileCapability(BaseTool):
    """工作区文件能力。"""

    name: str = "file"

    def __init__(
            self,
            sandbox: Sandbox,
            workspace_runtime_service: WorkspaceRuntimeService,
    ) -> None:
        super().__init__()
        self._sandbox = sandbox
        self._workspace_runtime_service = workspace_runtime_service

    @staticmethod
    def _build_file_tree_summary(*, dir_path: str, files: list[object]) -> str:
        normalized_dir = str(dir_path or "").strip()
        file_count = len([item for item in files if str(item).strip()])
        if normalized_dir:
            return f"目录 {normalized_dir} 共 {file_count} 项"
        return f"共 {file_count} 项"

    @tool(
        name="read_file",
        description="读取文件内容。用于检查文件内容、分析日志或读取配置文件。",
        parameters={
            "filepath": {
                "type": "string",
                "description": "要读取文件的绝对路径"
            },
            "start_line": {
                "type": "integer",
                "description": "(可选)读取的起始行, 索引从 0 开始",
            },
            "end_line": {
                "type": "integer",
                "description": "(可选)结束行号, 不包含该行",
            },
            "sudo": {
                "type": "boolean",
                "description": "(可选)是否使用 sudo 权限",
            },
            "max_length": {
                "type": "integer",
                "description": "(可选)读取文件内容的最大长度, 默认为10000"
            }
        },
        required=["filepath"],
    )
    async def read_file(
            self,
            filepath: str,
            start_line: Optional[int] = None,
            end_line: Optional[int] = None,
            sudo: Optional[bool] = False,
            max_length: int = 10000,
    ) -> ToolResult:
        return await self._sandbox.read_file(
            file_path=filepath,
            start_line=start_line,
            end_line=end_line,
            sudo=sudo,
            max_length=max_length,
        )

    @tool(
        name="write_file",
        description="对文件进行覆盖或追加写入。用于创建新文件、追加内容或修改现有文件。",
        parameters={
            "filepath": {
                "type": "string",
                "description": "要写入文件的绝对路径"
            },
            "content": {
                "type": "string",
                "description": "要写入的文本内容"
            },
            "append": {
                "type": "boolean",
                "description": "(可选)是否使用追加模式"
            },
            "leading_newline": {
                "type": "boolean",
                "description": "(可选)是否添加前置换行符, 在内容开头"
            },
            "trailing_newline": {
                "type": "boolean",
                "description": "(可选)是否添加后置换行符, 在内容结尾"
            },
            "sudo": {
                "type": "boolean",
                "description": "(可选)是否使用 sudo 权限"
            }
        },
        required=["filepath", "content"]
    )
    async def write_file(
            self,
            filepath: str,
            content: str,
            append: Optional[bool] = False,
            leading_newline: Optional[bool] = False,
            trailing_newline: Optional[bool] = False,
            sudo: Optional[bool] = False
    ) -> ToolResult:
        result = await self._sandbox.write_file(
            file_path=filepath,
            content=content,
            append=append,
            leading_newline=leading_newline,
            trailing_newline=trailing_newline,
            sudo=sudo,
        )
        if result.success:
            await self._workspace_runtime_service.upsert_artifact(
                path=filepath,
                artifact_type="file",
                summary=f"通过 write_file 更新文件: {filepath}",
                source_capability="write_file",
                metadata={
                    "append": bool(append),
                    "leading_newline": bool(leading_newline),
                    "trailing_newline": bool(trailing_newline),
                    "sudo": bool(sudo),
                },
            )
        return result

    @tool(
        name="replace_in_file",
        description="在文件中替换指定的字符串。用于更新文件中的特定内容或修复代码中的错误。",
        parameters={
            "filepath": {
                "type": "string",
                "description": "要执行替换操作的文件的绝对路径"
            },
            "old_str": {
                "type": "string",
                "description": "要被替换的原始字符串"
            },
            "new_str": {
                "type": "string",
                "description": "用于替换的新字符串"
            },
            "sudo": {
                "type": "boolean",
                "description": "(可选)是否使用 sudo 权限"
            }
        },
        required=["filepath", "old_str", "new_str"]
    )
    async def replace_in_file(
            self,
            filepath: str,
            old_str: str,
            new_str: str,
            sudo: Optional[bool] = False
    ) -> ToolResult:
        result = await self._sandbox.replace_in_file(
            file_path=filepath,
            old_text=old_str,
            new_text=new_str,
            sudo=sudo,
        )
        if result.success:
            await self._workspace_runtime_service.upsert_artifact(
                path=filepath,
                artifact_type="file",
                summary=f"通过 replace_in_file 更新文件: {filepath}",
                source_capability="replace_in_file",
                metadata={
                    "sudo": bool(sudo),
                },
            )
        return result

    @tool(
        name="search_in_file",
        description="在文件内容中搜索匹配的文本。用于查找文件中的特定内容或模式。",
        parameters={
            "filepath": {
                "type": "string",
                "description": "要进行搜索的文件的绝对路径"
            },
            "regex": {
                "type": "string",
                "description": "用于匹配的正则表达式模式"
            },
            "sudo": {
                "type": "boolean",
                "description": "(可选)是否使用 sudo 权限"
            }
        },
        required=["filepath", "regex"]
    )
    async def search_in_file(
            self,
            filepath: str,
            regex: str,
            sudo: Optional[bool] = False
    ) -> ToolResult:
        return await self._sandbox.search_in_file(
            file_path=filepath,
            regex=regex,
            sudo=sudo,
        )

    @tool(
        name="find_files",
        description="在指定目录中根据名称模式查找文件。用于定位具有特定命名模式的文件。",
        parameters={
            "dir_path": {
                "type": "string",
                "description": "要搜索的目录的绝对路径"
            },
            "glob_pattern": {
                "type": "string",
                "description": "使用 glob 语法通配符的文件名模式"
            }
        },
        required=["dir_path", "glob_pattern"]
    )
    async def find_files(
            self,
            dir_path: str,
            glob_pattern: str
    ) -> ToolResult:
        result = await self._sandbox.find_files(
            dir_path=dir_path,
            glob_pattern=glob_pattern,
        )
        if result.success:
            data = result.data if isinstance(result.data, dict) else {}
            await self._workspace_runtime_service.record_file_tree_summary(
                summary_text=self._build_file_tree_summary(
                    dir_path=str(data.get("dir_path") or dir_path or "").strip(),
                    files=list(data.get("files") or []),
                )
            )
        return result

    @tool(
        name="list_files",
        description="列出指定目录下的文件列表信息",
        parameters={
            "dir_path": {
                "type": "string",
                "description": "要列出文件列表的目录的绝对路径"
            },
        },
        required=["dir_path"]
    )
    async def list_files(self, dir_path: str) -> ToolResult:
        result = await self._sandbox.list_files(dir_path)
        if result.success:
            data = result.data if isinstance(result.data, dict) else {}
            await self._workspace_runtime_service.record_file_tree_summary(
                summary_text=self._build_file_tree_summary(
                    dir_path=str(data.get("dir_path") or dir_path or "").strip(),
                    files=list(data.get("files") or []),
                )
            )
        return result
