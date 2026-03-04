#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/12/12 09:55
@Author : caixiaorong01@outlook.com
@File   : sandbox.py
"""
from typing import Protocol, Optional, BinaryIO, Self

from app.domain.external import Browser
from app.domain.models import ToolResult


class Sandbox(Protocol):

    async def exec_command(self, session_id: str, exec_dir: str, command: str) -> ToolResult:
        """
        执行命令
        :param session_id: 会话ID
        :param exec_dir: 执行目录
        :param command: 命令
        :return:
        """
        ...

    async def read_shell_output(self, session_id: str, console: bool = False) -> ToolResult:
        """
        查看Shell
        :param session_id: 会话ID
        :param console: 是否返回控制台信息
        :return:
        """
        ...

    async def wait_process(self, session_id: str, seconds: Optional[int] = None) -> ToolResult:
        """
        等待进程结束
        :param session_id: 会话ID
        :param seconds: 等待秒数
        :return:
        """
        ...

    async def write_shell_input(
            self,
            session_id: str,
            input_text: str,
            press_enter: bool = True,
    ) -> ToolResult:
        """
        向进程写入数据
        :param session_id: 会话ID
        :param input_text: 要写入的数据
        :param press_enter: 是否在写入完成后按下回车键
        :return:
        """
        ...

    async def kill_process(self, session_id: str) -> ToolResult:
        """
        杀死进程
        :param session_id: 会话ID
        :return:
        """
        ...

    async def write_file(
            self,
            file_path: str,
            content: str,
            append: bool = False,
            leading_newline: bool = False,
            trailing_newline: bool = False,
            sudo: bool = False,
    ) -> ToolResult:
        """
        写入文件
        :param file_path: 文件路径
        :param content: 文件内容
        :param append: 是否追加
        :param leading_newline: 是否在开头添加换行符
        :param trailing_newline: 是否在结尾添加换行符
        :param sudo: 是否使用sudo权限
        :return:
        """
        ...

    async def read_file(
            self,
            file_path: str,
            start_line: Optional[int] = None,
            end_line: Optional[int] = None,
            sudo: bool = False,
            max_length: int = 10000
    ) -> ToolResult:
        """
        读取文件
        :param max_length: 最大长度
        :param end_line: 结束行
        :param start_line: 开始行
        :param file_path: 文件路径
        :param sudo: 是否使用sudo权限
        :return:
        """
        ...

    async def check_file_exists(self, file_path: str) -> ToolResult:
        """
        判断文件是否存在
        :param file_path: 文件路径
        :return:
        """
        ...

    async def delete_file(self, file_path: str) -> ToolResult:
        """
        删除文件
        :param file_path: 文件路径
        :return:
        """
        ...

    async def list_files(self, dir_path: str) -> ToolResult:
        """
        列出目录下的文件
        :param dir_path: 目录路径
        :return:
        """
        ...

    async def replace_in_file(
            self,
            file_path: str,
            old_text: str,
            new_text: str,
            sudo: bool = False,
    ) -> ToolResult:
        """
        替换文件中的文本
        :param file_path: 文件路径
        :param old_text: 要替换的文本
        :param new_text: 替换后的文本
        :param sudo: 是否使用sudo权限
        :return:
        """
        ...

    async def search_in_file(self, file_path: str, regex: str, sudo: bool = False) -> ToolResult:
        """
        搜索文件
        :param file_path: 文件路径
        :param regex: 正则表达式
        :param sudo: 是否使用sudo权限
        :return:
        """
        ...

    async def find_files(self, dir_path: str, glob_pattern: str) -> ToolResult:
        """
        查找文件
        :param dir_path: 目录路径
        :param glob_pattern: 正则表达式
        :return:
        """
        ...

    async def upload_file(
            self,
            file_data: BinaryIO,
            file_path: str,
            filename: str = None,
    ) -> ToolResult:
        """
        上传文件
        :param file_data: 文件数据
        :param file_path: 文件路径
        :param filename: 文件名
        :return:
        """
        ...

    async def download_file(self, file_path: str) -> BinaryIO:
        """
        下载文件
        :param file_path: 文件路径
        :return:
        """
        ...

    async def ensure_sandbox(self) -> None:
        """
        确保沙盒存在
        :return:
        """
        ...

    async def destroy(self) -> bool:
        """
        销毁沙盒
        :return:
        """
        ...

    async def get_browser(self) -> Browser:
        """
        获取浏览器
        :return:
        """
        ...

    @property
    def id(self) -> str:
        """
        获取沙盒ID
        :return:
        """
        ...

    @property
    def cdp_url(self) -> str:
        """
        获取CDP URL
        :return:
        """
        ...

    @property
    def vnc_url(self) -> str:
        """
        获取VNC URL
        :return:
        """
        ...

    @classmethod
    async def create(cls) -> Self:
        """
        创建沙盒
        :return:
        """
        ...

    @classmethod
    async def get(cls, id: str) -> Optional[Self]:
        """
        获取沙盒
        :param id: 沙盒ID
        :return:
        """
        ...
