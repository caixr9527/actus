#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/12/8 14:27
@Author : caixiaorong01@outlook.com
@File   : browser.py
"""
from typing import Protocol, Optional

from app.domain.models import ToolResult


class Browser(Protocol):
    """浏览器服务扩展"""

    async def read_current_page_structured(self) -> ToolResult:
        """读取当前页面的结构化摘要。"""
        ...

    async def extract_main_content(self) -> ToolResult:
        """提取当前页面的正文内容。"""
        ...

    async def extract_cards(self) -> ToolResult:
        """提取当前页面的候选卡片。"""
        ...

    async def find_link_by_text(self, text: str) -> ToolResult:
        """按文本查找最匹配的链接。"""
        ...

    async def find_actionable_elements(self) -> ToolResult:
        """提取当前页面的主要可交互元素。"""
        ...

    async def view_page(self) -> ToolResult:
        """查看页面"""
        ...

    async def navigate(self, url: str) -> ToolResult:
        """导航"""
        ...

    async def restart(self, url: str) -> ToolResult:
        """重启浏览器"""
        ...

    async def click(
            self,
            index: Optional[int] = None,
            coordinate_x: Optional[float] = None,
            coordinate_y: Optional[float] = None,
    ) -> ToolResult:
        """点击"""
        ...

    async def input(
            self,
            text: str,
            press_enter: bool,
            index: Optional[int] = None,
            coordinate_x: Optional[float] = None,
            coordinate_y: Optional[float] = None,
    ) -> ToolResult:
        """输入"""
        ...

    async def move_mouse(
            self,
            coordinate_x: float,
            coordinate_y: float,
    ) -> ToolResult:
        """移动鼠标"""
        ...

    async def press_key(
            self,
            key: str,
    ) -> ToolResult:
        """按下按键"""
        ...

    async def select_option(self, index: int, option: int) -> ToolResult:
        """选择选项"""
        ...

    async def scroll_up(self, to_top: Optional[bool] = None) -> ToolResult:
        """向上滚动"""
        ...

    async def scroll_down(self, to_bottom: Optional[bool] = None) -> ToolResult:
        """向下滚动"""
        ...

    async def screenshot(self, full_page: Optional[bool] = None) -> bytes:
        """截图"""
        ...

    async def console_exec(self, javascript: str) -> ToolResult:
        """执行 JavaScript"""
        ...

    async def console_view(self, max_lines: Optional[int] = None) -> ToolResult:
        """查看 JavaScript 控制台"""
        ...
