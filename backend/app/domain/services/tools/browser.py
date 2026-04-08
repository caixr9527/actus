#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/12/8 14:50
@Author : caixiaorong01@outlook.com
@File   : browser.py
"""
from typing import Optional

from app.domain.external import Browser
from app.domain.models import ToolResult
from .base import BaseTool, tool


class BrowserTool(BaseTool):
    name: str = "browser"

    def __init__(self, browser: Browser) -> None:
        super().__init__()
        self.browser = browser

    @tool(
        name="browser_read_current_page_structured",
        description="读取当前页面的结构化摘要。用于判断页面类型、主标题、候选卡片、可交互元素和是否需要继续滚动，是浏览器任务的默认入口。",
        parameters={},
        required=[],
    )
    async def browser_read_current_page_structured(self) -> ToolResult:
        """读取当前页面的结构化摘要。"""
        return await self.browser.read_current_page_structured()

    @tool(
        name="browser_extract_main_content",
        description="提取当前页面的正文内容。适用于文章、博客、课程详情、文档页等阅读型页面，应优先于反复滚动查看。",
        parameters={},
        required=[],
    )
    async def browser_extract_main_content(self) -> ToolResult:
        """提取当前页面的正文内容。"""
        return await self.browser.extract_main_content()

    @tool(
        name="browser_extract_cards",
        description="提取当前页面的候选卡片。适用于搜索结果页、列表页、导航页，应先抽取候选项再决定后续点击目标。",
        parameters={},
        required=[],
    )
    async def browser_extract_cards(self) -> ToolResult:
        """提取当前页面的候选卡片。"""
        return await self.browser.extract_cards()

    @tool(
        name="browser_find_link_by_text",
        description="按文本在当前页面查找最匹配的链接。适用于已知目标标题或关键词，需要从列表中定位具体链接时使用。",
        parameters={
            "text": {
                "type": "string",
                "description": "要匹配的链接文本或关键词",
            }
        },
        required=["text"],
    )
    async def browser_find_link_by_text(self, text: str) -> ToolResult:
        """按文本在当前页面查找最匹配的链接。"""
        return await self.browser.find_link_by_text(text)

    @tool(
        name="browser_find_actionable_elements",
        description="提取当前页面的主要可交互元素。适用于登录、表单填写、按钮点击等交互任务，帮助在执行原子动作前先观察页面操作面。",
        parameters={},
        required=[],
    )
    async def browser_find_actionable_elements(self) -> ToolResult:
        """提取当前页面的主要可交互元素。"""
        return await self.browser.find_actionable_elements()

    @tool(
        name="browser_view",
        description="查看当前浏览器页面内容。仅在高阶页面阅读能力不适用时作为兜底观察工具使用。",
        parameters={},
        required=[],
    )
    async def brow_ser_view(self) -> ToolResult:
        """查看当前浏览器页面内容，用于确认已打开页面的最新状态。"""
        return await self.browser.view_page()

    @tool(
        name="browser_navigate",
        description="将浏览器导航到指定 URL，当需要访问新页面时使用",
        parameters={
            "url": {
                "type": "string",
                "description": "要导航到的完整 URL，必须包含协议前缀(例如:https://)",
            }
        },
        required=["url"],
    )
    async def browser_navigate(self, url: str) -> ToolResult:
        """将浏览器导航到指定 URL，当需要访问新页面时使用"""
        return await self.browser.navigate(url)

    @tool(
        name="browser_restart",
        description="重启浏览器并导航到指定URL，当需要重置浏览器时使用",
        parameters={
            "url": {
                "type": "string",
                "description": "要导航到的完整 URL，必须包含协议前缀(例如:https://)",
            }
        },
        required=["url"],
    )
    async def browser_restart(self, url: str) -> ToolResult:
        """重启浏览器并导航到指定URL，当需要重置浏览器时使用"""
        return await self.browser.restart(url)

    @tool(
        name="browser_click",
        description="点击当前页面上的元素。仅在已经通过高阶阅读/元素提取确认目标后，确实需要页面交互时使用。",
        parameters={
            "index": {
                "type": "integer",
                "description": "（可选）要点击的元素在页面中的索引",
            },
            "coordinate_x": {
                "type": "number",
                "description": "（可选）要点击的元素在页面中的 X 坐标",
            },
            "coordinate_y": {
                "type": "number",
                "description": "（可选）要点击的元素在页面中的 Y 坐标",
            }
        },
        required=[],
    )
    async def browser_click(
            self,
            index: Optional[int] = None,
            coordinate_x: Optional[float] = None,
            coordinate_y: Optional[float] = None,
    ) -> ToolResult:
        """点击当前页面上的元素，当需要点击当前页面上的元素时使用"""
        return await self.browser.click(
            index=index,
            coordinate_x=coordinate_x,
            coordinate_y=coordinate_y,
        )

    @tool(
        name="browser_input",
        description="覆盖浏览器当前页面可编辑区域的文本。仅在已经确认当前是交互任务且知道目标输入位置时使用。",
        parameters={
            "text": {
                "type": "string",
                "description": "要输入的完整文本内容",
            },
            "press_enter": {
                "type": "boolean",
                "description": "是否在输入完成后按下回车键",
            },
            "index": {
                "type": "integer",
                "description": "（可选）要输入文本的元素在页面中的索引",
            },
            "coordinate_x": {
                "type": "number",
                "description": "（可选）要输入文本的元素在页面中的 X 坐标",
            },
            "coordinate_y": {
                "type": "number",
                "description": "（可选）要输入文本的元素在页面中的 Y 坐标",
            }
        },
        required=["text", "press_enter"],
    )
    async def browser_input(
            self,
            text: str,
            press_enter: bool,
            index: Optional[int] = None,
            coordinate_x: Optional[float] = None,
            coordinate_y: Optional[float] = None,
    ) -> ToolResult:
        """覆盖浏览器当前页面可编辑区域的文本(input/textarea输入框)，当需要填充输入文本时使用"""
        return await self.browser.input(
            text=text,
            press_enter=press_enter,
            index=index,
            coordinate_x=coordinate_x,
            coordinate_y=coordinate_y,
        )

    @tool(
        name="browser_move_mouse",
        description="将鼠标光标移动到当前浏览器指定位置，用于模拟用户的鼠标移动，当需要移动鼠标时使用",
        parameters={
            "coordinate_x": {
                "type": "number",
                "description": "目标光标的 X 坐标",
            },
            "coordinate_y": {
                "type": "number",
                "description": "目标光标的 Y 坐标",
            }
        },
        required=["coordinate_x", "coordinate_y"],
    )
    async def browser_move_mouse(
            self,
            coordinate_x: float,
            coordinate_y: float,
    ) -> ToolResult:
        """将鼠标光标移动到当前浏览器指定位置，用于模拟用户的鼠标移动，当需要移动鼠标时使用"""
        return await self.browser.move_mouse(
            coordinate_x=coordinate_x,
            coordinate_y=coordinate_y,
        )

    @tool(
        name="browser_press_key",
        description="在当前浏览器用于模拟用户的按键操作，当需要指定特定的键盘操作时使用",
        parameters={
            "key": {
                "type": "string",
                "description": "要模拟按下的按键名称，例如: 'Enter'、'Tab'、'ArrowUp',支持组合键(例如:Control+Enter)",
            }
        },
        required=["key"],
    )
    async def browser_press_key(
            self,
            key: str,
    ) -> ToolResult:
        """在当前浏览器用于模拟用户的按键操作，当需要指定特定的键盘操作时使用"""
        return await self.browser.press_key(
            key=key,
        )

    @tool(
        name="browser_select_option",
        description="从当前浏览器页面的下拉列表元素中选择指定选项，用于选择下拉菜单中的选项。",
        parameters={
            "index": {
                "type": "integer",
                "description": "需要操作的下拉列表元素的索引(序号)",
            },
            "option": {
                "type": "integer",
                "description": "需要选择的选项序号，从0开始(注:指下拉框里的第几项)",
            }
        },
        required=["index", "option"],
    )
    async def browser_select_option(
            self,
            index: int,
            option: int,
    ) -> ToolResult:
        """从当前浏览器页面的下拉列表元素中选择指定选项，用于选择下拉菜单中的选项。"""
        return await self.browser.select_option(
            index=index,
            option=option,
        )

    @tool(
        name="browser_scroll_up",
        description="向上滚动当前浏览器页面。仅在高阶正文抽取无法满足时作为兜底动作使用。",
        parameters={
            "to_top": {
                "type": "boolean",
                "description": "（可选）是否滚动到页面顶部，而非向上滚动一屏",
            }
        },
        required=[],
    )
    async def browser_scroll_up(
            self,
            to_top: Optional[bool] = None,
    ) -> ToolResult:
        """向上滚动当前浏览器页面，用于模拟用户向上滚动页面或返回页面顶部。"""
        return await self.browser.scroll_up(
            to_top=to_top,
        )

    @tool(
        name="browser_scroll_down",
        description="向下滚动当前浏览器页面。仅在高阶正文抽取或卡片提取无法满足时作为兜底动作使用。",
        parameters={
            "to_bottom": {
                "type": "boolean",
                "description": "（可选）是否滚动到页面底部，而非向下滚动一屏",
            }
        },
        required=[],
    )
    async def browser_scroll_down(
            self,
            to_bottom: Optional[bool] = None,
    ) -> ToolResult:
        """向下滚动当前浏览器页面，用于模拟用户向下滚动页面或返回页面底部。"""
        return await self.browser.scroll_down(
            to_bottom=to_bottom,
        )

    @tool(
        name="browser_console_exec",
        description="在浏览器控制台执行 JavaScript 脚本，当需要指定自定义脚本时使用",
        parameters={
            "javascript": {
                "type": "string",
                "description": "要执行的 JavaScript 脚本，请注意运行时环境为浏览器控制台",
            }
        },
        required=["javascript"],
    )
    async def browser_console_exec(
            self,
            javascript: str,
    ) -> ToolResult:
        """在浏览器控制台执行 JavaScript 脚本，当需要指定自定义脚本时使用"""
        return await self.browser.console_exec(
            javascript=javascript,
        )

    @tool(
        name="browser_console_view",
        description="查看当前浏览器控制台的输出内容，用于检查javascript日志或调试页面错误",
        parameters={
            "max_lines": {
                "type": "integer",
                "description": "（可选）返回的日志最大行数",
            }
        },
        required=[],
    )
    async def browser_console_view(
            self,
            max_lines: Optional[int] = None,
    ) -> ToolResult:
        """查看当前浏览器控制台的输出内容，当需要查看控制台输出时使用"""
        return await self.browser.console_view(
            max_lines=max_lines
        )
