#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""浏览器高阶能力的结构化领域模型。"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class BrowserPageType(str, Enum):
    """页面类型。

    用于让执行器和前端理解当前页面更接近“正文页”“列表页”还是“表单页”。
    """

    ARTICLE = "article"
    DOCUMENT = "document"
    LISTING = "listing"
    SEARCH_RESULTS = "search_results"
    FORM = "form"
    GENERIC = "generic"


class BrowserCardItem(BaseModel):
    """列表型页面中的候选卡片。"""

    index: int = 0
    title: str = ""
    summary: str = ""
    url: str = ""
    tags: List[str] = Field(default_factory=list)


class BrowserActionableElement(BaseModel):
    """供模型决策点击/输入前使用的可交互元素摘要。"""

    index: int = 0
    tag: str = ""
    text: str = ""
    role: str = ""
    selector: str = ""


class BrowserPageStructuredResult(BaseModel):
    """当前页面的统一结构化摘要。"""

    url: str = ""
    title: str = ""
    main_heading: str = ""
    page_type: BrowserPageType = BrowserPageType.GENERIC
    content_summary: str = ""
    main_content_preview: str = ""
    cards: List[BrowserCardItem] = Field(default_factory=list)
    actionable_elements: List[BrowserActionableElement] = Field(default_factory=list)
    should_continue_scrolling: bool = False
    scroll_progress: float = 0.0


class BrowserMainContentResult(BaseModel):
    """正文抽取结果。"""

    url: str = ""
    title: str = ""
    page_type: BrowserPageType = BrowserPageType.GENERIC
    content: str = ""
    excerpt: str = ""
    content_length: int = 0
    truncated: bool = False


class BrowserCardExtractionResult(BaseModel):
    """列表页候选抽取结果。"""

    url: str = ""
    title: str = ""
    page_type: BrowserPageType = BrowserPageType.GENERIC
    cards: List[BrowserCardItem] = Field(default_factory=list)
    total_cards: int = 0


class BrowserLinkMatchResult(BaseModel):
    """按文本查找链接的结果。"""

    query: str = ""
    matched_text: str = ""
    url: str = ""
    card: Optional[BrowserCardItem] = None
    index: Optional[int] = None
    selector: str = ""


class BrowserActionableElementsResult(BaseModel):
    """页面可交互元素提取结果。"""

    url: str = ""
    title: str = ""
    page_type: BrowserPageType = BrowserPageType.GENERIC
    elements: List[BrowserActionableElement] = Field(default_factory=list)
