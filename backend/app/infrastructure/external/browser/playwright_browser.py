#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time   : 2025/12/8 17:35
@Author : caixiaorong01@outlook.com
@File   : playwright_browser.py
"""
import asyncio
import logging
from typing import Any, List, Optional

from markdownify import markdownify
from playwright.async_api import Browser, Page, Playwright, async_playwright

from app.domain.external import Browser as BrowserProtocol, LLM
from app.domain.models import (
    BrowserActionableElement,
    BrowserActionableElementsResult,
    BrowserCardExtractionResult,
    BrowserCardItem,
    BrowserLinkMatchResult,
    BrowserMainContentResult,
    BrowserPageStructuredResult,
    BrowserPageType,
    ToolResult,
)
from .playwright_browser_fun import GET_PAGE_STRUCTURED_DATA_FUNC, INJECT_CONSOLE_LOGS_FUNC

logger = logging.getLogger(__name__)

MAIN_CONTENT_MAX_CHARS = 12000
MAIN_CONTENT_PREVIEW_MAX_CHARS = 1200
CONTENT_SUMMARY_MAX_CHARS = 320


class PlaywrightBrowser(BrowserProtocol):
    """Playwright 浏览器服务实现。"""

    def __init__(
            self,
            cdp_url: str,
            llm: Optional[LLM],
    ) -> None:
        self.llm: Optional[LLM] = llm
        self.cdp_url: str = cdp_url
        self.playwright: Optional[Playwright] = None
        self.browser: Optional[Browser] = None
        self.page: Optional[Page] = None

    async def _ensure_browser(self) -> None:
        if not self.browser or not self.page:
            if not await self.initialize():
                raise RuntimeError("初始化浏览器服务失败")

    async def _ensure_page(self) -> None:
        await self._ensure_browser()
        if not self.page:
            self.page = await self.browser.new_page()
            return

        contexts = self.browser.contexts if self.browser is not None else []
        if not contexts:
            return
        latest_pages = contexts[0].pages
        if latest_pages:
            self.page = latest_pages[-1]

    async def _evaluate_structured_page_data(self) -> dict[str, Any]:
        await self._ensure_page()
        raw_data = await self.page.evaluate(GET_PAGE_STRUCTURED_DATA_FUNC)
        return raw_data if isinstance(raw_data, dict) else {}

    @staticmethod
    def _normalize_text(value: Any, *, max_chars: int = 0) -> str:
        text = " ".join(str(value or "").split()).strip()
        if max_chars > 0 and len(text) > max_chars:
            return text[:max_chars - 3] + "..."
        return text

    @classmethod
    def _normalize_card_items(cls, raw_cards: Any) -> List[BrowserCardItem]:
        cards: List[BrowserCardItem] = []
        for index, item in enumerate(list(raw_cards or [])):
            if not isinstance(item, dict):
                continue
            title = cls._normalize_text(item.get("title"), max_chars=160)
            url = cls._normalize_text(item.get("url"), max_chars=500)
            if not title or not url:
                continue
            tags = [
                cls._normalize_text(tag, max_chars=40)
                for tag in list(item.get("tags") or [])
                if cls._normalize_text(tag, max_chars=40)
            ]
            cards.append(
                BrowserCardItem(
                    index=int(item.get("index") or index),
                    title=title,
                    summary=cls._normalize_text(item.get("summary"), max_chars=240),
                    url=url,
                    tags=tags[:6],
                )
            )
        return cards

    @classmethod
    def _normalize_actionable_elements(cls, raw_elements: Any) -> List[BrowserActionableElement]:
        elements: List[BrowserActionableElement] = []
        for index, item in enumerate(list(raw_elements or [])):
            if not isinstance(item, dict):
                continue
            elements.append(
                BrowserActionableElement(
                    index=int(item.get("index") or index),
                    tag=cls._normalize_text(item.get("tag"), max_chars=40),
                    text=cls._normalize_text(item.get("text"), max_chars=120),
                    role=cls._normalize_text(item.get("role"), max_chars=40),
                    selector=cls._normalize_text(item.get("selector"), max_chars=160),
                )
            )
        return elements

    @classmethod
    def _detect_page_type(
            cls,
            *,
            url: str,
            title: str,
            main_text: str,
            card_count: int,
            form_count: int,
            paragraph_count: int,
    ) -> BrowserPageType:
        normalized_url = url.lower()
        normalized_title = title.lower()
        if form_count > 0:
            return BrowserPageType.FORM
        if card_count >= 5:
            if any(token in normalized_url or token in normalized_title for token in
                   ("search", "result", "query", "q=")):
                return BrowserPageType.SEARCH_RESULTS
            return BrowserPageType.LISTING
        if any(token in normalized_url or token in normalized_title for token in
               ("docs", "documentation", "reference", "manual")):
            return BrowserPageType.DOCUMENT
        if paragraph_count >= 4 or len(main_text) >= 900:
            return BrowserPageType.ARTICLE
        return BrowserPageType.GENERIC

    @classmethod
    def _build_content_summary(
            cls,
            *,
            main_text: str,
            cards: List[BrowserCardItem],
    ) -> str:
        if main_text:
            return cls._normalize_text(main_text, max_chars=CONTENT_SUMMARY_MAX_CHARS)
        if cards:
            return cls._normalize_text("；".join(card.title for card in cards[:4]), max_chars=CONTENT_SUMMARY_MAX_CHARS)
        return ""

    @classmethod
    def _build_structured_result(cls, raw_data: dict[str, Any]) -> BrowserPageStructuredResult:
        cards = cls._normalize_card_items(raw_data.get("card_candidates"))
        actionable_elements = cls._normalize_actionable_elements(raw_data.get("actionable_elements"))
        url = cls._normalize_text(raw_data.get("url"), max_chars=500)
        title = cls._normalize_text(raw_data.get("title"), max_chars=200)
        main_heading = cls._normalize_text(raw_data.get("main_heading"), max_chars=200)
        main_text = cls._normalize_text(raw_data.get("main_text"), max_chars=MAIN_CONTENT_PREVIEW_MAX_CHARS)
        page_type = cls._detect_page_type(
            url=url,
            title=title,
            main_text=cls._normalize_text(raw_data.get("main_text"), max_chars=0),
            card_count=len(cards),
            form_count=int(raw_data.get("form_count") or 0),
            paragraph_count=int(raw_data.get("paragraph_count") or 0),
        )
        return BrowserPageStructuredResult(
            url=url,
            title=title,
            main_heading=main_heading,
            page_type=page_type,
            content_summary=cls._build_content_summary(
                main_text=cls._normalize_text(raw_data.get("main_text"), max_chars=0),
                cards=cards,
            ),
            main_content_preview=main_text,
            cards=cards,
            actionable_elements=actionable_elements,
            should_continue_scrolling=bool(raw_data.get("should_continue_scrolling")),
            scroll_progress=float(raw_data.get("scroll_progress") or 0.0),
        )

    @classmethod
    def _build_main_content_result(cls, raw_data: dict[str, Any]) -> BrowserMainContentResult:
        structured = cls._build_structured_result(raw_data)
        raw_html = str(raw_data.get("main_html") or "").strip()
        raw_main_text = cls._normalize_text(raw_data.get("main_text"), max_chars=0)
        content = markdownify(raw_html) if raw_html else raw_main_text
        content = cls._normalize_text(content, max_chars=MAIN_CONTENT_MAX_CHARS)
        excerpt = cls._normalize_text(content, max_chars=CONTENT_SUMMARY_MAX_CHARS)
        return BrowserMainContentResult(
            url=structured.url,
            title=structured.title,
            page_type=structured.page_type,
            content=content,
            excerpt=excerpt,
            content_length=len(content),
            truncated=len(content) >= MAIN_CONTENT_MAX_CHARS,
        )

    @classmethod
    def _build_card_result(cls, raw_data: dict[str, Any]) -> BrowserCardExtractionResult:
        structured = cls._build_structured_result(raw_data)
        return BrowserCardExtractionResult(
            url=structured.url,
            title=structured.title,
            page_type=structured.page_type,
            cards=structured.cards,
            total_cards=len(structured.cards),
        )

    async def _refresh_interactive_elements_cache(self, structured: BrowserPageStructuredResult) -> None:
        await self._ensure_page()
        # 原子点击/输入仍基于页面缓存索引，因此高阶结构化提取后同步刷新缓存。
        self.page.interactive_elements_cache = [item.model_dump(mode="json") for item in structured.actionable_elements]

    async def _read_structured_result(self) -> BrowserPageStructuredResult:
        await self.wait_for_page_load(timeout=8)
        raw_data = await self._evaluate_structured_page_data()
        structured = self._build_structured_result(raw_data)
        await self._refresh_interactive_elements_cache(structured)
        logger.info(
            "浏览器结构化页面提取完成",
            extra={
                "url": structured.url,
                "page_type": structured.page_type.value,
                "card_count": len(structured.cards),
                "actionable_count": len(structured.actionable_elements),
                "scroll_progress": structured.scroll_progress,
            },
        )
        return structured

    async def read_current_page_structured(self) -> ToolResult[BrowserPageStructuredResult]:
        try:
            return ToolResult(success=True, data=await self._read_structured_result())
        except Exception as e:
            logger.warning("浏览器结构化页面提取失败: %s", e)
            return ToolResult(success=False, message=f"读取当前页面结构化摘要失败: {e}")

    async def extract_main_content(self) -> ToolResult[BrowserMainContentResult]:
        try:
            await self.wait_for_page_load(timeout=8)
            raw_data = await self._evaluate_structured_page_data()
            main_content = self._build_main_content_result(raw_data)
            if not main_content.content:
                return ToolResult(success=False, message="当前页面未提取到可用正文")
            logger.info(
                "浏览器正文提取完成",
                extra={
                    "url": main_content.url,
                    "page_type": main_content.page_type.value,
                    "content_length": main_content.content_length,
                },
            )
            return ToolResult(success=True, data=main_content)
        except Exception as e:
            logger.warning("浏览器正文提取失败: %s", e)
            return ToolResult(success=False, message=f"提取当前页面正文失败: {e}")

    async def extract_cards(self) -> ToolResult[BrowserCardExtractionResult]:
        try:
            await self.wait_for_page_load(timeout=8)
            raw_data = await self._evaluate_structured_page_data()
            card_result = self._build_card_result(raw_data)
            if len(card_result.cards) == 0:
                return ToolResult(success=False, message="当前页面未提取到候选卡片")
            logger.info(
                "浏览器候选卡片提取完成",
                extra={
                    "url": card_result.url,
                    "page_type": card_result.page_type.value,
                    "card_count": card_result.total_cards,
                },
            )
            return ToolResult(success=True, data=card_result)
        except Exception as e:
            logger.warning("浏览器候选卡片提取失败: %s", e)
            return ToolResult(success=False, message=f"提取当前页面候选卡片失败: {e}")

    async def find_link_by_text(self, text: str) -> ToolResult[BrowserLinkMatchResult]:
        query = self._normalize_text(text, max_chars=120)
        if not query:
            return ToolResult(success=False, message="查找链接时缺少 text 参数")

        try:
            raw_data = await self._evaluate_structured_page_data()
            cards = self._normalize_card_items(raw_data.get("card_candidates"))
            actionable_elements = self._normalize_actionable_elements(raw_data.get("actionable_elements"))
            normalized_query = query.lower()
            best_card: Optional[BrowserCardItem] = None
            best_score = -1
            for card in cards:
                haystack = f"{card.title} {card.summary}".lower()
                score = 0
                if normalized_query in haystack:
                    score += 10
                score += sum(2 for token in normalized_query.split() if token and token in haystack)
                if score > best_score:
                    best_score = score
                    best_card = card
            if best_card is None or best_score <= 0:
                return ToolResult(success=False, message=f"当前页面未找到与“{query}”匹配的链接")
            matched_selector = ""
            matched_index: Optional[int] = None
            normalized_card_title = best_card.title.lower()
            for element in actionable_elements:
                haystack = f"{element.text} {element.role}".lower()
                if normalized_card_title and normalized_card_title in haystack:
                    matched_selector = element.selector
                    matched_index = element.index
                    break
                if normalized_query and normalized_query in haystack:
                    matched_selector = element.selector
                    matched_index = element.index
                    break
            return ToolResult(
                success=True,
                data=BrowserLinkMatchResult(
                    query=query,
                    matched_text=best_card.title,
                    url=best_card.url,
                    card=best_card,
                    index=matched_index,
                    selector=matched_selector,
                ),
            )
        except Exception as e:
            logger.warning("浏览器按文本查找链接失败: %s", e)
            return ToolResult(success=False, message=f"按文本查找链接失败: {e}")

    async def find_actionable_elements(self) -> ToolResult[BrowserActionableElementsResult]:
        try:
            structured = await self._read_structured_result()
            return ToolResult(
                success=True,
                data=BrowserActionableElementsResult(
                    url=structured.url,
                    title=structured.title,
                    page_type=structured.page_type,
                    elements=structured.actionable_elements,
                ),
            )
        except Exception as e:
            logger.warning("浏览器可交互元素提取失败: %s", e)
            return ToolResult(success=False, message=f"提取当前页面可交互元素失败: {e}")

    async def _get_element_by_id(self, index: int) -> Optional[Any]:
        if (
                not hasattr(self.page, "interactive_elements_cache")
                or not self.page.interactive_elements_cache
                or index >= len(self.page.interactive_elements_cache)
        ):
            return None
        selector = f'[data-manus-id="manus-element-{index}"]'
        return await self.page.query_selector(selector)

    async def _build_post_action_result(self) -> ToolResult[BrowserPageStructuredResult]:
        structured = await self._read_structured_result()
        return ToolResult(success=True, data=structured)

    async def initialize(self) -> bool:
        max_retries = 5
        retry_interval = 1
        for attempt in range(max_retries):
            try:
                self.playwright = await async_playwright().start()
                self.browser = await self.playwright.chromium.connect_over_cdp(self.cdp_url)
                contexts = self.browser.contexts
                if contexts and len(contexts[0].pages) == 1:
                    page = contexts[0].pages[0]
                    if page.url in {"about:blank", "chrome://newtab/", "chrome://new-tab-page/"} or not page.url:
                        self.page = page
                    else:
                        self.page = await contexts[0].new_page()
                else:
                    context = contexts[0] if contexts else await self.browser.new_context()
                    self.page = await context.new_page()
                return True
            except Exception as e:
                await self.cleanup()
                if attempt == max_retries - 1:
                    logger.error("初始化浏览器服务失败，已重试 %s 次: %s", max_retries, e)
                    return False
                retry_interval = min(retry_interval * 2, 10)
                logger.warning("初始化浏览器服务失败，准备重试(%s/%s)", attempt + 1, max_retries)
                await asyncio.sleep(retry_interval)
        return False

    async def cleanup(self) -> None:
        try:
            if self.browser:
                for context in self.browser.contexts:
                    for page in context.pages:
                        if not page.is_closed():
                            await page.close()
            if self.page and not self.page.is_closed():
                await self.page.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
        except Exception as e:
            logger.error("清理浏览器服务时发生错误: %s", e)
        finally:
            self.page = None
            self.browser = None
            self.playwright = None

    async def wait_for_page_load(self, timeout: int = 15) -> bool:
        await self._ensure_page()
        start_time = asyncio.get_event_loop().time()
        check_interval = 0.5
        while asyncio.get_event_loop().time() - start_time < timeout:
            is_completed = await self.page.evaluate("""() => document.readyState === 'complete'""")
            if is_completed:
                return True
            await asyncio.sleep(check_interval)
        return False

    async def navigate(self, url: str) -> ToolResult:
        await self._ensure_page()
        try:
            self.page.interactive_elements_cache = []
            await self.page.goto(url)
            return await self._build_post_action_result()
        except Exception as e:
            return ToolResult(success=False, message=f"导航到 URL {url} 失败: {e}")

    async def view_page(self) -> ToolResult:
        return await self.read_current_page_structured()

    async def click(
            self,
            index: Optional[int] = None,
            coordinate_x: Optional[float] = None,
            coordinate_y: Optional[float] = None,
    ) -> ToolResult:
        await self._ensure_page()
        if coordinate_x is not None and coordinate_y is not None:
            await self.page.mouse.click(coordinate_x, coordinate_y)
            return await self._build_post_action_result()
        if index is None:
            return ToolResult(success=False, message="点击元素失败，缺少 index 或坐标")
        try:
            element = await self._get_element_by_id(index)
            if not element:
                return ToolResult(success=False, message=f"未找到索引为 {index} 的元素")
            await element.click(timeout=5000)
            return await self._build_post_action_result()
        except Exception as e:
            return ToolResult(success=False, message=f"点击元素失败: {e}")

    async def input(
            self,
            text: str,
            press_enter: bool,
            index: Optional[int] = None,
            coordinate_x: Optional[float] = None,
            coordinate_y: Optional[float] = None,
    ) -> ToolResult:
        await self._ensure_page()
        try:
            if coordinate_x is not None and coordinate_y is not None:
                await self.page.mouse.click(coordinate_x, coordinate_y)
                await self.page.keyboard.type(text)
            elif index is not None:
                element = await self._get_element_by_id(index)
                if not element:
                    return ToolResult(success=False, message="输入文本失败，目标元素不存在")
                await element.fill("")
                await element.type(text)
            else:
                return ToolResult(success=False, message="输入文本失败，缺少 index 或坐标")
            if press_enter:
                await self.page.keyboard.press("Enter")
            return await self._build_post_action_result()
        except Exception as e:
            return ToolResult(success=False, message=f"输入文本失败: {e}")

    async def move_mouse(self, coordinate_x: float, coordinate_y: float) -> ToolResult:
        await self._ensure_page()
        await self.page.mouse.move(coordinate_x, coordinate_y)
        return ToolResult(success=True)

    async def press_key(self, key: str) -> ToolResult:
        await self._ensure_page()
        await self.page.keyboard.press(key)
        return await self._build_post_action_result()

    async def select_option(self, index: int, option: int) -> ToolResult:
        await self._ensure_page()
        try:
            element = await self._get_element_by_id(index)
            if not element:
                return ToolResult(success=False, message=f"索引[{index}]对应的下拉菜单不存在")
            await element.select_option(index=option)
            return await self._build_post_action_result()
        except Exception as e:
            return ToolResult(success=False, message=f"选择下拉菜单选项失败: {e}")

    async def restart(self, url: str) -> ToolResult:
        await self.cleanup()
        return await self.navigate(url)

    async def scroll_up(self, to_top: Optional[bool] = None) -> ToolResult:
        await self._ensure_page()
        if to_top:
            await self.page.evaluate("window.scrollTo(0, 0)")
        else:
            await self.page.evaluate("window.scrollBy(0, -window.innerHeight)")
        return await self._build_post_action_result()

    async def scroll_down(self, to_bottom: Optional[bool] = None) -> ToolResult:
        await self._ensure_page()
        if to_bottom:
            await self.page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        else:
            await self.page.evaluate("window.scrollBy(0, window.innerHeight)")
        return await self._build_post_action_result()

    async def screenshot(self, full_page: Optional[bool] = None) -> bytes:
        await self._ensure_page()
        return await self.page.screenshot(full_page=full_page, type="png")

    async def console_exec(self, javascript: str) -> ToolResult:
        await self._ensure_page()
        try:
            await self.page.evaluate(INJECT_CONSOLE_LOGS_FUNC)
        except Exception as e:
            logger.warning("注入 console 日志捕获失败: %s", e)
        result = await self.page.evaluate(javascript)
        return ToolResult(success=True, data={"result": result})

    async def console_view(self, max_lines: Optional[int] = None) -> ToolResult:
        await self._ensure_page()
        logs = await self.page.evaluate("""() => window.console.logs || []""")
        if max_lines is not None:
            logs = logs[-max_lines:]
        return ToolResult(success=True, data={"logs": logs})
