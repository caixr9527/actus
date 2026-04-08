import asyncio

from app.infrastructure.external.browser.playwright_browser import PlaywrightBrowser


class _FakePage:
    def __init__(self, raw_data: dict) -> None:
        self._raw_data = raw_data
        self.interactive_elements_cache = []

    async def evaluate(self, script):
        script_text = str(script)
        if "document.readyState" in script_text:
            return True
        return self._raw_data

    def is_closed(self) -> bool:
        return False


def _build_raw_page_data() -> dict:
    return {
        "url": "https://example.com/docs/runtime",
        "title": "Runtime Docs",
        "main_heading": "Runtime Overview",
        "main_text": "This page explains the runtime architecture in detail. " * 30,
        "main_html": "<article><h1>Runtime Overview</h1><p>This page explains the runtime architecture.</p></article>",
        "body_text": "Runtime body",
        "card_candidates": [
            {
                "index": 0,
                "title": "Runtime Overview",
                "summary": "Architecture guide",
                "url": "https://example.com/docs/runtime",
                "tags": ["docs"],
            },
            {
                "index": 1,
                "title": "Execution Model",
                "summary": "Execution details",
                "url": "https://example.com/docs/execution",
                "tags": ["guide"],
            },
        ],
        "actionable_elements": [
            {
                "index": 0,
                "tag": "a",
                "text": "Runtime Overview",
                "role": "",
                "selector": "[data-manus-id='manus-element-0']",
            },
            {
                "index": 1,
                "tag": "a",
                "text": "Execution Model",
                "role": "",
                "selector": "[data-manus-id='manus-element-1']",
            }
        ],
        "paragraph_count": 8,
        "heading_count": 3,
        "form_count": 0,
        "link_count": 12,
        "scroll_progress": 35,
        "should_continue_scrolling": True,
    }


def _build_browser(raw_data: dict | None = None) -> PlaywrightBrowser:
    browser = PlaywrightBrowser(cdp_url="ws://example.com", llm=None)
    browser.page = _FakePage(raw_data or _build_raw_page_data())
    browser.browser = type("_FakeBrowser", (), {"contexts": []})()
    return browser


def test_playwright_browser_should_read_current_page_structured() -> None:
    browser = _build_browser()

    result = asyncio.run(browser.read_current_page_structured())

    assert result.success is True
    assert result.data is not None
    assert result.data.title == "Runtime Docs"
    assert result.data.main_heading == "Runtime Overview"
    assert result.data.page_type.value == "document"
    assert len(result.data.cards) == 2
    assert len(result.data.actionable_elements) == 2
    assert browser.page.interactive_elements_cache[0]["text"] == "Runtime Overview"


def test_playwright_browser_should_extract_main_content() -> None:
    browser = _build_browser()

    result = asyncio.run(browser.extract_main_content())

    assert result.success is True
    assert result.data is not None
    assert "Runtime Overview" in result.data.content
    assert result.data.content_length > 0


def test_playwright_browser_should_extract_cards() -> None:
    browser = _build_browser()

    result = asyncio.run(browser.extract_cards())

    assert result.success is True
    assert result.data is not None
    assert result.data.total_cards == 2
    assert result.data.cards[1].title == "Execution Model"


def test_playwright_browser_should_find_link_by_text() -> None:
    browser = _build_browser()

    result = asyncio.run(browser.find_link_by_text("execution"))

    assert result.success is True
    assert result.data is not None
    assert result.data.url == "https://example.com/docs/execution"
    assert result.data.card is not None
    assert result.data.card.title == "Execution Model"
    assert result.data.index == 1
    assert result.data.selector == "[data-manus-id='manus-element-1']"


def test_playwright_browser_should_find_actionable_elements() -> None:
    browser = _build_browser()

    result = asyncio.run(browser.find_actionable_elements())

    assert result.success is True
    assert result.data is not None
    assert result.data.page_type.value == "document"
    assert result.data.elements[0].selector == "[data-manus-id='manus-element-0']"
