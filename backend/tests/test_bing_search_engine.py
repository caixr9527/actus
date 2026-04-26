import asyncio

import httpx

from app.infrastructure.external.search.bing_search import BingSearchEngine


class _FakeResponse:
    def __init__(self, html: str, status_code: int = 200) -> None:
        self._response = httpx.Response(
            status_code=status_code,
            text=html,
            request=httpx.Request("GET", "https://www.bing.com/search"),
        )

    @property
    def text(self) -> str:
        return self._response.text

    @property
    def cookies(self):
        return self._response.cookies

    def raise_for_status(self) -> None:
        self._response.raise_for_status()


class _FakeAsyncClient:
    default_html: str = ""
    status_code: int = 200
    html_by_query: dict[str, str] = {}
    requests: list[dict] = []

    def __init__(self, *, headers=None, cookies=None, timeout=None, follow_redirects=None) -> None:
        self._headers = dict(headers or {})

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def get(self, _url: str, params=None):
        current_params = dict(params or {})
        _FakeAsyncClient.requests.append(
            {
                "params": current_params,
                "headers": dict(self._headers),
            }
        )
        query = str(current_params.get("q") or "")
        html = _FakeAsyncClient.html_by_query.get(query, _FakeAsyncClient.default_html)
        return _FakeResponse(html=html, status_code=_FakeAsyncClient.status_code)


def _reset_fake_client() -> None:
    _FakeAsyncClient.default_html = ""
    _FakeAsyncClient.status_code = 200
    _FakeAsyncClient.html_by_query = {}
    _FakeAsyncClient.requests = []


def test_bing_search_should_build_unescaped_time_filter_and_zh_locale(monkeypatch) -> None:
    _reset_fake_client()
    query = "北京 天气"
    _FakeAsyncClient.html_by_query = {
        query: """
        <html><body>
          <li class='b_algo'>
            <h2><a href='https://weather.example.com/beijing'>北京天气预报</a></h2>
            <div class='b_caption'><p>北京天气预报，今天晴。</p></div>
          </li>
          <li class='b_algo'>
            <h2><a href='https://news.example.com/beijing-weather'>北京天气新闻</a></h2>
            <div class='b_caption'><p>北京天气相关资讯。</p></div>
          </li>
        </body></html>
        """,
    }
    monkeypatch.setattr("app.infrastructure.external.search.bing_search.httpx.AsyncClient", _FakeAsyncClient)

    engine = BingSearchEngine()
    result = asyncio.run(engine.invoke(query=query, date_range="past_day"))

    assert result.success is True
    assert len(_FakeAsyncClient.requests) == 1
    request = _FakeAsyncClient.requests[0]
    assert request["params"]["qft"] == "+filterui:age-lt1440"
    assert "%3" not in request["params"]["qft"]
    assert request["params"]["mkt"] == "zh-CN"
    assert request["headers"]["Accept-Language"].startswith("zh-CN")


def test_bing_search_should_dedupe_and_rank_results(monkeypatch) -> None:
    _reset_fake_client()
    _FakeAsyncClient.default_html = """
    <html><body>
      <span class='sb_count'>约 12,345 条结果</span>
      <li class='b_algo'>
        <h2><a href='https://news.example.com/a?id=1&utm=1'>北京天气预报</a></h2>
        <div class='b_caption'><p>今天北京天气晴，最高气温 20 度。</p></div>
      </li>
      <li class='b_algo'>
        <h2><a href='https://news.example.com/a?id=1&utm=2'>北京天气预报（重复）</a></h2>
        <div class='b_caption'><p>重复结果。</p></div>
      </li>
      <li class='b_algo'>
        <h2><a href='https://other.example.org/x'>其他内容</a></h2>
        <div class='b_caption'><p>与天气关系较弱。</p></div>
      </li>
    </body></html>
    """
    monkeypatch.setattr("app.infrastructure.external.search.bing_search.httpx.AsyncClient", _FakeAsyncClient)

    engine = BingSearchEngine()
    result = asyncio.run(engine.invoke(query="北京 天气", date_range="all"))

    assert result.success is True
    assert result.data is not None
    assert result.data.total_results == 12345
    assert len(result.data.results) == 2
    assert result.data.results[0].url.startswith("https://news.example.com/a")


def test_bing_search_should_ignore_unknown_date_range(monkeypatch) -> None:
    _reset_fake_client()
    _FakeAsyncClient.default_html = """
    <html><body>
      <li class='b_algo'>
        <h2><a href='https://news.example.com/openai-a'>OpenAI 发布会</a></h2>
        <div class='b_caption'><p>OpenAI 最新模型能力更新。</p></div>
      </li>
      <li class='b_algo'>
        <h2><a href='https://blog.example.org/openai-b'>OpenAI 技术解读</a></h2>
        <div class='b_caption'><p>围绕 OpenAI 的技术路线分析。</p></div>
      </li>
    </body></html>
    """
    monkeypatch.setattr("app.infrastructure.external.search.bing_search.httpx.AsyncClient", _FakeAsyncClient)

    engine = BingSearchEngine()
    result = asyncio.run(engine.invoke(query="openai", date_range="invalid-range"))

    assert result.success is True
    request = _FakeAsyncClient.requests[0]
    assert "qft" not in request["params"]
    assert "filters" not in request["params"]


def test_bing_search_should_fail_when_results_remain_low_quality(monkeypatch) -> None:
    _reset_fake_client()
    _FakeAsyncClient.default_html = """
    <html><body>
      <li class='b_algo'><h2><a href='https://noise.example.com/1'>无关内容 1</a></h2><p>和目标主题无关。</p></li>
      <li class='b_algo'><h2><a href='https://noise.example.com/2'>无关内容 2</a></h2><p>和目标主题无关。</p></li>
      <li class='b_algo'><h2><a href='https://noise.example.com/3'>无关内容 3</a></h2><p>和目标主题无关。</p></li>
    </body></html>
    """
    monkeypatch.setattr("app.infrastructure.external.search.bing_search.httpx.AsyncClient", _FakeAsyncClient)

    engine = BingSearchEngine()
    result = asyncio.run(engine.invoke(query="慕课网 AI Agent 体系课", date_range="all"))

    assert result.success is False
    assert result.data is not None
    assert result.data.total_results == 0
    assert result.data.results == []
    assert "相关性较低" in (result.message or "")
    assert len(_FakeAsyncClient.requests) >= 2


def test_bing_search_should_expand_query_variants_when_first_page_low_quality(monkeypatch) -> None:
    _reset_fake_client()
    original_query = "慕课网 AI Agent 体系课"
    quoted_query = f'"{original_query}"'

    # 首轮结果低质量：同域名 + 无关键词覆盖。
    _FakeAsyncClient.html_by_query[original_query] = """
    <html><body>
      <li class='b_algo'><h2><a href='https://noise.example.com/1'>无关问题1</a></h2><p>与主题无关</p></li>
      <li class='b_algo'><h2><a href='https://noise.example.com/2'>无关问题2</a></h2><p>与主题无关</p></li>
      <li class='b_algo'><h2><a href='https://noise.example.com/3'>无关问题3</a></h2><p>与主题无关</p></li>
    </body></html>
    """
    # 次轮短语检索返回相关结果。
    _FakeAsyncClient.html_by_query[quoted_query] = """
    <html><body>
      <li class='b_algo'>
        <h2><a href='https://course.example.com/ai-agent'>慕课网 AI Agent 体系课</a></h2>
        <div class='b_caption'><p>AI Agent 体系化学习课程介绍。</p></div>
      </li>
    </body></html>
    """
    monkeypatch.setattr("app.infrastructure.external.search.bing_search.httpx.AsyncClient", _FakeAsyncClient)

    engine = BingSearchEngine()
    result = asyncio.run(engine.invoke(query=original_query, date_range="all"))

    assert result.success is True
    assert result.data is not None
    assert len(_FakeAsyncClient.requests) >= 2
    requested_queries = [entry["params"].get("q") for entry in _FakeAsyncClient.requests]
    assert original_query in requested_queries
    assert quoted_query in requested_queries
    assert any("AI Agent" in item.title for item in result.data.results)
