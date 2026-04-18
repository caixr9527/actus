import asyncio
from types import SimpleNamespace

from app.infrastructure.sandbox import docker_sandbox


class _FakeContainer:
    def __init__(self) -> None:
        self.attrs = {
            "NetworkSettings": {
                "IPAddress": "172.18.0.9",
            }
        }

    def reload(self) -> None:
        return None


class _FakeContainers:
    def __init__(self) -> None:
        self.last_run_kwargs: dict | None = None

    def run(self, **kwargs):
        self.last_run_kwargs = kwargs
        return _FakeContainer()


class _FakeDockerClient:
    def __init__(self) -> None:
        self.containers = _FakeContainers()


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class _FakeAsyncClient:
    def __init__(self) -> None:
        self.last_get_url: str | None = None
        self.last_post_url: str | None = None
        self.last_post_json: dict | None = None

    async def get(self, url: str, **kwargs) -> _FakeResponse:
        self.last_get_url = url
        return _FakeResponse({"code": 200, "msg": "ok", "data": {"available": True}})

    async def post(self, url: str, json: dict | None = None, **kwargs) -> _FakeResponse:
        self.last_post_url = url
        self.last_post_json = json
        return _FakeResponse({"code": 200, "msg": "ok", "data": {"query": "openai", "results": []}})


def test_create_task_should_pass_random_searxng_secret_to_container_environment(monkeypatch) -> None:
    fake_client = _FakeDockerClient()
    monkeypatch.setattr(docker_sandbox, "get_settings", lambda: SimpleNamespace(
        sandbox_image="actus-sandbox",
        sandbox_name_prefix="actus-sandbox",
        sandbox_ttl_minutes=60,
        sandbox_chrome_args="--headless=false",
        sandbox_https_proxy="https://proxy.example.com",
        sandbox_http_proxy="http://proxy.example.com",
        sandbox_no_proxy="localhost,127.0.0.1",
        sandbox_network="actus-network",
    ))
    monkeypatch.setattr(docker_sandbox.docker, "from_env", lambda: fake_client)
    monkeypatch.setattr(docker_sandbox.uuid, "uuid4", lambda: "sandbox-uuid")
    monkeypatch.setattr(
        docker_sandbox.DockerSandbox,
        "_generate_searxng_secret",
        staticmethod(lambda: "random-searxng-secret"),
    )

    sandbox = docker_sandbox.DockerSandbox._create_task()

    assert sandbox.id == "actus-sandbox-sandbox-uuid"
    assert fake_client.containers.last_run_kwargs is not None
    assert fake_client.containers.last_run_kwargs["network"] == "actus-network"
    assert fake_client.containers.last_run_kwargs["environment"]["SEARXNG_SECRET"] == "random-searxng-secret"


def test_get_searxng_status_should_call_sandbox_status_endpoint() -> None:
    fake_client = _FakeAsyncClient()
    sandbox = docker_sandbox.DockerSandbox(ip="127.0.0.1", container_name="sandbox-1")
    sandbox.client = fake_client

    result = asyncio.run(sandbox.get_searxng_status())

    assert fake_client.last_get_url == "http://127.0.0.1:8081/api/searxng/status"
    assert result.success is True
    assert result.data == {"available": True}


def test_search_searxng_should_post_search_request_to_sandbox() -> None:
    fake_client = _FakeAsyncClient()
    sandbox = docker_sandbox.DockerSandbox(ip="127.0.0.1", container_name="sandbox-1")
    sandbox.client = fake_client

    result = asyncio.run(sandbox.search(
        query="openai",
        language="zh-CN",
        page=2,
        safesearch=1,
    ))

    assert fake_client.last_post_url == "http://127.0.0.1:8081/api/searxng/search"
    assert fake_client.last_post_json == {
        "query": "openai",
        "categories": None,
        "engines": None,
        "language": "zh-CN",
        "page": 2,
        "time_range": None,
        "safesearch": 1,
    }
    assert result.success is True
    assert result.data == {"query": "openai", "results": []}


def test_fetch_searxng_page_should_post_fetch_request_to_sandbox() -> None:
    fake_client = _FakeAsyncClient()
    sandbox = docker_sandbox.DockerSandbox(ip="127.0.0.1", container_name="sandbox-1")
    sandbox.client = fake_client

    result = asyncio.run(sandbox.fetch_searxng_page(
        url="https://example.com/article",
        max_chars=5000,
    ))

    assert fake_client.last_post_url == "http://127.0.0.1:8081/api/searxng/fetch-page"
    assert fake_client.last_post_json == {
        "url": "https://example.com/article",
        "max_chars": 5000,
    }
    assert result.success is True
    assert result.data == {"query": "openai", "results": []}
