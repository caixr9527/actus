import asyncio

from app.domain.models import (
    BrowserPageStructuredResult,
    BrowserPageType,
    ToolResult,
)
from app.domain.services.workspace_runtime.capabilities import (
    WorkspaceBrowserCapability,
    WorkspaceFileCapability,
    WorkspaceResearchCapability,
)


class _FakeWorkspaceRuntimeService:
    def __init__(self) -> None:
        self.browser_snapshots: list[dict] = []
        self.search_results: list[dict] = []
        self.fetched_pages: list[dict] = []
        self.file_tree_summaries: list[str] = []
        self.artifacts: list[dict] = []

    async def record_browser_snapshot(self, *, snapshot: dict):
        self.browser_snapshots.append(dict(snapshot))

    async def record_search_results(self, *, query: str, candidate_links: list[dict]):
        self.search_results.append(
            {
                "query": query,
                "candidate_links": list(candidate_links),
            }
        )

    async def record_fetched_page_summary(self, *, page_summary: dict):
        self.fetched_pages.append(dict(page_summary))

    async def record_file_tree_summary(self, *, summary_text: str):
        self.file_tree_summaries.append(summary_text)

    async def upsert_artifact(self, **kwargs):
        self.artifacts.append(dict(kwargs))
        return kwargs


class _FakeBrowser:
    async def read_current_page_structured(self):
        return ToolResult(
            success=True,
            data=BrowserPageStructuredResult(
                url="https://example.com/docs/runtime",
                title="Runtime Docs",
                page_type=BrowserPageType.DOCUMENT,
                content_summary="runtime summary",
            ),
        )


class _FakeSandbox:
    async def search_searxng(self, **kwargs):
        return ToolResult(
            success=True,
            data={
                "query": kwargs["query"],
                "results": [
                    {
                        "url": "https://example.com/result-1",
                        "title": "Result 1",
                        "content": "snippet 1",
                    }
                ],
            },
        )

    async def fetch_searxng_page(self, **kwargs):
        return ToolResult(
            success=True,
            data={
                "url": kwargs["url"],
                "final_url": kwargs["url"],
                "title": "Fetched Page",
                "content": "full content",
                "excerpt": "excerpt content",
                "status_code": 200,
                "content_type": "text/html",
                "content_length": 12,
                "truncated": False,
            },
        )

    async def write_file(self, **kwargs):
        return ToolResult(success=True, data={"filepath": kwargs["file_path"]})

    async def find_files(self, **kwargs):
        return ToolResult(
            success=True,
            data={
                "dir_path": kwargs["dir_path"],
                "files": ["/workspace/project/a.py", "/workspace/project/b.py"],
            },
        )

    async def list_files(self, dir_path: str):
        return await self.find_files(dir_path=dir_path, glob_pattern="*")


def test_browser_capability_should_record_workspace_snapshot() -> None:
    workspace_runtime_service = _FakeWorkspaceRuntimeService()
    tool = WorkspaceBrowserCapability(
        browser=_FakeBrowser(),
        workspace_runtime_service=workspace_runtime_service,
    )

    result = asyncio.run(tool.browser_read_current_page_structured())

    assert result.success is True
    assert workspace_runtime_service.browser_snapshots == [
        {
            "url": "https://example.com/docs/runtime",
            "title": "Runtime Docs",
            "page_type": "document",
            "main_content_summary": "runtime summary",
            "actionable_elements": [],
        }
    ]


def test_search_tool_should_record_workspace_observations() -> None:
    workspace_runtime_service = _FakeWorkspaceRuntimeService()
    tool = WorkspaceResearchCapability(
        sandbox=_FakeSandbox(),
        workspace_runtime_service=workspace_runtime_service,
    )

    search_result = asyncio.run(tool.search_web(query="runtime workspace"))
    fetch_result = asyncio.run(tool.fetch_page(url="https://example.com/result-1"))

    assert search_result.success is True
    assert fetch_result.success is True
    assert workspace_runtime_service.search_results == [
        {
            "query": "runtime workspace",
            "candidate_links": [
                {
                    "title": "Result 1",
                    "url": "https://example.com/result-1",
                    "snippet": "snippet 1",
                }
            ],
        }
    ]
    assert workspace_runtime_service.fetched_pages == [
        {
            "title": "Fetched Page",
            "url": "https://example.com/result-1",
            "excerpt": "excerpt content",
        }
    ]


def test_file_capability_should_record_artifact_and_tree_summary() -> None:
    workspace_runtime_service = _FakeWorkspaceRuntimeService()
    tool = WorkspaceFileCapability(
        sandbox=_FakeSandbox(),
        workspace_runtime_service=workspace_runtime_service,
    )

    write_result = asyncio.run(tool.write_file(filepath="/workspace/project/report.md", content="# Report"))
    find_result = asyncio.run(tool.find_files(dir_path="/workspace/project", glob_pattern="*.py"))

    assert write_result.success is True
    assert find_result.success is True
    assert workspace_runtime_service.artifacts == [
        {
            "path": "/workspace/project/report.md",
            "artifact_type": "file",
            "summary": "通过 write_file 更新文件: /workspace/project/report.md",
            "source_capability": "write_file",
            "metadata": {
                "append": False,
                "leading_newline": False,
                "trailing_newline": False,
                "sudo": False,
            },
        }
    ]
    assert workspace_runtime_service.file_tree_summaries == [
        "目录 /workspace/project 共 2 项"
    ]
