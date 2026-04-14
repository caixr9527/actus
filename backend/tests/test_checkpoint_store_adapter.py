import asyncio
from typing import Optional

from app.domain.models import Session, WorkflowRun, Workspace
from app.infrastructure.runtime.langgraph.engine.checkpoint_store_adapter import CheckpointStoreAdapter


class _SessionRepo:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def get_by_id(self, session_id: str, user_id=None):
        if session_id != self._session.id:
            return None
        return self._session


class _WorkflowRunRepo:
    def __init__(self, run: WorkflowRun) -> None:
        self._run = run
        self.updated_refs: list[tuple[str, str, str]] = []

    async def get_by_id(self, run_id: str):
        if run_id != self._run.id:
            return None
        return self._run

    async def update_checkpoint_ref(
            self,
            run_id: str,
            checkpoint_namespace: str,
            checkpoint_id: str,
    ) -> None:
        if run_id != self._run.id:
            raise ValueError("run not found")
        self.updated_refs.append((run_id, checkpoint_namespace, checkpoint_id))
        self._run.checkpoint_namespace = checkpoint_namespace
        self._run.checkpoint_id = checkpoint_id


class _WorkspaceRepo:
    def __init__(self, workspace: Optional[Workspace]) -> None:
        self._workspace = workspace

    async def get_by_id(self, workspace_id: str):
        if self._workspace is None or workspace_id != self._workspace.id:
            return None
        return self._workspace

    async def get_by_session_id(self, session_id: str):
        if self._workspace is None or session_id != self._workspace.session_id:
            return None
        return self._workspace


class _UoW:
    def __init__(
            self,
            session_repo: _SessionRepo,
            workflow_run_repo: _WorkflowRunRepo,
            workspace_repo: Optional[_WorkspaceRepo] = None,
    ) -> None:
        self.session = session_repo
        self.workflow_run = workflow_run_repo
        self.workspace = workspace_repo or _WorkspaceRepo(None)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _CheckpointTuple:
    def __init__(self, config):
        self.config = config


class _FakeCheckpointer:
    def __init__(self, checkpoint_id: str) -> None:
        self._checkpoint_id = checkpoint_id
        self.lookup_configs = []

    async def aget_tuple(self, config):
        self.lookup_configs.append(config)
        return _CheckpointTuple(
            {
                "configurable": {
                    "thread_id": "thread-1",
                    "checkpoint_ns": "",
                    "checkpoint_id": self._checkpoint_id,
                }
            }
        )


def test_checkpoint_store_adapter_should_resolve_invoke_config_from_workflow_run() -> None:
    session = Session(id="session-1", workspace_id="workspace-1", current_run_id="run-legacy")
    workspace = Workspace(id="workspace-1", session_id="session-1", current_run_id="run-1")
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        thread_id="thread-1",
        checkpoint_namespace="",
        checkpoint_id="cp-old",
    )
    adapter = CheckpointStoreAdapter(
        session_id="session-1",
        uow_factory=lambda: _UoW(_SessionRepo(session), _WorkflowRunRepo(run), _WorkspaceRepo(workspace)),
    )

    config, run_id = asyncio.run(adapter.resolve_invoke_config())

    assert run_id == "run-1"
    assert config == {
        "configurable": {
            "thread_id": "thread-1",
            "checkpoint_ns": "",
            "checkpoint_id": "cp-old",
        }
    }


def test_checkpoint_store_adapter_should_sync_latest_checkpoint_ref() -> None:
    session = Session(id="session-1", workspace_id="workspace-1", current_run_id="run-legacy")
    workspace = Workspace(id="workspace-1", session_id="session-1", current_run_id="run-1")
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        thread_id="thread-1",
        checkpoint_namespace="",
        checkpoint_id="cp-old",
    )
    workflow_run_repo = _WorkflowRunRepo(run)
    adapter = CheckpointStoreAdapter(
        session_id="session-1",
        uow_factory=lambda: _UoW(_SessionRepo(session), workflow_run_repo, _WorkspaceRepo(workspace)),
    )
    checkpointer = _FakeCheckpointer(checkpoint_id="cp-new")

    asyncio.run(
        adapter.sync_latest_checkpoint_ref(
            run_id="run-1",
            checkpointer=checkpointer,
            invoke_config={"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}},
        )
    )

    assert checkpointer.lookup_configs == [
        {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
    ]
    assert workflow_run_repo.updated_refs == [("run-1", "", "cp-new")]
    assert run.checkpoint_id == "cp-new"


def test_checkpoint_store_adapter_should_not_fallback_to_session_current_run_id() -> None:
    session = Session(id="session-1", workspace_id="workspace-1", current_run_id="run-legacy")
    workspace = Workspace(id="workspace-1", session_id="session-1", current_run_id=None)
    run = WorkflowRun(
        id="run-1",
        session_id="session-1",
        thread_id="thread-1",
        checkpoint_namespace="",
        checkpoint_id="cp-old",
    )
    adapter = CheckpointStoreAdapter(
        session_id="session-1",
        uow_factory=lambda: _UoW(_SessionRepo(session), _WorkflowRunRepo(run), _WorkspaceRepo(workspace)),
    )

    config, run_id = asyncio.run(adapter.resolve_invoke_config())

    assert run_id is None
    assert config == {
        "configurable": {
            "thread_id": "session-1",
            "checkpoint_ns": "",
        }
    }
