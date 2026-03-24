import asyncio

from app.application.service.session_service import SessionService
from app.domain.models import Session, WorkflowRun


class _SessionRepo:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def get_by_id(self, session_id: str, user_id: str | None = None):
        if session_id == self._session.id and user_id == self._session.user_id:
            return self._session
        return None


class _WorkflowRunRepo:
    def __init__(self, run: WorkflowRun | None) -> None:
        self._run = run

    async def get_by_id(self, run_id: str):
        if self._run is None:
            return None
        return self._run if self._run.id == run_id else None


class _SessionRuntimeUoW:
    def __init__(self, session: Session, run: WorkflowRun | None) -> None:
        self.session = _SessionRepo(session=session)
        self.workflow_run = _WorkflowRunRepo(run=run)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _DummySandbox:
    @classmethod
    async def get(cls, id: str):
        return None


def test_session_service_get_runtime_extensions_should_return_empty_when_no_current_run() -> None:
    session = Session(id="session-a", user_id="user-a", current_run_id=None)
    service = SessionService(
        uow_factory=lambda: _SessionRuntimeUoW(session=session, run=None),
        sandbox_cls=_DummySandbox,
    )

    run_id, runtime_extensions = asyncio.run(service.get_runtime_extensions("user-a", "session-a"))

    assert run_id is None
    assert runtime_extensions == {}


def test_session_service_get_runtime_extensions_should_extract_input_policy_summary() -> None:
    session = Session(id="session-a", user_id="user-a", current_run_id="run-1")
    run = WorkflowRun(
        id="run-1",
        session_id="session-a",
        runtime_metadata={
            "graph_state_contract": {
                "graph_state": {
                    "input_parts": [
                        {"type": "text", "text": "请读取附件"},
                        {"type": "file_ref", "filepath": "/home/ubuntu/upload/a.md"},
                        {"type": "image", "filepath": "/home/ubuntu/upload/a.png"},
                    ],
                    "metadata": {
                        "input_policy": {
                            "unsupported_parts": [
                                {
                                    "type": "image",
                                    "filepath": "/home/ubuntu/upload/a.png",
                                    "reason": "model_multimodal_disabled",
                                }
                            ]
                        }
                    },
                }
            }
        },
    )
    service = SessionService(
        uow_factory=lambda: _SessionRuntimeUoW(session=session, run=run),
        sandbox_cls=_DummySandbox,
    )

    run_id, runtime_extensions = asyncio.run(service.get_runtime_extensions("user-a", "session-a"))

    assert run_id == "run-1"
    assert runtime_extensions["input_part_summary"]["total"] == 3
    assert runtime_extensions["input_part_summary"]["by_type"] == {
        "text": 1,
        "file_ref": 1,
        "image": 1,
    }
    assert runtime_extensions["unsupported_parts"][0]["reason"] == "model_multimodal_disabled"
    assert runtime_extensions["downgrade_reason"] == "model_multimodal_disabled"
