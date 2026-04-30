import pytest

from app.application.errors import BadRequestError, error_keys
from app.application.service.agent_service import AgentService
from app.application.service.document_input_preflight_policy import DocumentInputPreflightPolicy
from app.application.service.runtime_access_control_service import AccessScopeResult
from app.domain.models import ErrorEvent, File, Session, SessionStatus


def test_image_attachment_should_be_rejected_before_message_event() -> None:
    policy = DocumentInputPreflightPolicy()

    with pytest.raises(BadRequestError) as exc:
        policy.validate(
            [
                File(
                    id="file-image",
                    user_id="user-1",
                    filename="photo.png",
                    extension=".png",
                    mime_type="image/png",
                    size=128,
                )
            ],
            user_id="user-1",
            session_id="session-1",
            request_id="request-1",
        )

    assert exc.value.error_key == error_keys.DOCUMENT_INPUT_UNSUPPORTED_MEDIA_IMAGE
    assert exc.value.error_params["reason_code"] == "unsupported_media_image"
    assert exc.value.error_params["file_id"] == "file-image"


def test_preflight_should_reject_too_many_attachments() -> None:
    policy = DocumentInputPreflightPolicy(max_attachments=1)

    with pytest.raises(BadRequestError) as exc:
        policy.validate(
            [
                File(id="file-1", filename="a.txt", extension=".txt", mime_type="text/plain", size=1),
                File(id="file-2", filename="b.txt", extension=".txt", mime_type="text/plain", size=1),
            ],
            user_id="user-1",
            session_id="session-1",
        )

    assert exc.value.error_key == error_keys.DOCUMENT_INPUT_TOO_MANY_ATTACHMENTS
    assert exc.value.error_params["reason_code"] == "too_many_attachments"


def test_preflight_should_accept_document_metadata_without_reading_body() -> None:
    policy = DocumentInputPreflightPolicy()
    files = [
        File(id="file-1", filename="notes.md", extension=".md", mime_type="text/markdown", size=12)
    ]

    assert policy.validate(files, user_id="user-1", session_id="session-1") == files


def test_agent_service_chat_should_reject_image_attachment_before_create_task() -> None:
    class _AccessControl:
        async def resolve_session_scope(self, *, user_id: str, session_id: str):
            return AccessScopeResult(tenant_id=user_id, user_id=user_id, session_id=session_id)

        async def assert_file_access(self, *, user_id: str, file_id: str, action):
            return AccessScopeResult(tenant_id=user_id, user_id=user_id)

    class _Coordinator:
        async def reconcile_current_run(self, session_id: str, *, reason: str):
            return type("Result", (), {"warnings": [], "snapshot_after": None})()

    class _FileRepo:
        async def get_by_id_and_user_id(self, *, file_id: str, user_id: str):
            return File(
                id=file_id,
                user_id=user_id,
                filename="photo.png",
                extension=".png",
                mime_type="image/png",
                size=128,
            )

    class _SessionRepo:
        async def get_by_id(self, session_id: str, user_id: str | None = None):
            return Session(id=session_id, user_id=user_id or "user-1", status=SessionStatus.COMPLETED)

        async def add_event(self, session_id: str, event):
            return None

        async def update_unread_message_count(self, session_id: str, count: int):
            return None

    class _UoW:
        def __init__(self) -> None:
            self.file = _FileRepo()
            self.session = _SessionRepo()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    service = object.__new__(AgentService)
    service._uow_factory = lambda: _UoW()
    service._access_control_service = _AccessControl()
    service._document_input_preflight_policy = DocumentInputPreflightPolicy()
    service._runtime_state_coordinator = _Coordinator()
    service._graph_runtime = type(
        "GraphRuntime",
        (),
        {"get_task": lambda self, session: _async_none()},
    )()
    service._task_cls = object
    create_task_calls = []

    async def _create_task(*args, **kwargs):
        create_task_calls.append((args, kwargs))
        raise AssertionError("_create_task should not be called")

    service._create_task = _create_task

    import asyncio

    async def _collect_first_event():
        async for event in service.chat(
                session_id="session-1",
                user_id="user-1",
                message="读取图片",
                attachments=["file-image"],
        ):
            return event
        return None

    event = asyncio.run(_collect_first_event())

    assert isinstance(event, ErrorEvent)
    assert event.error_key == error_keys.DOCUMENT_INPUT_UNSUPPORTED_MEDIA_IMAGE
    assert event.error_params["reason_code"] == "unsupported_media_image"
    assert create_task_calls == []


async def _async_none():
    return None
