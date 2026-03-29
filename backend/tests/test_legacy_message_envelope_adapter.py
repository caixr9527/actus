import asyncio
from typing import Any, AsyncGenerator

from app.domain.models import (
    Message,
    Plan,
    Step,
)
from app.domain.services.agents.planner import PlannerAgent
from app.domain.services.agents.react import ReActAgent


async def _drain_events(generator: AsyncGenerator[Any, None]) -> list[Any]:
    events: list[Any] = []
    async for event in generator:
        events.append(event)
    return events


def test_planner_should_use_message_attachments_for_legacy_prompt() -> None:
    planner = object.__new__(PlannerAgent)
    captured_query: dict[str, str] = {}

    async def _fake_invoke(query: str):  # type: ignore[no-untyped-def]
        captured_query["value"] = query
        if False:
            yield None

    planner.invoke = _fake_invoke  # type: ignore[assignment]
    message = Message(
        message="请根据附件生成计划",
        attachments=["/home/ubuntu/upload/spec.md"],
    )

    asyncio.run(_drain_events(planner.create_plan(message)))

    query = captured_query.get("value", "")
    assert "/home/ubuntu/upload/spec.md" in query


def test_react_should_use_message_attachments_for_legacy_prompt() -> None:
    react = object.__new__(ReActAgent)
    react._session_id = "session-1"
    captured_query: dict[str, str] = {}

    async def _fake_invoke(query: str):  # type: ignore[no-untyped-def]
        captured_query["value"] = query
        if False:
            yield None

    react.invoke = _fake_invoke  # type: ignore[assignment]
    message = Message(
        message="请阅读附件并总结",
        attachments=["/home/ubuntu/upload/notes.txt"],
    )
    plan = Plan(language="zh", steps=[Step(id="1", description="阅读附件并输出总结")])
    step = plan.steps[0]

    asyncio.run(_drain_events(react.execute_step(plan=plan, step=step, message=message)))

    query = captured_query.get("value", "")
    assert "/home/ubuntu/upload/notes.txt" in query
