import asyncio

from app.domain.models import (
    Message,
    TitleEvent,
    PlanEvent,
    StepEvent,
    MessageEvent,
    DoneEvent,
    Plan,
    Step,
    ExecutionStatus,
    PlanEventStatus,
    StepEventStatus,
)
from app.infrastructure.runtime.langgraph_run_engine import LangGraphRunEngine


class _FakeGraph:
    async def ainvoke(self, _state, config=None):
        return {
            "emitted_events": [
                TitleEvent(title="POC 标题"),
                PlanEvent(
                    plan=Plan(
                        title="x",
                        goal="",
                        language="zh",
                        steps=[],
                        message="",
                        status=ExecutionStatus.PENDING,
                    ),
                    status=PlanEventStatus.CREATED,
                ),
                StepEvent(
                    step=Step(
                        id="s1",
                        description="step",
                        status=ExecutionStatus.COMPLETED,
                        success=True,
                    ),
                    status=StepEventStatus.COMPLETED,
                ),
                MessageEvent(role="assistant", message="完成"),
                DoneEvent(),
            ]
        }


def test_langgraph_run_engine_yields_emitted_events(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.infrastructure.runtime.langgraph_run_engine.build_planner_react_poc_graph",
        lambda llm: _FakeGraph(),
    )

    engine = LangGraphRunEngine(session_id="session-1", llm=object())

    async def _collect():
        return [event async for event in engine.invoke(Message(message="hello"))]

    events = asyncio.run(_collect())

    assert len(events) == 5
    assert isinstance(events[0], TitleEvent)
    assert isinstance(events[1], PlanEvent)
    assert isinstance(events[2], StepEvent)
    assert isinstance(events[3], MessageEvent)
    assert isinstance(events[4], DoneEvent)
