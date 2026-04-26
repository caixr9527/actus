import asyncio
from typing import Any, Dict

import pytest
from pydantic import BaseModel, Field, ValidationError

from app.domain.services.runtime.skill_graph_registry import (
    SkillDefinition,
    SkillGraphRegistry,
    SkillRuntimePolicy,
)
from app.infrastructure.runtime.langgraph.graphs.skills.registry import (
    build_default_skill_graph_registry,
)


class _EchoSkillInput(BaseModel):
    message: str


class _EchoSkillOutput(BaseModel):
    success: bool = True
    result: str
    attachments: list[str] = Field(default_factory=list)


class _EchoSkillGraph:
    def __init__(self, suffix: str) -> None:
        self._suffix = suffix

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "success": True,
            "result": f"{state['message']}{self._suffix}",
            "attachments": [],
        }


def _build_echo_skill_definition(*, version: str, suffix: str) -> SkillDefinition:
    return SkillDefinition(
        skill_id="demo.echo",
        version=version,
        description="回声 Skill",
        input_schema=_EchoSkillInput,
        output_schema=_EchoSkillOutput,
        runtime_policy=SkillRuntimePolicy(
            capability_dependencies=(),
            tool_strategy="none",
            max_iterations=1,
        ),
        graph_factory=lambda _llm: _EchoSkillGraph(suffix=suffix),
    )


def test_skill_graph_registry_should_resolve_latest_version_by_default() -> None:
    registry = SkillGraphRegistry()
    registry.register(_build_echo_skill_definition(version="1.0.0", suffix="-v1"))
    registry.register(_build_echo_skill_definition(version="1.2.0", suffix="-v2"))

    resolved = registry.resolve(skill_id="demo.echo")
    assert resolved.version == "1.2.0"
    assert registry.list_versions("demo.echo") == ["1.0.0", "1.2.0"]


def test_skill_graph_registry_should_reject_duplicate_version() -> None:
    registry = SkillGraphRegistry()
    registry.register(_build_echo_skill_definition(version="1.0.0", suffix="-v1"))

    with pytest.raises(ValueError):
        registry.register(_build_echo_skill_definition(version="1.0.0", suffix="-dup"))


def test_skill_graph_runtime_should_validate_schema_and_execute_graph() -> None:
    registry = SkillGraphRegistry()
    registry.register(_build_echo_skill_definition(version="1.0.0", suffix="-ok"))
    runtime = registry.create_runtime(llm=object())

    async def _execute():
        return await runtime.execute_skill(
            skill_id="demo.echo",
            payload={"message": "hello"},
        )

    output = asyncio.run(_execute())
    assert output.result == "hello-ok"
    assert output.success is True

    async def _execute_invalid_input():
        return await runtime.execute_skill(
            skill_id="demo.echo",
            payload={"unexpected": "value"},
        )

    with pytest.raises(ValidationError):
        asyncio.run(_execute_invalid_input())


def test_default_skill_registry_should_run_planner_execute_step_subgraph() -> None:
    class _StepLLM:
        async def invoke(self, messages, tools=None, response_format=None, tool_choice=None):
            # 断言 skill 输入契约，不绑定具体 prompt 文案，避免模板迭代导致无意义回归。
            content = str(messages[0]["content"])
            assert "请继续执行" in content
            assert "执行第一步" in content
            return {
                "role": "assistant",
                "content": "{\"success\": true, \"result\": \"步骤执行完成\", \"attachments\": []}",
            }

    registry = build_default_skill_graph_registry()
    runtime = registry.create_runtime(llm=_StepLLM())

    async def _execute():
        return await runtime.execute_skill(
            skill_id="planner_react.execute_step",
            payload={
                "session_id": "session-1",
                "user_message": "请继续执行",
                "step_description": "执行第一步",
                "language": "zh",
                "attachments": [],
            },
        )

    output = asyncio.run(_execute())
    assert output.success is True
    assert output.result == "步骤执行完成"
