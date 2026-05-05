import asyncio

from app.domain.models import StepTaskModeHint, Workspace
from app.domain.services.prompts import SYSTEM_PROMPT
from app.domain.services.runtime.contracts.sandbox_capability_profile_contract import (
    SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY,
)
from app.domain.services.workspace_runtime import WorkspaceEnvironmentSnapshot
from app.domain.services.workspace_runtime.context import RuntimeContextService
from app.infrastructure.runtime.langgraph.graphs.planner_react.nodes.prompt_context_helpers import (
    _append_prompt_context_to_prompt,
)


class _WorkspaceRuntimeService:
    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    async def build_environment_snapshot(self):
        return WorkspaceEnvironmentSnapshot(workspace=self._workspace, artifacts=[])


def _build_context_service() -> RuntimeContextService:
    workspace = Workspace(
        id="workspace-1",
        session_id="session-1",
        user_id="user-1",
        environment_summary={
            SANDBOX_CAPABILITY_PROFILE_ENVIRONMENT_KEY: {
                "prompt_summary": {
                    "health_status": "available",
                    "cwd": "/workspace",
                    "available_runtime": {"python": "3.12.1"},
                    "available_tools": ["shell", "file"],
                    "sandbox_profile_stale": False,
                }
            }
        },
    )
    return RuntimeContextService(
        workspace_runtime_service=_WorkspaceRuntimeService(workspace),
    )


def test_runtime_context_should_include_sandbox_profile_for_planner_execute_replan() -> None:
    context_service = _build_context_service()
    state = {"user_message": "处理项目代码"}

    planner_packet = asyncio.run(
        context_service.build_packet_async(
            stage="planner",
            state=state,
            task_mode=StepTaskModeHint.GENERAL.value,
        )
    )
    execute_packet = asyncio.run(
        context_service.build_packet_async(
            stage="execute",
            state=state,
            task_mode=StepTaskModeHint.CODING.value,
        )
    )
    replan_packet = asyncio.run(
        context_service.build_packet_async(
            stage="replan",
            state=state,
            task_mode=StepTaskModeHint.GENERAL.value,
        )
    )

    assert planner_packet["environment_digest"]["sandbox_capability_profile"]["cwd"] == "/workspace"
    assert execute_packet["environment_digest"]["sandbox_capability_profile"]["available_tools"] == ["shell", "file"]
    assert replan_packet["environment_digest"]["sandbox_capability_profile"]["health_status"] == "available"


def test_runtime_context_should_not_include_sandbox_profile_for_summary_or_direct_answer_by_default() -> None:
    context_service = _build_context_service()
    state = {"user_message": "处理项目代码"}

    summary_packet = asyncio.run(
        context_service.build_packet_async(
            stage="summary",
            state=state,
            task_mode=StepTaskModeHint.GENERAL.value,
        )
    )
    direct_answer_packet = asyncio.run(
        context_service.build_packet_async(
            stage="direct_answer",
            state=state,
            task_mode=StepTaskModeHint.GENERAL.value,
        )
    )

    assert "environment_digest" not in summary_packet
    assert "environment_digest" not in direct_answer_packet


def test_runtime_context_should_include_sandbox_profile_for_summary_when_explicitly_requested() -> None:
    context_service = _build_context_service()

    packet = asyncio.run(
        context_service.build_packet_async(
            stage="summary",
            state={"user_message": "处理项目代码"},
            task_mode=StepTaskModeHint.GENERAL.value,
            include_sandbox_profile_for_summary=True,
        )
    )

    assert packet["environment_digest"]["sandbox_capability_profile"]["cwd"] == "/workspace"


def test_system_prompt_should_not_define_static_sandbox_environment_facts() -> None:
    forbidden_static_facts = [
        "<sandbox_environment>",
        "Ubuntu 22.04",
        "Python 3.10.12",
        "Node.js 20.18.0",
        "具备互联网访问权限",
        "sudo",
        "/home/ubuntu",
        "基础计算器",
        "命令: bc",
    ]

    for fact in forbidden_static_facts:
        assert fact not in SYSTEM_PROMPT


def test_planner_execute_replan_prompt_should_use_dynamic_sandbox_profile_context() -> None:
    context_service = _build_context_service()
    state = {"user_message": "处理项目代码"}

    for stage, task_mode in [
        ("planner", StepTaskModeHint.GENERAL.value),
        ("execute", StepTaskModeHint.CODING.value),
        ("replan", StepTaskModeHint.GENERAL.value),
    ]:
        packet = asyncio.run(
            context_service.build_packet_async(
                stage=stage,
                state=state,
                task_mode=task_mode,
            )
        )
        prompt = _append_prompt_context_to_prompt("用户任务", packet)

        assert "已知上下文" in prompt
        assert '"sandbox_capability_profile"' in prompt
        assert '"python": "3.12.1"' in prompt
        assert '"available_tools": [' in prompt
        assert '"shell"' in prompt
        assert '"file"' in prompt
