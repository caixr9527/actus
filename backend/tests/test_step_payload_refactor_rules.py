import importlib.util
from pathlib import Path

from app.domain.models import build_step_objective_key, build_step_objective_source
from app.infrastructure.runtime.langgraph.graphs.planner_react.parsers import (
    build_step_from_payload,
)


def _load_refactor_migration_module():
    migration_path = (
        Path(__file__).resolve().parents[1]
        / "alembic"
        / "versions"
        / "f0a1b2c3d4e5_refactor_workflow_run_steps_outcome.py"
    )
    spec = importlib.util.spec_from_file_location(
        "workflow_run_steps_outcome_refactor_migration",
        migration_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_step_from_payload_should_fallback_description_to_title() -> None:
    step = build_step_from_payload({"title": "整理需求"}, fallback_index=0)

    assert step.title == "整理需求"
    assert step.description == "整理需求"
    assert step.success_criteria == ["整理需求"]
    assert step.objective_key == build_step_objective_key("整理需求", "整理需求")


def test_build_step_from_payload_should_ignore_legacy_execution_fields() -> None:
    step = build_step_from_payload(
        {
            "description": "生成最终输出",
            "execution_template": "根据{{city}}整理{{days}}天行程",
            "required_slots": ["city", "days"],
            "execution_slots": {"city": "上海", "days": 3, "": "invalid"},
        },
        fallback_index=0,
    )

    assert step.description == "生成最终输出"
    assert step.success_criteria == ["生成最终输出"]


def test_build_step_from_payload_should_filter_low_value_success_criteria() -> None:
    step = build_step_from_payload(
        {
            "description": "根据检索结果生成候选清单",
            "success_criteria": ["完成任务", "继续处理", "至少给出3个候选并附来源链接"],
        },
        fallback_index=0,
    )

    assert step.success_criteria == ["至少给出3个候选并附来源链接"]


def test_build_step_from_payload_should_normalize_task_mode_hint() -> None:
    step = build_step_from_payload(
        {
            "description": "等待用户选择课程",
            "task_mode_hint": " HUMAN_WAIT ",
        },
        fallback_index=0,
    )

    assert step.task_mode_hint == "human_wait"
    assert step.output_mode == "none"
    assert step.artifact_policy == "forbid_file_output"
    assert step.delivery_role == "none"


def test_build_step_from_payload_should_default_research_step_to_forbid_file_output() -> None:
    step = build_step_from_payload(
        {
            "description": "访问搜索结果中的多个页面，提取至少三门课程的名称，并将这些名称保存到临时文件中以备后续使用。",
            "task_mode_hint": "research",
        },
        fallback_index=0,
        user_message="先简单从慕课网上找三门关于AI Agent的课程名称",
    )

    assert step.description == "访问搜索结果中的多个页面，提取至少三门课程的名称，并将这些名称保存到临时文件中以备后续使用。"
    assert step.output_mode == "none"
    assert step.artifact_policy == "forbid_file_output"
    assert step.delivery_role == "none"


def test_build_step_from_payload_should_keep_file_output_when_user_explicitly_requests_it() -> None:
    step = build_step_from_payload(
        {
            "description": "搜索网页并把结果保存到临时文件中。",
            "task_mode_hint": "research",
            "output_mode": "file",
            "artifact_policy": "require_file_output",
        },
        fallback_index=0,
        user_message="搜索网页后把结果保存到文件里",
    )

    assert step.description == "搜索网页并把结果保存到临时文件中。"
    assert step.output_mode == "file"
    assert step.artifact_policy == "require_file_output"
    assert step.delivery_role == "none"


def test_build_step_from_payload_should_keep_final_delivery_role_for_inline_step() -> None:
    step = build_step_from_payload(
        {
            "description": "基于前序检索结果整理最终旅游攻略",
            "task_mode_hint": "general",
            "output_mode": "inline",
            "artifact_policy": "default",
            "delivery_role": "final",
        },
        fallback_index=0,
    )

    assert step.output_mode == "inline"
    assert step.artifact_policy == "default"
    assert step.delivery_role == "final"
    assert step.delivery_context_state == "ready"


def test_build_step_from_payload_should_default_inline_step_to_non_delivery_role() -> None:
    step = build_step_from_payload(
        {
            "description": "展示候选城市供用户选择",
            "task_mode_hint": "general",
            "output_mode": "inline",
            "artifact_policy": "default",
        },
        fallback_index=0,
    )

    assert step.delivery_role == "none"
    assert step.delivery_context_state == "none"


def test_build_step_from_payload_should_keep_intermediate_delivery_role_for_explicit_preview_request() -> None:
    step = build_step_from_payload(
        {
            "description": "展示候选城市供用户选择",
            "task_mode_hint": "general",
            "output_mode": "inline",
            "artifact_policy": "default",
            "delivery_role": "intermediate",
        },
        fallback_index=0,
        user_message="先给我看一个候选方案草稿，我确认后再继续。",
    )

    assert step.delivery_role == "intermediate"
    assert step.delivery_context_state == "none"


def test_build_step_from_payload_should_default_non_general_final_inline_step_to_needs_preparation() -> None:
    step = build_step_from_payload(
        {
            "description": "读取课程详情页并直接作为最终交付",
            "task_mode_hint": "web_reading",
            "output_mode": "inline",
            "artifact_policy": "default",
            "delivery_role": "final",
        },
        fallback_index=0,
    )

    assert step.task_mode_hint == "web_reading"
    assert step.output_mode == "inline"
    assert step.artifact_policy == "default"
    assert step.delivery_role == "final"
    assert step.delivery_context_state == "needs_preparation"


def test_build_step_from_payload_should_mark_non_general_final_inline_step_as_needs_preparation() -> None:
    step = build_step_from_payload(
        {
            "description": "继续读取页面并在同一步中输出最终课程详情",
            "task_mode_hint": "web_reading",
            "output_mode": "inline",
            "artifact_policy": "default",
            "delivery_role": "final",
            "delivery_context_state": "needs_preparation",
        },
        fallback_index=0,
    )

    assert step.task_mode_hint == "web_reading"
    assert step.output_mode == "inline"
    assert step.artifact_policy == "default"
    assert step.delivery_role == "final"
    assert step.delivery_context_state == "needs_preparation"


def test_migration_should_normalize_legacy_step_payload_into_outcome() -> None:
    migration = _load_refactor_migration_module()

    normalized_payload = migration._normalize_step_payload(
        {
            "id": "step-1",
            "description": "执行步骤",
            "status": "completed",
            "result": "执行完成",
            "success": True,
            "attachments": ["file-1"],
        }
    )

    assert normalized_payload["title"] == "执行步骤"
    assert normalized_payload["description"] == "执行步骤"
    assert normalized_payload["success_criteria"] == ["执行步骤"]
    assert normalized_payload["outcome"] == {
        "done": True,
        "summary": "执行完成",
        "produced_artifacts": ["file-1"],
        "blockers": [],
        "facts_learned": [],
        "open_questions": [],
        "next_hint": None,
        "reused_from_run_id": None,
        "reused_from_step_id": None,
    }


def test_migration_objective_key_rule_should_match_runtime_rule() -> None:
    migration = _load_refactor_migration_module()

    cases = [
        ("整理需求", "整理需求"),
        ("整理需求", "明确交付范围"),
        ("", "明确交付范围"),
        ("", ""),
    ]

    for title, description in cases:
        assert migration._build_objective_key_source(title, description) == build_step_objective_source(
            title,
            description,
        )
        assert migration._build_objective_key(title, description) == build_step_objective_key(
            title,
            description,
        )
