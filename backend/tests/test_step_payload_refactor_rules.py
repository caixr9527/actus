import importlib.util
from pathlib import Path

from app.domain.models import build_step_objective_key, build_step_objective_source
from app.domain.services.prompts import CREATE_PLAN_PROMPT, EXECUTION_PROMPT, UPDATE_PLAN_PROMPT
from app.domain.services.prompts.en.planner import (
    CREATE_PLAN_PROMPT as EN_CREATE_PLAN_PROMPT,
    UPDATE_PLAN_PROMPT as EN_UPDATE_PLAN_PROMPT,
)
from app.domain.services.prompts.en.react import (
    EXECUTION_PROMPT as EN_EXECUTION_PROMPT,
    SUMMARIZE_PROMPT as EN_SUMMARIZE_PROMPT,
)
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

def test_build_step_from_payload_should_keep_general_none_output_mode() -> None:
    step = build_step_from_payload(
        {
            "description": "基于前序检索结果提取可供后续总结使用的事实",
            "task_mode_hint": "general",
            "output_mode": "none",
            "artifact_policy": "default",
        },
        fallback_index=0,
    )

    assert step.output_mode == "none"
    assert step.artifact_policy == "default"

def test_build_step_from_payload_should_keep_preview_request_as_execution_step() -> None:
    step = build_step_from_payload(
        {
            "description": "收集候选城市事实供后续总结使用",
            "task_mode_hint": "general",
            "output_mode": "none",
            "artifact_policy": "default",
        },
        fallback_index=0,
    )

    assert step.output_mode == "none"
    assert step.artifact_policy == "default"

def test_build_step_from_payload_should_not_expose_delivery_fields() -> None:
    step = build_step_from_payload(
        {
            "description": "收集候选城市事实供后续总结使用",
            "task_mode_hint": "general",
            "output_mode": "none",
            "artifact_policy": "default",
        },
        fallback_index=0,
        user_message="先给我看一个候选方案草稿，我确认后再继续。",
    )

    assert step.output_mode == "none"
    assert step.artifact_policy == "default"
    assert not hasattr(step, "delivery_role")
    assert not hasattr(step, "delivery_context_state")

def test_build_step_from_payload_should_force_web_reading_without_file_request_to_none_output() -> None:
    step = build_step_from_payload(
        {
            "description": "读取课程详情页并提取课程名称、讲师和适用人群",
            "task_mode_hint": "web_reading",
            "output_mode": "none",
            "artifact_policy": "default",
        },
        fallback_index=0,
    )

    assert step.task_mode_hint == "web_reading"
    assert step.output_mode == "none"
    assert step.artifact_policy == "forbid_file_output"


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


def test_prompts_should_require_single_topic_natural_language_search() -> None:
    for prompt in (CREATE_PLAN_PROMPT, UPDATE_PLAN_PROMPT, EXECUTION_PROMPT):
        assert "单主题自然语言" in prompt
        assert "关键词堆叠" in prompt


def test_en_prompts_should_require_single_topic_natural_language_search() -> None:
    for prompt in (EN_CREATE_PLAN_PROMPT, EN_UPDATE_PLAN_PROMPT, EN_EXECUTION_PROMPT):
        assert "single-topic natural-language" in prompt
        assert "keyword stacking" in prompt


def test_en_summarize_prompt_should_use_summary_only_final_answer_contract() -> None:
    assert "final_answer_text" in EN_SUMMARIZE_PROMPT
    assert "final delivery payload" not in EN_SUMMARIZE_PROMPT
    assert "heavy delivery" not in EN_SUMMARIZE_PROMPT
