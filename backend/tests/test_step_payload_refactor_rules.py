import importlib.util
from pathlib import Path

from app.domain.models import build_step_objective_key, build_step_objective_source
from app.infrastructure.runtime.langgraph_graphs.planner_react_langgraph.parsers import (
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
