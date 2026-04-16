#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ReplanMergeEngine 单元测试。"""

import re

from app.domain.models import ExecutionStatus, Plan, Step
from app.infrastructure.runtime.langgraph.graphs.planner_react.replan import ReplanMergeEngine


def _build_engine() -> ReplanMergeEngine:
    return ReplanMergeEngine(
        replan_meta_validation_step_pattern=re.compile(r"工具可用性|测试工具", re.IGNORECASE),
        replan_meta_validation_allow_pattern=re.compile(r"验证工具|测试工具", re.IGNORECASE),
        replan_meta_validation_deny_pattern=re.compile(r"不要测试工具|不需要验证工具", re.IGNORECASE),
    )


def test_merge_engine_should_replace_pending_with_semantic_cover_steps() -> None:
    engine = _build_engine()
    plan = Plan(
        title="replan",
        goal="收集目录信息",
        language="zh",
        steps=[
            Step(
                id="done-1",
                title="准备环境",
                description="准备环境",
                success_criteria=["准备环境"],
                status=ExecutionStatus.COMPLETED,
            ),
            Step(
                id="pending-old-1",
                title="读取目录列表",
                description="读取目录并输出",
                success_criteria=["读取目录"],
                status=ExecutionStatus.PENDING,
            ),
            Step(
                id="pending-old-2",
                title="读取文件内容",
                description="读取 hello.txt",
                success_criteria=["读取文件内容"],
                status=ExecutionStatus.PENDING,
            ),
        ],
    )
    new_steps = [
        Step(
            id="new-1",
            title="读取目录列表",
            description="读取目录并输出",
            success_criteria=["读取目录"],
            status=ExecutionStatus.PENDING,
        ),
        Step(
            id="new-2",
            title="读取 hello.txt 内容",
            description="读取 hello.txt 并确认",
            success_criteria=["读取文件内容"],
            status=ExecutionStatus.PENDING,
        ),
    ]

    merged_steps, merge_mode = engine.merge_replanned_steps_into_plan(plan, new_steps)

    assert merge_mode == "replace_remaining_pending_steps"
    assert len(merged_steps) == 3
    assert merged_steps[0].id == "done-1"
    assert merged_steps[1].id == "new-1"
    assert merged_steps[2].id == "new-2"


def test_merge_engine_should_dedupe_semantic_duplicates_in_new_steps() -> None:
    engine = _build_engine()
    plan = Plan(
        title="replan",
        goal="收集目录信息",
        language="zh",
        steps=[],
    )
    new_steps = [
        Step(
            id="new-a",
            title="读取目录",
            description="读取目录并输出",
            success_criteria=["读取目录", "输出结果"],
            status=ExecutionStatus.PENDING,
        ),
        Step(
            id="new-b",
            title="读取目录",
            description="读取目录并输出",
            success_criteria=["输出结果", "读取目录"],
            status=ExecutionStatus.PENDING,
        ),
        Step(
            id="new-c",
            title="读取文件",
            description="读取 hello.txt",
            success_criteria=["读取文件内容"],
            status=ExecutionStatus.PENDING,
        ),
    ]

    merged_steps, merge_mode = engine.merge_replanned_steps_into_plan(plan, new_steps)

    assert merge_mode == "append_after_completed_batch"
    assert len(merged_steps) == 2
    assert merged_steps[0].id == "new-a"
    assert merged_steps[1].id == "new-c"


def test_filter_replan_drift_steps_should_respect_user_deny_signal() -> None:
    engine = _build_engine()
    steps = [
        Step(
            id="meta-1",
            title="测试工具可用性",
            description="先做工具可用性检查",
            success_criteria=["验证工具"],
            status=ExecutionStatus.PENDING,
        ),
        Step(
            id="real-1",
            title="读取目录",
            description="读取目录并输出",
            success_criteria=["读取目录"],
            status=ExecutionStatus.PENDING,
        ),
    ]

    filtered_steps, dropped_count = engine.filter_replan_drift_steps(
        steps,
        user_message="继续读取目录，不要测试工具",
    )

    assert dropped_count == 1
    assert len(filtered_steps) == 1
    assert filtered_steps[0].id == "real-1"


def test_filter_replan_drift_steps_should_allow_meta_when_user_explicitly_requests_validation() -> None:
    engine = _build_engine()
    steps = [
        Step(
            id="meta-1",
            title="测试工具可用性",
            description="先做工具可用性检查",
            success_criteria=["验证工具"],
            status=ExecutionStatus.PENDING,
        ),
    ]

    filtered_steps, dropped_count = engine.filter_replan_drift_steps(
        steps,
        user_message="请验证工具可用性后再继续",
    )

    assert dropped_count == 0
    assert len(filtered_steps) == 1
    assert filtered_steps[0].id == "meta-1"

