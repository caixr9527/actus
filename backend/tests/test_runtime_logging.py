from app.domain.services.runtime.contracts.runtime_logging import (
    bind_trace_id,
    build_trace_id,
    format_runtime_log,
    reset_trace_id,
)


def test_format_runtime_log_should_include_bound_trace_id() -> None:
    trace_id = build_trace_id("session-1", "run-1")
    token = bind_trace_id(trace_id)
    try:
        rendered = format_runtime_log(
            "开始创建计划",
            state={
                "session_id": "session-1",
                "run_id": "run-1",
            },
            step_id="step-1",
        )
    finally:
        reset_trace_id(token)

    assert f'追踪ID="{trace_id}"' in rendered
    assert '会话ID="session-1"' in rendered
    assert '运行ID="run-1"' in rendered
    assert '步骤ID="step-1"' in rendered
