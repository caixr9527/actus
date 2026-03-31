from app.domain.models import validate_wait_resume_value


def test_validate_wait_resume_value_should_accept_input_text_with_matching_response_key() -> None:
    assert validate_wait_resume_value(
        {
            "kind": "input_text",
            "prompt": "请输入目标网址",
            "response_key": "website",
            "allow_empty": False,
        },
        {"website": "https://example.com"},
    ) is True


def test_validate_wait_resume_value_should_reject_input_text_with_wrong_response_key() -> None:
    assert validate_wait_resume_value(
        {
            "kind": "input_text",
            "prompt": "请输入目标网址",
            "response_key": "website",
            "allow_empty": False,
        },
        {"message": "https://example.com"},
    ) is False


def test_validate_wait_resume_value_should_reject_input_text_with_empty_text_when_not_allowed() -> None:
    assert validate_wait_resume_value(
        {
            "kind": "input_text",
            "prompt": "请输入目标网址",
            "response_key": "website",
            "allow_empty": False,
        },
        {"website": "   "},
    ) is False


def test_validate_wait_resume_value_should_accept_confirm_only_for_declared_resume_values() -> None:
    payload = {
        "kind": "confirm",
        "prompt": "确认继续执行？",
        "confirm_resume_value": {"action": "continue"},
        "cancel_resume_value": {"action": "cancel"},
    }

    assert validate_wait_resume_value(payload, {"action": "continue"}) is True
    assert validate_wait_resume_value(payload, {"approved": True}) is False


def test_validate_wait_resume_value_should_accept_select_only_for_declared_options() -> None:
    payload = {
        "kind": "select",
        "prompt": "请选择执行方式",
        "options": [
            {"label": "方案A", "resume_value": "a"},
            {"label": "方案B", "resume_value": {"plan": "b"}},
        ],
    }

    assert validate_wait_resume_value(payload, {"plan": "b"}) is True
    assert validate_wait_resume_value(payload, "c") is False

