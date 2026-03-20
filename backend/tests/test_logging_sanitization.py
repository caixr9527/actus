from app.infrastructure.logging.logging import _mask_sensitive_text


def test_mask_sensitive_text_should_mask_common_sensitive_fields() -> None:
    raw = (
        'password=Password123! token=abc123 refresh_token="rt-1" '
        'authorization=Bearer abc.def.ghi cookie=session=xyz'
    )

    masked = _mask_sensitive_text(raw)

    assert "Password123!" not in masked
    assert "abc123" not in masked
    assert "rt-1" not in masked
    assert "abc.def.ghi" not in masked
    assert "session=xyz" not in masked
    assert "password=***" in masked
    assert "token=***" in masked
    assert 'refresh_token=***' in masked


def test_mask_sensitive_text_should_keep_normal_content() -> None:
    raw = "event=login_failed reason=password_mismatch ip=127.0.0.1"

    masked = _mask_sensitive_text(raw)

    assert masked == raw
