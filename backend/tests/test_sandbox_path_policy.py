import pytest

from app.domain.services.runtime.contracts.sandbox_path_policy import (
    is_allowed_sandbox_read_path,
    normalize_sandbox_path,
)


@pytest.mark.parametrize(
    ("raw_path", "normalized_path"),
    [
        ("/workspace/report.md", "/workspace/report.md"),
        ("/tmp/output.txt", "/tmp/output.txt"),
        ("reports/final.md", "/workspace/reports/final.md"),
        ("./reports/final.md", "/workspace/reports/final.md"),
    ],
)
def test_sandbox_path_policy_should_allow_workspace_tmp_and_relative_paths(
    raw_path,
    normalized_path,
) -> None:
    assert is_allowed_sandbox_read_path(raw_path) is True
    assert normalize_sandbox_path(raw_path) == normalized_path


@pytest.mark.parametrize(
    "raw_path",
    [
        "",
        "../secret.txt",
        "/workspace/../secret.txt",
        "https://example.com/a.txt",
        "file:///etc/passwd",
        "s3://bucket/key",
        "C:\\Users\\secret.txt",
        "/etc/passwd",
    ],
)
def test_sandbox_path_policy_should_reject_external_or_escaping_paths(raw_path) -> None:
    assert is_allowed_sandbox_read_path(raw_path) is False
