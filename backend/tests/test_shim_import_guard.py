from __future__ import annotations

import ast
from pathlib import Path


FORBIDDEN_MODULES = {
    "app.interfaces.service_dependencies",
    "app.interfaces.auth_dependencies",
}


def _collect_forbidden_imports(py_file: Path) -> list[str]:
    source = py_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(py_file))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module in FORBIDDEN_MODULES:
                violations.append(f"{py_file}:from {module} import ...")
                continue
            if module == "app.interfaces":
                for alias in node.names:
                    imported_name = alias.name
                    target = f"{module}.{imported_name}"
                    if target in FORBIDDEN_MODULES:
                        violations.append(f"{py_file}:from app.interfaces import {imported_name}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported_name = alias.name
                if imported_name in FORBIDDEN_MODULES:
                    violations.append(f"{py_file}:import {imported_name}")

    return violations


def test_app_layer_should_not_add_new_legacy_shim_imports() -> None:
    app_root = Path(__file__).resolve().parents[1] / "app"
    shim_files = {
        app_root / "interfaces" / "service_dependencies.py",
        app_root / "interfaces" / "auth_dependencies.py",
    }
    violations: list[str] = []

    for py_file in app_root.rglob("*.py"):
        if py_file in shim_files:
            continue
        violations.extend(_collect_forbidden_imports(py_file))

    assert not violations, (
        "检测到新增 shim 依赖，请改为引用新的 dependencies 模块：\n"
        + "\n".join(sorted(violations))
    )
