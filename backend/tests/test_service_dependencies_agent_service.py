from types import SimpleNamespace

from app.interfaces import service_dependencies


def test_get_agent_service_prefers_app_state_instance() -> None:
    expected_service = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(agent_service=expected_service)))

    got = service_dependencies.get_agent_service(request=request)

    assert got is expected_service


def test_get_agent_service_falls_back_to_lifespan_singleton(monkeypatch) -> None:
    fallback_service = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    monkeypatch.setattr(
        service_dependencies,
        "get_agent_service_for_lifespan",
        lambda: fallback_service,
    )

    got = service_dependencies.get_agent_service(request=request)

    assert got is fallback_service
