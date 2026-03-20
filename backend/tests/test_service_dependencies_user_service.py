from app.application.service import UserService
from app.interfaces.dependencies import services as service_dependencies


def test_get_user_service_should_return_new_service_instance() -> None:
    first = service_dependencies.get_user_service()
    second = service_dependencies.get_user_service()

    assert isinstance(first, UserService)
    assert isinstance(second, UserService)
    assert first is not second
