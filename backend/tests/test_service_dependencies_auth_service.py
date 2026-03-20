from app.interfaces.dependencies import services as service_dependencies


def test_get_auth_service_should_bind_current_redis_client_each_call(monkeypatch) -> None:
    redis_client_1 = object()
    redis_client_2 = object()
    redis_clients = iter([redis_client_1, redis_client_2])

    class _FakeRefreshTokenStore:
        def __init__(self, redis_client) -> None:
            self.redis_client = redis_client

    class _FakeAccessTokenBlacklistStore:
        def __init__(self, redis_client) -> None:
            self.redis_client = redis_client

    class _FakeRegisterVerificationCodeStore:
        def __init__(self, redis_client) -> None:
            self.redis_client = redis_client

    class _FakeAuthRateLimitStore:
        def __init__(self, redis_client) -> None:
            self.redis_client = redis_client

    class _FakeEmailSender:
        pass

    class _FakeAuthService:
        def __init__(
                self,
                *,
                uow_factory,
                refresh_token_store,
                access_token_blacklist_store,
                auth_rate_limit_store,
                register_verification_code_store,
                email_sender,
        ) -> None:
            self.uow_factory = uow_factory
            self.refresh_token_store = refresh_token_store
            self.access_token_blacklist_store = access_token_blacklist_store
            self.auth_rate_limit_store = auth_rate_limit_store
            self.register_verification_code_store = register_verification_code_store
            self.email_sender = email_sender

    monkeypatch.setattr(service_dependencies, "get_redis_client", lambda: next(redis_clients))
    monkeypatch.setattr(service_dependencies, "RedisRefreshTokenStore", _FakeRefreshTokenStore)
    monkeypatch.setattr(
        service_dependencies,
        "RedisAccessTokenBlacklistStore",
        _FakeAccessTokenBlacklistStore,
    )
    monkeypatch.setattr(
        service_dependencies,
        "RedisRegisterVerificationCodeStore",
        _FakeRegisterVerificationCodeStore,
    )
    monkeypatch.setattr(
        service_dependencies,
        "RedisAuthRateLimitStore",
        _FakeAuthRateLimitStore,
    )
    monkeypatch.setattr(service_dependencies, "SMTPEmailSender", _FakeEmailSender)
    monkeypatch.setattr(service_dependencies, "AuthService", _FakeAuthService)

    first = service_dependencies.get_auth_service()
    second = service_dependencies.get_auth_service()

    assert first is not second
    assert first.refresh_token_store.redis_client is redis_client_1
    assert second.refresh_token_store.redis_client is redis_client_2
    assert first.access_token_blacklist_store.redis_client is redis_client_1
    assert second.access_token_blacklist_store.redis_client is redis_client_2
    assert first.auth_rate_limit_store.redis_client is redis_client_1
    assert second.auth_rate_limit_store.redis_client is redis_client_2
