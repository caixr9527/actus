from unittest.mock import MagicMock

from app.infrastructure.storage.postgres import _register_pgvector_codec, register_vector


def test_register_pgvector_codec_should_delegate_to_asyncpg_helper() -> None:
    dbapi_connection = MagicMock()

    _register_pgvector_codec(dbapi_connection, None)

    dbapi_connection.run_async.assert_called_once_with(register_vector)
