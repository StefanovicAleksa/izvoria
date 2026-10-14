# Path: tests/core/database/data/test_credentials.py

from unittest.mock import patch

import pytest
from sqlalchemy import text

from app.core.database.data import credentials
from app.core.database.data.engine import SQLAlchemyEngine


@pytest.fixture(autouse=True)
def no_provider():
    credentials.set_password_provider(None)
    yield
    credentials.set_password_provider(None)


class TestCredentials:
    def test_without_a_provider_the_urls_password_is_kept(self):
        cparams = {"password": "from-url"}
        credentials.apply_to_connection(cparams)
        assert cparams == {"password": "from-url"}

    def test_a_provider_supplies_each_new_connections_password(self):
        answers = iter(["first", "rotated"])
        credentials.set_password_provider(lambda: next(answers))

        first, second = {}, {"password": ""}
        credentials.apply_to_connection(first)
        credentials.apply_to_connection(second)

        assert (first["password"], second["password"]) == ("first", "rotated")


class TestEngineHook:
    def test_every_new_connection_goes_through_the_hook(self, tmp_path):
        seen = []
        with patch("app.core.database.data.engine.apply_to_connection", side_effect=lambda c: seen.append(dict(c))):
            engine = SQLAlchemyEngine(url=f"sqlite:///{tmp_path / 'db.sqlite3'}", pool_size=1, max_overflow=0)
            with engine.get_engine().connect() as connection:
                assert connection.execute(text("SELECT 1")).scalar() == 1
            engine.dispose_engine()

        assert len(seen) == 1

    def test_an_appliance_engine_is_unchanged_without_a_provider(self, tmp_path):
        engine = SQLAlchemyEngine(url=f"sqlite:///{tmp_path / 'db.sqlite3'}", pool_size=1, max_overflow=0)
        with engine.get_engine().connect() as connection:
            assert connection.execute(text("SELECT 2")).scalar() == 2
        engine.dispose_engine()
