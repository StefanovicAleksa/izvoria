# Path: integration_tests/core/database/test_firm_database_integration.py

import psycopg2
import pytest
from sqlalchemy.engine import make_url

from app.core.config import settings
from app.core.database.service.manager import DatabaseManager
from app.core.database.service.migrations import upgrade_to_head
from app.features.search_history.data.repositories import SearchHistoryRepository
from app.features.search_history.domain.enums import SearchHistoryMode

pytestmark = pytest.mark.integration

FIRMS = ("onyx_test_firm_a", "onyx_test_firm_b")


def _admin(sql: str) -> None:
    connection = psycopg2.connect(host=settings.POSTGRES_SERVER, port=settings.POSTGRES_PORT,
                                  user=settings.POSTGRES_USER, password=settings.POSTGRES_PASSWORD,
                                  dbname=settings.POSTGRES_DB)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
    finally:
        connection.close()


@pytest.fixture
def firm_databases():
    """Two firms' own databases, as the hosted API creates them: same schema, nothing shared."""
    managers = []
    try:
        for name in FIRMS:
            _admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
            _admin(f'CREATE DATABASE "{name}"')
            manager = DatabaseManager(url=make_url(settings.DATABASE_URL).set(database=name)
                                      .render_as_string(hide_password=False), pool_size=1, max_overflow=0)
            upgrade_to_head(manager.get_engine())
            managers.append(manager)
        yield managers
    finally:
        for manager in managers:
            manager.dispose()
        for name in FIRMS:
            _admin(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')


class TestFirmDatabases:
    def test_one_firms_history_never_reaches_another_firm(self, firm_databases):
        firm_a, firm_b = (SearchHistoryRepository(manager) for manager in firm_databases)

        record = firm_a.record("termination clause", SearchHistoryMode.STANDARD, 5, [], [], 1, owner_id="alice")

        assert firm_a.get(record.id) is not None
        assert firm_b.get(record.id) is None
        assert firm_b.list(50) == []

    def test_a_firm_pool_is_sized_as_asked(self, firm_databases):
        pool = firm_databases[0].get_engine().pool

        assert (pool.size(), pool._max_overflow) == (1, 0)  # noqa: SLF001
