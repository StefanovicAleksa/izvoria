import pytest
from unittest.mock import MagicMock, call, patch
from sqlalchemy.exc import OperationalError, ProgrammingError

from app.core.database import exceptions as exc
from app.core.database.data.migrations.pre_alembic import PRE_ALEMBIC_CATCH_UP
from app.core.database.service.migrations import BASELINE_REVISION, MIGRATIONS_DIR, upgrade_to_head

MODULE = "app.core.database.service.migrations"


def _inspector(*tables):
    inspector = MagicMock()
    inspector.has_table.side_effect = lambda name: name in tables
    return inspector


class TestUpgradeToHead:
    @pytest.fixture
    def engine(self):
        return MagicMock()

    @pytest.fixture
    def connection(self, engine):
        return engine.begin.return_value.__enter__.return_value

    @pytest.fixture
    def command(self):
        with patch(f"{MODULE}.command") as command:
            yield command

    @pytest.fixture
    def base(self):
        with patch(f"{MODULE}.Base") as base:
            yield base

    @pytest.fixture
    def import_models(self):
        with patch(f"{MODULE}.import_all_models") as import_models:
            yield import_models

    def test_fresh_database_is_only_upgraded(self, engine, connection, command, base, import_models):
        with patch(f"{MODULE}.inspect", return_value=_inspector()):
            upgrade_to_head(engine)

        [(config, revision)] = [c.args for c in command.upgrade.call_args_list]
        assert revision == "head"
        assert config.get_main_option("script_location") == str(MIGRATIONS_DIR)
        assert config.attributes["connection"] is connection
        command.stamp.assert_not_called()
        base.metadata.create_all.assert_not_called()
        connection.execute.assert_not_called()

    def test_already_migrated_database_is_only_upgraded(self, engine, connection, command, base, import_models):
        with patch(f"{MODULE}.inspect", return_value=_inspector("alembic_version", "sources")):
            upgrade_to_head(engine)

        command.upgrade.assert_called_once()
        command.stamp.assert_not_called()
        base.metadata.create_all.assert_not_called()

    def test_pre_alembic_database_is_caught_up_and_stamped_before_upgrading(
            self, engine, connection, command, base, import_models):
        order = MagicMock()
        order.attach_mock(import_models, "import_models")
        order.attach_mock(base.metadata.create_all, "create_all")
        order.attach_mock(connection.execute, "execute")
        order.attach_mock(command.stamp, "stamp")
        order.attach_mock(command.upgrade, "upgrade")

        with patch(f"{MODULE}.inspect", return_value=_inspector("sources")):
            upgrade_to_head(engine)

        names = [c[0] for c in order.mock_calls]
        assert names == ["import_models", "create_all", *["execute"] * len(PRE_ALEMBIC_CATCH_UP), "stamp", "upgrade"]
        base.metadata.create_all.assert_called_once_with(bind=connection)
        assert [str(c.args[0]) for c in connection.execute.call_args_list] == list(PRE_ALEMBIC_CATCH_UP)
        assert command.stamp.call_args.args[1] == BASELINE_REVISION == "0001"

    def test_unexpected_failure_raises_a_migration_error(self, engine, command, base, import_models):
        command.upgrade.side_effect = RuntimeError("Can't locate revision")

        with patch(f"{MODULE}.inspect", return_value=_inspector()):
            with pytest.raises(exc.DBMigrationError, match="Can't locate revision"):
                upgrade_to_head(engine)

    def test_failing_ddl_raises_a_migration_error(self, engine, command, base, import_models):
        command.upgrade.side_effect = ProgrammingError("ALTER TABLE", {}, Exception("column already exists"))

        with patch(f"{MODULE}.inspect", return_value=_inspector()):
            with pytest.raises(exc.DBMigrationError):
                upgrade_to_head(engine)

    def test_unreachable_database_is_reported_as_unreachable(self, engine):
        engine.begin.side_effect = OperationalError("connect", {}, Exception("connection refused"))

        with pytest.raises(exc.DBUnreachableError):
            upgrade_to_head(engine)


class TestPreAlembicCatchUp:
    """Frozen history, but these backfills are what keep a pre-rename row from breaking its conversation or job."""

    def test_backfills_the_renamed_reasoning_values(self):
        assert "UPDATE session_turns SET mode = 'reasoning' WHERE mode = 'deep_research'" in PRE_ALEMBIC_CATCH_UP
        assert ("UPDATE jobs SET job_type = 'reasoning_turn' WHERE job_type = 'deep_research_reasoning'"
                in PRE_ALEMBIC_CATCH_UP)

    def test_every_column_addition_is_idempotent(self):
        alters = [s for s in PRE_ALEMBIC_CATCH_UP if s.startswith("ALTER TABLE") and "ALTER COLUMN" not in s]
        assert alters and all("ADD COLUMN IF NOT EXISTS" in s for s in alters)
