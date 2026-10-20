# Path: tests/features/source_profile/data/test_postgres_profile_repo.py

import uuid
from unittest.mock import MagicMock, Mock

import pytest

from app.core.database import exceptions as db_exc
from app.features.source_profile.data.postgres_profile_repo import PostgresSourceProfileRepository
from app.features.source_profile.domain.entities import SourceIdentity, SourceProfile
from app.features.source_profile.domain.enums import ProfileOrigin
from app.features.source_profile.exceptions import SourceProfileDataError, SourceProfileSystemError


class TestPostgresSourceProfileRepository:
    @pytest.fixture
    def mock_db_manager(self):
        db_manager = Mock()
        ctx = MagicMock()
        db_manager.session.return_value = ctx
        ctx.__enter__.return_value = MagicMock()
        return db_manager

    @pytest.fixture
    def repository(self, mock_db_manager):
        return PostgresSourceProfileRepository(mock_db_manager)

    @pytest.fixture
    def session(self, mock_db_manager):
        return mock_db_manager.session.return_value.__enter__.return_value

    @staticmethod
    def _row(**overrides):
        row = {"source_id": uuid.uuid4(), "name": "module_01.mp4", "language": "en", "file_name": "module_01.mp4",
               "title": None, "topic": None, "origin": None, "profile_language": None}
        row.update(overrides)
        return row

    def test_get_identity_maps_the_source_row(self, repository, session):
        row = self._row(file_name=None)
        session.execute.return_value.mappings.return_value.first.return_value = row

        identity = repository.get_identity(row["source_id"])

        assert identity == SourceIdentity(source_id=row["source_id"], name="module_01.mp4", file_name="",
                                          language="en")
        assert session.execute.call_args.args[1] == {"source_id": row["source_id"]}

    def test_get_identity_of_a_missing_source_is_none(self, repository, session):
        session.execute.return_value.mappings.return_value.first.return_value = None
        assert repository.get_identity(uuid.uuid4()) is None

    def test_get_maps_the_profile_model(self, repository, session):
        source_id = uuid.uuid4()
        model = MagicMock(source_id=source_id, title="Contract Law 1", topic=None, origin="edited", language="sr")
        session.execute.return_value.scalar_one_or_none.return_value = model

        assert repository.get(source_id) == SourceProfile(source_id=source_id, title="Contract Law 1", topic="",
                                                          origin=ProfileOrigin.EDITED, language="sr")

    def test_save_upserts_on_the_source_id(self, repository, session):
        profile = SourceProfile(source_id=uuid.uuid4(), title="T", topic="x", origin=ProfileOrigin.GENERATED,
                                language="en")

        assert repository.save(profile) is profile

        statement = session.execute.call_args.args[0]
        compiled = str(statement.compile(compile_kwargs={"literal_binds": False}))
        assert "INSERT INTO source_profiles" in compiled and "ON CONFLICT (source_id) DO UPDATE" in compiled

    def test_list_indexed_attaches_profiles_where_present(self, repository, session):
        bare = self._row()
        profiled = self._row(title="Statistics 110, Lecture 4", topic="Bayes.", origin="generated",
                             profile_language="en")
        session.execute.return_value.mappings.return_value.all.return_value = [bare, profiled]

        sources = repository.list_indexed()

        assert sources[0].profile is None and sources[0].display_title == "module_01.mp4"
        assert sources[1].profile.title == "Statistics 110, Lecture 4"
        assert sources[1].profile.origin is ProfileOrigin.GENERATED
        assert [s.source_id for s in repository.list_unprofiled()] == [bare["source_id"]]

    @pytest.mark.parametrize("raised, expected", [
        (db_exc.DBUnreachableError("down"), SourceProfileSystemError),
        (db_exc.DBForeignKeyError("no source"), SourceProfileDataError),
    ])
    def test_database_errors_are_mapped(self, repository, session, raised, expected):
        session.execute.side_effect = raised
        with pytest.raises(expected):
            repository.list_indexed()
