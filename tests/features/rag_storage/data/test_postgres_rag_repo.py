# Canvas: test_postgres_rag_repo_v2
# Path: tests/features/rag_storage/data/test_postgres_rag_repo.py

import uuid
import pytest
from unittest.mock import MagicMock, patch
from sqlalchemy.exc import OperationalError

from app.core.config import Settings
from app.features.context_unification.domain.entities import UnifiedChunk
from app.features.rag_storage.data.postgres_rag_repo import PostgresRagRepository
from app.features.rag_storage.domain.entities import StoredChunkText
from app.features.rag_storage.exceptions.database import VectorDatabaseError


def _session(mock_db_manager) -> MagicMock:
    session_mock = MagicMock()
    mock_db_manager.session.return_value.__enter__.return_value = session_mock
    # Don't let the mocked context manager swallow exceptions raised inside the block
    mock_db_manager.session.return_value.__exit__.return_value = False
    return session_mock


class TestPostgresRagRepository:

    @patch("app.features.rag_storage.data.postgres_rag_repo.db_manager")
    def test_replace_rag_chunks_deletes_and_inserts_in_one_session(self, mock_db_manager):
        session_mock = _session(mock_db_manager)
        session_mock.execute.return_value.rowcount = 3

        repo = PostgresRagRepository()
        source_id = uuid.uuid4()
        chunk = UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0, end_sec=1, text_payload="Test")
        embedding = [0.1] * 1024

        result = repo.replace_rag_chunks(source_id, [chunk], [embedding])

        assert result == 1
        mock_db_manager.session.assert_called_once()
        session_mock.execute.assert_called_once()
        session_mock.bulk_save_objects.assert_called_once()
        saved_models = session_mock.bulk_save_objects.call_args[0][0]
        assert len(saved_models) == 1
        assert saved_models[0].source_id == source_id
        assert saved_models[0].chunk_id == chunk.chunk_id
        assert saved_models[0].text_payload == "Test"
        assert saved_models[0].embedding == embedding

    def test_replace_rag_chunks_mismatch_raises(self):
        repo = PostgresRagRepository()
        chunk = UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0, end_sec=1, text_payload="Test")

        with pytest.raises(ValueError, match="Mismatch between number of chunks and embeddings"):
            repo.replace_rag_chunks(uuid.uuid4(), [chunk], [[0.1], [0.2]])

    @patch("app.features.rag_storage.data.postgres_rag_repo.db_manager")
    def test_replace_rag_chunks_maps_database_errors(self, mock_db_manager):
        session_mock = _session(mock_db_manager)
        session_mock.bulk_save_objects.side_effect = OperationalError("INSERT", {}, Exception("connection lost"))

        repo = PostgresRagRepository()
        chunk = UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0, end_sec=1, text_payload="Test")

        with pytest.raises(VectorDatabaseError):
            repo.replace_rag_chunks(uuid.uuid4(), [chunk], [[0.1]])

    @patch("app.features.rag_storage.data.postgres_rag_repo.db_manager")
    def test_delete_by_source_id_success(self, mock_db_manager):
        session_mock = _session(mock_db_manager)
        session_mock.execute.return_value.rowcount = 5

        repo = PostgresRagRepository()
        repo.delete_by_source_id(uuid.uuid4())

        session_mock.execute.assert_called_once()

    @patch("app.features.rag_storage.data.postgres_rag_repo.db_manager")
    def test_replace_rag_chunks_persists_raw_text_context_and_speakers(self, mock_db_manager):
        session_mock = _session(mock_db_manager)
        chunk = UnifiedChunk(
            chunk_id=uuid.uuid4(), start_sec=0, end_sec=1,
            text_payload="[Audio Context: Intake.]\n\n[Speaker_0 @ 00:00:00]: I have angina.",
            raw_text="[Speaker_0 @ 00:00:00]: I have angina.",
            context_text="[Audio Context: Intake.]",
            speaker_labels=["Speaker_0"]
        )

        PostgresRagRepository().replace_rag_chunks(uuid.uuid4(), [chunk], [[0.1]])

        saved = session_mock.bulk_save_objects.call_args[0][0][0]
        assert saved.raw_text == "[Speaker_0 @ 00:00:00]: I have angina."
        assert saved.context_text == "[Audio Context: Intake.]"
        assert saved.speaker_labels == ["Speaker_0"]

    @patch("app.features.rag_storage.data.postgres_rag_repo.db_manager")
    def test_replace_rag_chunks_falls_back_to_payload_without_raw_text(self, mock_db_manager):
        session_mock = _session(mock_db_manager)
        chunk = UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0, end_sec=1, text_payload="Plain text")

        PostgresRagRepository().replace_rag_chunks(uuid.uuid4(), [chunk], [[0.1]])

        saved = session_mock.bulk_save_objects.call_args[0][0][0]
        assert saved.raw_text == "Plain text"
        assert saved.context_text is None
        assert saved.speaker_labels == []

    @pytest.mark.parametrize("language, fts_config", [("en", "english"), ("sr", "serbian")])
    @patch("app.features.rag_storage.data.postgres_rag_repo.db_manager")
    def test_chunks_are_keyword_indexed_in_the_batch_language(self, mock_db_manager, language, fts_config):
        session_mock = _session(mock_db_manager)
        chunks = [UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0, end_sec=1, text_payload="a"),
                  UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=1, end_sec=2, text_payload="b")]

        with patch("app.features.rag_storage.data.postgres_rag_repo.settings",
                   Settings(APPLIANCE_LANGUAGE=language)):
            PostgresRagRepository().replace_rag_chunks(uuid.uuid4(), chunks, [[0.1], [0.2]])

        saved = session_mock.bulk_save_objects.call_args[0][0]
        assert [model.fts_config for model in saved] == [fts_config, fts_config]


class TestInjectedDatabase:
    """The hosted API gives each firm's index its own database; without one, the appliance's own."""

    @patch("app.features.rag_storage.data.postgres_rag_repo.db_manager")
    def test_a_given_database_is_used_instead_of_the_appliances(self, appliance_db):
        firm_db = MagicMock()
        session = _session(firm_db)
        session.execute.return_value.rowcount = 2
        source_id = uuid.uuid4()

        PostgresRagRepository(firm_db).delete_by_source_id(source_id)

        session.execute.assert_called_once()
        appliance_db.session.assert_not_called()

    @patch("app.features.rag_storage.data.postgres_rag_repo.db_manager")
    def test_without_a_database_the_appliances_is_used(self, appliance_db):
        session = _session(appliance_db)
        session.execute.return_value.rowcount = 0

        PostgresRagRepository().delete_by_source_id(uuid.uuid4())

        session.execute.assert_called_once()


class TestPostgresRagRepositoryPayloadUpdates:

    @patch("app.features.rag_storage.data.postgres_rag_repo.db_manager")
    def test_read_chunk_texts_maps_rows_in_reading_order(self, mock_db_manager):
        session_mock = _session(mock_db_manager)
        chunk_id = uuid.uuid4()
        session_mock.execute.return_value.all.return_value = [
            MagicMock(chunk_id=chunk_id, context_text="[Audio Context: a]", raw_text="one")]

        texts = PostgresRagRepository().read_chunk_texts(uuid.uuid4())

        assert texts == [StoredChunkText(chunk_id=chunk_id, context_text="[Audio Context: a]", raw_text="one")]
        assert "ORDER BY vector_chunks.position, vector_chunks.start_sec" in str(session_mock.execute.call_args.args[0])

    @patch("app.features.rag_storage.data.postgres_rag_repo.db_manager")
    def test_update_chunk_payloads_updates_each_chunk_in_one_session(self, mock_db_manager):
        session_mock = _session(mock_db_manager)
        session_mock.execute.return_value.rowcount = 1
        chunks = [StoredChunkText(chunk_id=uuid.uuid4(), context_text="c", raw_text="r") for _ in range(2)]

        updated = PostgresRagRepository().update_chunk_payloads(uuid.uuid4(), chunks, ["p1", "p2"], [[0.1], [0.2]])

        assert updated == 2
        mock_db_manager.session.assert_called_once()
        statement = session_mock.execute.call_args_list[0].args[0]
        assert str(statement) == ("UPDATE vector_chunks SET text_payload=:text_payload, context_text=:context_text, "
                                  "embedding=:embedding WHERE vector_chunks.source_id = :source_id_1 "
                                  "AND vector_chunks.chunk_id = :chunk_id_1")
        assert statement.compile().params["text_payload"] == "p1"

    @patch("app.features.rag_storage.data.postgres_rag_repo.db_manager")
    def test_update_chunk_payloads_refuses_mismatched_lists(self, mock_db_manager):
        _session(mock_db_manager)
        chunks = [StoredChunkText(chunk_id=uuid.uuid4(), context_text="c", raw_text="r")]

        with pytest.raises(ValueError):
            PostgresRagRepository().update_chunk_payloads(uuid.uuid4(), chunks, ["p1", "p2"], [[0.1]])
