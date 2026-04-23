# Canvas: test_postgres_dense_repo_v3
# Path: tests/features/rag_storage/data/test_postgres_dense_repo.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.features.context_unification.domain.entities import UnifiedChunk
from app.features.rag_storage.data.postgres_dense_repo import PostgresDenseRepository


class TestPostgresDenseRepository:

    @patch("app.features.rag_storage.data.postgres_dense_repo.db_manager")
    def test_save_dense_vectors_success(self, mock_db_manager):
        session_mock = MagicMock()
        mock_db_manager.session.return_value.__enter__.return_value = session_mock

        repo = PostgresDenseRepository()

        source_id = uuid.uuid4()
        chunk = UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0, end_sec=1, text_payload="Test")
        embedding = [0.1] * 1024

        result = repo.save_dense_vectors(source_id, [chunk], [embedding])

        assert result == 1
        session_mock.bulk_save_objects.assert_called_once()
        saved_models = session_mock.bulk_save_objects.call_args[0][0]
        assert len(saved_models) == 1
        assert saved_models[0].source_id == source_id

    def test_save_dense_vectors_mismatch_raises(self):
        repo = PostgresDenseRepository()
        chunk = UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0, end_sec=1, text_payload="Test")

        # FIX: Pass 1 chunk, but 2 embeddings so neither is empty, but they mismatch
        bad_embeddings = [[0.1], [0.2]]

        with pytest.raises(ValueError, match="Mismatch between number of chunks and embeddings"):
            repo.save_dense_vectors(uuid.uuid4(), [chunk], bad_embeddings)

    @patch("app.features.rag_storage.data.postgres_dense_repo.db_manager")
    def test_delete_by_source_id_success(self, mock_db_manager):
        session_mock = MagicMock()
        mock_db_manager.session.return_value.__enter__.return_value = session_mock
        session_mock.execute.return_value.rowcount = 5

        repo = PostgresDenseRepository()
        repo.delete_by_source_id(uuid.uuid4())

        session_mock.execute.assert_called_once()
        session_mock.commit.assert_called_once()