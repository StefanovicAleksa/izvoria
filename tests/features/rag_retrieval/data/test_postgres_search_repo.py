# Canvas: test_rag_retrieval_postgres_repo_v4
# Path: tests/features/rag_retrieval/data/test_postgres_search_repo.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.features.rag_retrieval.data.postgres_search_repo import PostgresSearchRepository, _HYBRID_SEARCH_SQL
from app.features.rag_retrieval.exceptions.base import RetrievalError


class TestPostgresSearchRepository:

    @patch("app.features.rag_retrieval.data.postgres_search_repo.db_manager")
    def test_execute_hybrid_search_success(self, mock_db_manager):
        session_mock = MagicMock()
        mock_db_manager.session.return_value.__enter__.return_value = session_mock

        expected_file_id = uuid.uuid4()
        source_id = uuid.uuid4()

        mock_result = MagicMock()
        mock_row = {
            "chunk_id": uuid.uuid4(),
            "source_id": source_id,
            "text_payload": "[Audio Context: Intake visit.]\n\n[Speaker_0 @ 00:00:10]: Found text",
            "raw_text": "[Speaker_0 @ 00:00:10]: Found text",
            "speaker_labels": ["Speaker_0"],
            "start_sec": 10.5,
            "end_sec": 15.0,
            "rrf_score": 0.033,
            "extracted_file_id": expected_file_id
        }
        mock_result.mappings().all.return_value = [mock_row]
        session_mock.execute.return_value = mock_result

        repo = PostgresSearchRepository()
        results = repo.execute_hybrid_search(
            query_vector=[0.1, 0.2, 0.3],
            query_text="Find text",
            limit=40,
            source_id_filter=source_id
        )

        assert len(results) == 1
        assert results[0].text_payload == mock_row["text_payload"]
        assert results[0].raw_text == "[Speaker_0 @ 00:00:10]: Found text"
        assert results[0].speaker_labels == ["Speaker_0"]
        assert results[0].relevance_score == 0.033
        assert results[0].extracted_file_id == expected_file_id

        session_mock.execute.assert_called_once()
        sql, params = session_mock.execute.call_args[0]
        assert sql is _HYBRID_SEARCH_SQL
        assert params == {
            "query_vector": "[0.1,0.2,0.3]",
            "query_text": "Find text",
            "fts_config": "english",
            "source_id": source_id,
            "limit": 40,
            "fetch_depth": 200,
            "rrf_k": 60
        }

    @patch("app.features.rag_retrieval.data.postgres_search_repo.db_manager")
    def test_execute_hybrid_search_handles_null_speaker_labels(self, mock_db_manager):
        session_mock = MagicMock()
        mock_db_manager.session.return_value.__enter__.return_value = session_mock
        mock_result = MagicMock()
        mock_result.mappings().all.return_value = [{
            "chunk_id": uuid.uuid4(), "source_id": uuid.uuid4(), "text_payload": "p", "raw_text": "r",
            "speaker_labels": None, "start_sec": 0, "end_sec": 1, "rrf_score": 0.01, "extracted_file_id": None
        }]
        session_mock.execute.return_value = mock_result

        results = PostgresSearchRepository().execute_hybrid_search([0.1], "q", limit=5)

        assert results[0].speaker_labels == []

    def test_sparse_leg_uses_or_semantics_and_length_normalized_rank(self):
        """plainto_tsquery ANDs every lexeme; the query must be rewritten to OR before matching."""
        sql = str(_HYBRID_SEARCH_SQL)
        assert "replace(plainto_tsquery(CAST(:fts_config AS regconfig), :query_text)::text, ' & ', ' | ')" in sql
        assert "ts_rank(v.fts_document, sq.tsq, 1)" in sql
        assert "sq.tsq IS NOT NULL" in sql
        assert "v.raw_text" in sql

    @patch("app.features.rag_retrieval.data.postgres_search_repo.db_manager")
    def test_execute_hybrid_search_maps_errors(self, mock_db_manager):
        mock_db_manager.session.return_value.__enter__.side_effect = RuntimeError("connection refused")

        with pytest.raises(RetrievalError):
            PostgresSearchRepository().execute_hybrid_search([0.1], "q", limit=5)
