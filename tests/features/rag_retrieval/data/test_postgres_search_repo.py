# Canvas: test_rag_retrieval_postgres_repo_v3
# Path: tests/features/rag_retrieval/data/test_postgres_search_repo.py

import uuid
from unittest.mock import MagicMock, patch

from app.features.rag_retrieval.data.postgres_search_repo import PostgresSearchRepository


class TestPostgresSearchRepository:

    @patch("app.features.rag_retrieval.data.postgres_search_repo.db_manager")
    def test_execute_hybrid_search_success(self, mock_db_manager):
        session_mock = MagicMock()
        mock_db_manager.session.return_value.__enter__.return_value = session_mock

        expected_file_id = uuid.uuid4()

        # Mock the result rows that Postgres would return with the LEFT JOIN
        mock_result = MagicMock()
        mock_row = {
            "chunk_id": uuid.uuid4(),
            "source_id": uuid.uuid4(),
            "text_payload": "Found text",
            "start_sec": 10.5,
            "end_sec": 15.0,
            "rrf_score": 0.033,
            "extracted_file_id": expected_file_id
        }
        mock_result.mappings().all.return_value = [mock_row]
        session_mock.execute.return_value = mock_result

        repo = PostgresSearchRepository()
        query_vector = [0.1, 0.2, 0.3]

        results = repo.execute_hybrid_search(
            query_vector=query_vector,
            query_text="Find text",
            limit=5
        )

        assert len(results) == 1
        assert results[0].text_payload == "Found text"
        assert results[0].relevance_score == 0.033
        assert results[0].extracted_file_id == expected_file_id

        session_mock.execute.assert_called_once()