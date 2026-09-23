# Canvas: test_rag_retrieval_postgres_repo_v4
# Path: tests/features/rag_retrieval/data/test_postgres_search_repo.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.features.rag_retrieval.data.postgres_search_repo import (
    PostgresSearchRepository, _CHUNKS_BY_ID_SQL, _FTS_EXPRESSION_SQL, _HYBRID_SEARCH_SQL, _NEIGHBORS_SQL
)
from app.core.anchors.domain.enums import AnchorKind
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
            "anchor_kind": "time", "page_start": None, "page_end": None, "section_path": None, "position": 3,
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
            "speaker_labels": None, "start_sec": 0, "end_sec": 1, "anchor_kind": "time", "page_start": None, "page_end": None, "section_path": None, "position": 3, "rrf_score": 0.01, "extracted_file_id": None
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

    @patch("app.features.rag_retrieval.data.postgres_search_repo.db_manager")
    def test_fetch_neighbors_maps_rows_as_unranked_chunks(self, mock_db_manager):
        session_mock = MagicMock()
        mock_db_manager.session.return_value.__enter__.return_value = session_mock
        anchor_ids = [uuid.uuid4(), uuid.uuid4()]
        row = {
            "chunk_id": uuid.uuid4(), "source_id": uuid.uuid4(), "text_payload": "payload",
            "raw_text": "[Speaker_0 @ 00:00:01]: Before.", "speaker_labels": None,
            "start_sec": 1, "end_sec": 9, "extracted_file_id": None,
            "anchor_kind": "time", "page_start": None, "page_end": None, "section_path": None, "position": 1,
        }
        session_mock.execute.return_value.mappings.return_value.all.return_value = [row]

        neighbors = PostgresSearchRepository().fetch_neighbors(anchor_ids)

        session_mock.execute.assert_called_once_with(
            _NEIGHBORS_SQL, {"chunk_ids": [str(anchor_ids[0]), str(anchor_ids[1])]}
        )
        assert len(neighbors) == 1
        assert neighbors[0].chunk_id == row["chunk_id"]
        assert neighbors[0].raw_text == "[Speaker_0 @ 00:00:01]: Before."
        assert neighbors[0].speaker_labels == []
        assert (neighbors[0].start_sec, neighbors[0].end_sec) == (1.0, 9.0)
        assert neighbors[0].relevance_score == 0.0
        assert neighbors[0].rerank_score is None

    @patch("app.features.rag_retrieval.data.postgres_search_repo.db_manager")
    def test_fetch_neighbors_without_anchors_skips_the_database(self, mock_db_manager):
        assert PostgresSearchRepository().fetch_neighbors([]) == []
        mock_db_manager.session.assert_not_called()

    def test_neighbors_query_takes_one_chunk_each_side_within_the_source(self):
        sql = str(_NEIGHBORS_SQL)

        # Reading order first (a document's chunks all have start_sec 0.0), then time for rows with no position
        assert "(COALESCE(v.position, -1), v.start_sec) < (a.position, a.start_sec)" in sql
        assert "(COALESCE(v.position, -1), v.start_sec) > (a.position, a.start_sec)" in sql
        assert "ORDER BY COALESCE(v.position, -1) DESC, v.start_sec DESC LIMIT 1" in sql
        assert "ORDER BY COALESCE(v.position, -1) ASC, v.start_sec ASC LIMIT 1" in sql
        assert sql.count("LIMIT 1") == 2
        assert "DISTINCT ON (n.chunk_id)" in sql

    @patch("app.features.rag_retrieval.data.postgres_search_repo.db_manager")
    def test_fetch_neighbors_maps_errors(self, mock_db_manager):
        mock_db_manager.session.return_value.__enter__.side_effect = RuntimeError("connection refused")

        with pytest.raises(RetrievalError):
            PostgresSearchRepository().fetch_neighbors([uuid.uuid4()])

    @staticmethod
    def _row(chunk_id):
        return {
            "chunk_id": str(chunk_id), "source_id": str(uuid.uuid4()), "text_payload": "payload",
            "raw_text": "[Speaker_0 @ 00:00:01]: Hello.", "speaker_labels": ["Speaker_0"],
            "start_sec": 1, "end_sec": 9, "extracted_file_id": None,
            "anchor_kind": "time", "page_start": None, "page_end": None, "section_path": None, "position": 1,
        }

    @patch("app.features.rag_retrieval.data.postgres_search_repo.db_manager")
    def test_fetch_chunks_returns_them_in_requested_order_skipping_missing(self, mock_db_manager):
        session_mock = MagicMock()
        mock_db_manager.session.return_value.__enter__.return_value = session_mock
        first, missing, second = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
        # The query returns rows in chunk_id order, and ids as strings
        session_mock.execute.return_value.mappings.return_value.all.return_value = [self._row(second), self._row(first)]

        chunks = PostgresSearchRepository().fetch_chunks([first, missing, second, first])

        session_mock.execute.assert_called_once_with(
            _CHUNKS_BY_ID_SQL, {"chunk_ids": [str(first), str(missing), str(second), str(first)]}
        )
        assert [c.chunk_id for c in chunks] == [first, second]
        assert chunks[0].raw_text == "[Speaker_0 @ 00:00:01]: Hello."
        assert chunks[0].speaker_labels == ["Speaker_0"]
        assert chunks[0].relevance_score == 0.0

    @patch("app.features.rag_retrieval.data.postgres_search_repo.db_manager")
    def test_fetch_chunks_without_ids_skips_the_database(self, mock_db_manager):
        assert PostgresSearchRepository().fetch_chunks([]) == []
        mock_db_manager.session.assert_not_called()

    @patch("app.features.rag_retrieval.data.postgres_search_repo.db_manager")
    def test_fetch_chunks_maps_errors(self, mock_db_manager):
        mock_db_manager.session.return_value.__enter__.side_effect = RuntimeError("connection refused")

        with pytest.raises(RetrievalError):
            PostgresSearchRepository().fetch_chunks([uuid.uuid4()])

    def test_chunks_by_id_query_joins_the_clip(self):
        sql = str(_CHUNKS_BY_ID_SQL)

        assert "LEFT JOIN video_clips vc ON vc.chunk_id = v.chunk_id" in sql
        assert "v.chunk_id = ANY(CAST(:chunk_ids AS uuid[]))" in sql


    @patch("app.features.rag_retrieval.data.postgres_search_repo.db_manager")
    def test_maps_a_document_row_to_its_page_anchor(self, mock_db_manager):
        session_mock = MagicMock()
        mock_db_manager.session.return_value.__enter__.return_value = session_mock
        chunk_id = uuid.uuid4()
        row = {
            "chunk_id": chunk_id, "source_id": uuid.uuid4(), "text_payload": "[Document Context: lease]\n\nText",
            "raw_text": "Text", "speaker_labels": [], "start_sec": 0.0, "end_sec": 0.0, "extracted_file_id": None,
            "anchor_kind": "page", "page_start": 7, "page_end": 8, "section_path": "Article 7", "position": 12,
        }
        session_mock.execute.return_value.mappings.return_value.all.return_value = [row]

        [chunk] = PostgresSearchRepository().fetch_chunks([chunk_id])

        assert chunk.anchor_kind is AnchorKind.PAGE
        assert (chunk.page_start, chunk.page_end, chunk.section_path, chunk.position) == (7, 8, "Article 7", 12)


class TestGetIndexFtsConfig:
    @pytest.mark.parametrize("expression, config", [
        ("to_tsvector('english'::regconfig, (text_payload)::text)", "english"),
        ("to_tsvector('serbian'::regconfig, (text_payload)::text)", "serbian"),
        (None, None),                    # vector_chunks not created yet
        ("text_payload", None),          # not a tsvector expression at all
    ])
    @patch("app.features.rag_retrieval.data.postgres_search_repo.db_manager")
    def test_reads_the_regconfig_from_the_generated_column(self, mock_db_manager, expression, config):
        session = mock_db_manager.session.return_value.__enter__.return_value
        session.execute.return_value.scalar.return_value = expression

        assert PostgresSearchRepository().get_index_fts_config() == config
        assert session.execute.call_args.args[0] is _FTS_EXPRESSION_SQL

    @patch("app.features.rag_retrieval.data.postgres_search_repo.db_manager")
    def test_database_failure_is_a_retrieval_error(self, mock_db_manager):
        mock_db_manager.session.side_effect = RuntimeError("connection refused")

        with pytest.raises(RetrievalError):
            PostgresSearchRepository().get_index_fts_config()
