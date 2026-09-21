# Canvas: test_rag_retrieval_api_v2
# Path: tests/features/rag_retrieval/service/test_retrieval_api.py

import pytest
import uuid
from unittest.mock import MagicMock

from app.features.rag_retrieval.service.retrieval_api import RetrievalAPI
from app.features.rag_retrieval.domain.entities import SearchQuery, RetrievedChunk
from app.core.model_manager.domain.entities import LoadedModel


class TestRetrievalAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "orchestrator": MagicMock(),
            "embedding_adapter": MagicMock(),
            "search_repo": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return RetrievalAPI(
            orchestrator=mocks["orchestrator"],
            embedding_adapter=mocks["embedding_adapter"],
            search_repo=mocks["search_repo"]
        )

    def test_search_success_flow(self, api, mocks):
        query = SearchQuery(prompt="test search", limit=5)

        # Mock VRAM Load
        mock_loaded_model = MagicMock(spec=LoadedModel)
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model

        # Mock Embedder
        mocks["embedding_adapter"].embed_prompt.return_value = [0.1, 0.2]

        # Mock DB Search
        chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(),
            source_id=uuid.uuid4(),
            text_payload="Match",
            start_sec=0,
            end_sec=5,
            relevance_score=0.05,
            extracted_file_id=uuid.uuid4()
        )
        mocks["search_repo"].execute_hybrid_search.return_value = [chunk]

        result = api.search(query)

        assert result.query == "test search"
        assert len(result.results) == 1

        # Verify Orchestrator Lifecycle (Zero-Leakage check)
        mocks["orchestrator"].acquire_model.assert_called_once()
        mocks["orchestrator"].release_model.assert_called_once_with(
            mock_loaded_model, mocks["embedding_adapter"], flush=True
        )

    def test_search_guarantees_vram_release_on_error(self, api, mocks):
        query = SearchQuery(prompt="crash test", limit=5)

        mock_loaded_model = MagicMock(spec=LoadedModel)
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model

        # Force a crash during embedding
        mocks["embedding_adapter"].embed_prompt.side_effect = Exception("Model Crash")

        with pytest.raises(Exception):
            api.search(query)

        # Ensure the finally block released the VRAM even during a crash
        mocks["orchestrator"].release_model.assert_called_once()

    def test_search_fetches_the_reranker_candidate_pool(self, api, mocks):
        from app.core.config import settings

        source_id = uuid.uuid4()
        mocks["orchestrator"].acquire_model.return_value = MagicMock(spec=LoadedModel)
        mocks["embedding_adapter"].embed_prompt.return_value = [0.1, 0.2]
        mocks["search_repo"].execute_hybrid_search.return_value = []

        api.search(SearchQuery(prompt="what did the witness say", limit=5, source_id_filter=source_id))

        mocks["search_repo"].execute_hybrid_search.assert_called_once_with(
            query_vector=[0.1, 0.2],
            query_text="what did the witness say",
            limit=settings.RAG_RETRIEVAL_CANDIDATES,
            source_id_filter=source_id
        )

    @staticmethod
    def _chunk(source_id=None, start=0.0) -> RetrievedChunk:
        return RetrievedChunk(chunk_id=uuid.uuid4(), source_id=source_id or uuid.uuid4(), text_payload="t",
                              start_sec=start, end_sec=start + 1, relevance_score=0.5)

    def test_expand_neighbors_appends_new_neighbours_of_the_top_chunks(self, api, mocks):
        source = uuid.uuid4()
        chunks = [self._chunk(source, 50), self._chunk(source, 60), self._chunk(source, 90)]
        after, before = self._chunk(source, 70), self._chunk(source, 40)
        # chunks[1] is itself a neighbour of chunks[0] and must not be added twice
        mocks["search_repo"].fetch_neighbors.return_value = [after, chunks[1], before]

        result = api.expand_neighbors(chunks, top_k=2)

        mocks["search_repo"].fetch_neighbors.assert_called_once_with([chunks[0].chunk_id, chunks[1].chunk_id])
        assert result == chunks + [before, after]
        mocks["orchestrator"].acquire_model.assert_not_called()

    @pytest.mark.parametrize("top_k, chunk_count", [(0, 2), (3, 0)])
    def test_expand_neighbors_noop_cases_skip_the_database(self, api, mocks, top_k, chunk_count):
        chunks = [self._chunk() for _ in range(chunk_count)]

        assert api.expand_neighbors(chunks, top_k=top_k) == chunks
        mocks["search_repo"].fetch_neighbors.assert_not_called()

    def test_expand_neighbors_maps_errors(self, api, mocks):
        mocks["search_repo"].fetch_neighbors.side_effect = RuntimeError("db down")

        with pytest.raises(Exception) as exc_info:
            api.expand_neighbors([self._chunk()], top_k=1)
        assert "neighbouring" in str(exc_info.value)

    def test_get_chunks_delegates_to_the_repository_without_a_model(self, api, mocks):
        ids = [uuid.uuid4(), uuid.uuid4()]
        mocks["search_repo"].fetch_chunks.return_value = ["chunk"]

        assert api.get_chunks(ids) == ["chunk"]
        mocks["search_repo"].fetch_chunks.assert_called_once_with(ids)
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_get_chunks_maps_errors(self, api, mocks):
        mocks["search_repo"].fetch_chunks.side_effect = RuntimeError("db down")

        with pytest.raises(Exception) as exc_info:
            api.get_chunks([uuid.uuid4()])
        assert "by id" in str(exc_info.value)
