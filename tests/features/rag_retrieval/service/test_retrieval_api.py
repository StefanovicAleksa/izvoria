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