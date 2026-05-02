# Canvas: test_generation_api_v2
# Path: tests/features/qa_generation/service/test_generation_api.py

import pytest
import uuid
from unittest.mock import MagicMock
from app.features.qa_generation.service.generation_api import QAGenerationAPI
from app.features.rag_retrieval.domain.entities import RetrievedChunk


class TestQAGenerationAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "adapter": MagicMock(),
            "orchestrator": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return QAGenerationAPI(adapter=mocks["adapter"], orchestrator=mocks["orchestrator"])

    def test_generate_success_flow(self, api, mocks):
        chunk = RetrievedChunk(
            chunk_id=uuid.uuid4(), source_id=uuid.uuid4(),
            text_payload="Context text", start_sec=0, end_sec=5, relevance_score=1.0
        )
        object.__setattr__(chunk, 'physical_path', '/vault/vid.mp4')

        mocks["adapter"].generate_answer.return_value = "Answer [Doc 1]"

        response = api.generate("What happened?", [chunk])

        assert response.query == "What happened?"
        assert "[Doc 1]" in response.generated_answer
        assert len(response.citations) == 1

        # Verify VRAM Lifecycle
        mocks["orchestrator"].acquire_model.assert_called_once()
        mocks["orchestrator"].release_model.assert_called_once()

    def test_generate_error_mapping(self, api, mocks):
        mocks["orchestrator"].acquire_model.side_effect = RuntimeError("OOM")

        with pytest.raises(Exception):  # Caught and re-raised by map_qa_error
            api.generate("query", [])