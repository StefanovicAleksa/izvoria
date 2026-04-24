# Canvas: test_storage_api_v3
# Path: tests/features/rag_storage/service/test_storage_api.py

import pytest
import uuid
from unittest.mock import MagicMock

from app.features.context_unification.domain.entities import UnifiedTranscript, UnifiedChunk
from app.features.rag_storage.service.storage_api import RAGStorageService
from app.core.model_manager.domain.entities import LoadedModel


class TestRAGStorageService:
    @pytest.fixture
    def mocks(self):
        return {
            "rag_repo": MagicMock(),
            "embedder": MagicMock(),
            "orchestrator": MagicMock()
        }

    @pytest.fixture
    def service(self, mocks):
        return RAGStorageService(
            rag_repo=mocks["rag_repo"],
            orchestrator=mocks["orchestrator"],
            embedder=mocks["embedder"]
        )

    def test_index_transcript_success_flow(self, service, mocks):
        source_id = uuid.uuid4()
        chunk = UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0, end_sec=1, text_payload="Test\u200b string")
        transcript = UnifiedTranscript(source_id=source_id, chunks=[chunk])

        # Setup Orchestrator mock to return a dummy LoadedModel
        mock_loaded_model = MagicMock(spec=LoadedModel)
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model

        mocks["embedder"].embed_batch.return_value = [[0.1, 0.2]]
        mocks["rag_repo"].save_rag_chunks.return_value = 1

        result = service.index_transcript(source_id, transcript)

        assert result.success is True

        # Verify single atomic delete ran
        mocks["rag_repo"].delete_by_source_id.assert_called_once_with(source_id)

        # Verify Payload Normalization reached the embedder
        normalized_texts = mocks["embedder"].embed_batch.call_args[0][1]
        assert normalized_texts[0] == "Test string"

        # Verify Orchestrator Lifecycle
        mocks["orchestrator"].acquire_model.assert_called_once()
        mocks["orchestrator"].release_model.assert_called_once_with(
            mock_loaded_model, mocks["embedder"], flush=True
        )

    def test_index_transcript_empty_chunks(self, service, mocks):
        source_id = uuid.uuid4()
        transcript = UnifiedTranscript(source_id=source_id, chunks=[])

        result = service.index_transcript(source_id, transcript)

        assert result.success is True
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_index_transcript_guarantees_vram_release_on_error(self, service, mocks):
        source_id = uuid.uuid4()
        chunk = UnifiedChunk(chunk_id=uuid.uuid4(), start_sec=0, end_sec=1, text_payload="Crash text")
        transcript = UnifiedTranscript(source_id=source_id, chunks=[chunk])

        mock_loaded_model = MagicMock(spec=LoadedModel)
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model

        mocks["embedder"].embed_batch.side_effect = Exception("Inference Error")

        result = service.index_transcript(source_id, transcript)

        assert result.success is False
        mocks["orchestrator"].release_model.assert_called_once()