# Canvas: test_contextualizer_api_v2
# Path: tests/features/contextual_retrieval/service/test_contextualizer_api.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.phonetic_correction.domain.entities import PhoneticCorrectedTranscript
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.contextual_retrieval.service.contextualizer_api import ContextualizerAPI
from app.features.contextual_retrieval.domain.entities import ContextualizedTranscript
from app.features.contextual_retrieval.exceptions.model import SLMInferenceError


class TestContextualizerAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "repo": MagicMock(),
            "orchestrator": MagicMock(spec=ModelManagerOrchestrator),
            "slm_adapter": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return ContextualizerAPI(
            repository=mocks["repo"],
            orchestrator=mocks["orchestrator"],
            slm_adapter=mocks["slm_adapter"]
        )

    def test_update_transcript_delegates_to_repo(self, api, mocks):
        """Verifies the new boundary method correctly proxies to the repository."""
        mock_transcript = MagicMock(spec=ContextualizedTranscript)
        mocks["repo"].update.return_value = mock_transcript

        result = api.update_transcript(mock_transcript)

        assert result == mock_transcript
        mocks["repo"].update.assert_called_once_with(mock_transcript)

    def test_execute_idempotent_return(self, api, mocks):
        source_id = uuid.uuid4()
        mock_existing = ContextualizedTranscript(source_id=source_id, chunks=[])
        mocks["repo"].get_by_source_id.return_value = mock_existing

        dummy_input = PhoneticCorrectedTranscript(source_id=source_id, sentences=[], audit_trail=[])
        result = api.execute(dummy_input)

        assert result == mock_existing
        mocks["repo"].get_by_source_id.assert_called_once_with(source_id)
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_execute_passthrough_empty_transcript(self, api, mocks):
        mocks["repo"].get_by_source_id.return_value = None

        source_id = uuid.uuid4()
        dummy_input = PhoneticCorrectedTranscript(source_id=source_id, sentences=[], audit_trail=[])

        api.execute(dummy_input)

        mocks["repo"].save.assert_called_once()
        saved_arg = mocks["repo"].save.call_args[0][0]
        assert len(saved_arg.chunks) == 0
        mocks["orchestrator"].acquire_model.assert_not_called()

    @patch("app.features.contextual_retrieval.service.contextualizer_api.build_rag_chunks")
    def test_execute_success_flow_and_vram_release(self, mock_chunker, api, mocks):
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None

        dummy_sentence = ReconstructedSentence(
            sentence_id=uuid.uuid4(), text="I have angina.", speaker_id="S0", start_sec=0, end_sec=1, words=[]
        )
        dummy_input = PhoneticCorrectedTranscript(
            source_id=source_id, sentences=[dummy_sentence], audit_trail=[]
        )

        mock_chunker.return_value = [[dummy_sentence]]

        mock_loaded_model = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model
        mocks["slm_adapter"].generate_context.return_value = "Patient statement."
        mocks["repo"].save.side_effect = lambda x: x

        result = api.execute(dummy_input, release_after=True)

        mocks["orchestrator"].acquire_model.assert_called_once()
        mocks["slm_adapter"].generate_context.assert_called_once_with(
            model=mock_loaded_model,
            full_document_text="I have angina.",
            chunk_text="I have angina."
        )
        mocks["orchestrator"].release_model.assert_called_once_with(
            mock_loaded_model, mocks["slm_adapter"], flush=True
        )

        assert len(result.chunks) == 1
        assert result.chunks[0].context_string == "Patient statement."
        assert result.chunks[0].original_text == "I have angina."
        assert "Patient statement." in result.chunks[0].combined_text

    def test_execute_guarantees_vram_release_on_crash(self, api, mocks):
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None

        dummy_sentence = ReconstructedSentence(
            sentence_id=uuid.uuid4(), text="Test.", speaker_id="S0", start_sec=0, end_sec=1, words=[]
        )
        dummy_input = PhoneticCorrectedTranscript(
            source_id=source_id, sentences=[dummy_sentence], audit_trail=[]
        )

        mock_loaded_model = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = mock_loaded_model

        mocks["slm_adapter"].generate_context.side_effect = SLMInferenceError("Out of Memory")

        with pytest.raises(SLMInferenceError):
            api.execute(dummy_input, release_after=True)

        mocks["orchestrator"].release_model.assert_called_once_with(
            mock_loaded_model, mocks["slm_adapter"], flush=True
        )