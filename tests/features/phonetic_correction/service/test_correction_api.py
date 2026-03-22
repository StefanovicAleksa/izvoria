# Canvas: test_correction_api_v4
# Path: tests/features/phonetic_correction/service/test_correction_api.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.phonetic_correction.service.correction_api import PhoneticCorrectionAPI
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence
from app.features.phonetic_correction.domain.entities import PhoneticCorrectedTranscript, PhoneticPatch


class TestPhoneticCorrectionAPI:
    @pytest.fixture
    def mock_repo(self):
        return MagicMock()

    @pytest.fixture
    def mock_orchestrator(self):
        return MagicMock(spec=ModelManagerOrchestrator)

    @pytest.fixture
    def mock_adapter(self):
        return MagicMock()

    @pytest.fixture
    def api(self, mock_repo, mock_orchestrator, mock_adapter):
        return PhoneticCorrectionAPI(
            repository=mock_repo,
            orchestrator=mock_orchestrator,
            patch_adapter=mock_adapter
        )

    def test_execute_idempotent_return(self, api, mock_repo):
        transcript_id = uuid.uuid4()
        mock_existing = PhoneticCorrectedTranscript(source_id=transcript_id, sentences=[], audit_trail=[])
        mock_repo.get_by_source_id.return_value = mock_existing

        dummy_transcript = ReconstructedTranscript(source_id=transcript_id, sentences=[])
        result = api.execute(dummy_transcript, "GENERAL", [])

        assert result == mock_existing
        mock_repo.get_by_source_id.assert_called_once_with(transcript_id)

    def test_execute_passthrough_empty_transcript(self, api, mock_repo, mock_orchestrator):
        mock_repo.get_by_source_id.return_value = None

        transcript_id = uuid.uuid4()
        dummy_transcript = ReconstructedTranscript(source_id=transcript_id, sentences=[])

        result = api.execute(dummy_transcript, "GENERAL", [])

        assert result.source_id == transcript_id
        assert len(result.sentences) == 0
        mock_orchestrator.acquire_model.assert_not_called()

    @patch("app.features.phonetic_correction.service.correction_api.DictionaryLoader")
    def test_execute_applies_patches_and_unloads_vram(self, mock_dict_loader, api, mock_repo, mock_orchestrator, mock_adapter):
        mock_repo.get_by_source_id.return_value = None
        mock_dict_loader.build_prompt_dictionary.return_value = "- test dictionary term"

        transcript_id = uuid.uuid4()
        dummy_sentence = ReconstructedSentence(
            sentence_id=uuid.uuid4(),
            start_sec=0,
            end_sec=1,
            text="I have a cute angina.",
            speaker_id="S0",
            words=[]
        )
        dummy_transcript = ReconstructedTranscript(source_id=transcript_id, sentences=[dummy_sentence])

        mock_loaded_model = MagicMock()
        mock_orchestrator.acquire_model.return_value = mock_loaded_model

        mock_adapter.generate_patches.return_value = [
            PhoneticPatch(original="a cute angina", correction="acute angina")
        ]

        mock_repo.save.side_effect = lambda x: x

        result = api.execute(dummy_transcript, "MEDICAL", [], release_after=True)

        mock_orchestrator.acquire_model.assert_called_once()
        mock_adapter.generate_patches.assert_called_once()
        mock_orchestrator.release_model.assert_called_once_with(mock_loaded_model, mock_adapter, flush=True)

        assert "acute angina" in result.sentences[0].text
        assert len(result.audit_trail) == 1
        assert result.audit_trail[0].original_text == "I have a cute angina."
        assert result.audit_trail[0].corrected_text == "I have acute angina."