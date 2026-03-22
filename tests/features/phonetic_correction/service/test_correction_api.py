# Canvas: test_correction_api_v3
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
        # API now correctly takes the orchestrator dependency
        return PhoneticCorrectionAPI(
            repository=mock_repo,
            orchestrator=mock_orchestrator,
            patch_adapter=mock_adapter
        )

    def test_execute_idempotent_return(self, api, mock_repo):
        """Ensures the API returns early if the Phase 8 transcript already exists in the DB."""
        transcript_id = uuid.uuid4()
        mock_existing = PhoneticCorrectedTranscript(source_id=transcript_id, sentences=[], audit_trail=[])
        mock_repo.get_by_source_id.return_value = mock_existing

        dummy_transcript = ReconstructedTranscript(source_id=transcript_id, sentences=[])
        result = api.execute(dummy_transcript, "GENERAL", [])

        assert result == mock_existing
        mock_repo.get_by_source_id.assert_called_once_with(transcript_id)

    def test_execute_passthrough_empty_transcript(self, api, mock_repo, mock_orchestrator):
        """Ensures that an empty transcript bypasses the GPU entirely."""
        mock_repo.get_by_source_id.return_value = None

        transcript_id = uuid.uuid4()
        dummy_transcript = ReconstructedTranscript(source_id=transcript_id, sentences=[])

        result = api.execute(dummy_transcript, "GENERAL", [])

        assert result.source_id == transcript_id
        assert len(result.sentences) == 0
        assert len(result.audit_trail) == 0

        # GPU should never be touched for an empty file
        mock_orchestrator.acquire_model.assert_not_called()

    @patch("app.features.phonetic_correction.service.correction_api.DictionaryLoader")
    def test_execute_applies_patches_and_unloads_vram(self, mock_dict_loader, api, mock_repo, mock_orchestrator,
                                                      mock_adapter):
        """Tests the full success path: acquiring VRAM, chunking, patching, auditing, and releasing VRAM."""
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

        # Mock the Orchestrator's behavior
        mock_loaded_model = MagicMock()

        # Ensure the mock tokenizer returns a valid list so len(tokens) in the chunker doesn't crash
        mock_loaded_model.model_instance.tokenize.return_value = [1, 2, 3, 4]
        mock_orchestrator.acquire_model.return_value = mock_loaded_model

        mock_adapter.generate_patches.return_value = [
            PhoneticPatch(original="a cute angina", correction="acute angina")]

        # Echo the saved object back from the repository mock
        mock_repo.save.side_effect = lambda x: x

        # Execute
        result = api.execute(dummy_transcript, "MEDICAL", [], release_after=True)

        # 1. Ensure the API asked the Orchestrator for the model, NOT the adapter
        mock_orchestrator.acquire_model.assert_called_once()

        # 2. Ensure inference occurred
        mock_adapter.generate_patches.assert_called_once()

        # 3. Ensure the Orchestrator was commanded to release the model back from VRAM
        mock_orchestrator.release_model.assert_called_once_with(mock_loaded_model, mock_adapter, flush=True)

        # 4. Verify Domain logic
        assert "acute angina" in result.sentences[0].text
        assert len(result.audit_trail) == 1
        assert result.audit_trail[0].original_text == "I have a cute angina."
        assert result.audit_trail[0].corrected_text == "I have acute angina."