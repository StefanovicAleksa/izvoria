# Canvas: test_correction_api_v2
# Path: tests/features/phonetic_correction/service/test_correction_api.py

import uuid
import pytest
from unittest.mock import MagicMock, patch
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence
from app.features.phonetic_correction.service.correction_api import PhoneticCorrectionAPI
from app.features.phonetic_correction.domain.entities import PhoneticCorrectedTranscript, PhoneticPatch

class TestPhoneticCorrectionAPI:
    @pytest.fixture
    def mocks(self):
        return {"repo": MagicMock(), "adapter": MagicMock()}

    @pytest.fixture
    def api(self, mocks):
        return PhoneticCorrectionAPI(repository=mocks["repo"], patch_adapter=mocks["adapter"])

    def test_execute_idempotent_return(self, api, mocks):
        source_id = uuid.uuid4()
        existing_record = PhoneticCorrectedTranscript(source_id=source_id, sentences=[], audit_trail=[])
        mocks["repo"].get_by_source_id.return_value = existing_record

        transcript = ReconstructedTranscript(source_id=source_id, sentences=[])
        # FIX: Replaced string sub_domain with a list
        result = api.execute(transcript, "MEDICAL", ["CARDIOLOGY"])

        assert result == existing_record
        mocks["adapter"].load.assert_not_called()

    @patch("app.features.phonetic_correction.service.correction_api.DictionaryLoader")
    def test_execute_applies_patches_and_unloads_vram(self, mock_loader, api, mocks):
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None
        mock_loader.build_prompt_dictionary.return_value = "- acute angina"

        sentence = ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=0, end_sec=1, text="He has a cute angina.",
                                         speaker_id="S0", words=[])
        transcript = ReconstructedTranscript(source_id=source_id, sentences=[sentence])

        # Mock Model & Tokenizer
        mock_model = MagicMock()
        mock_model.model_instance.tokenize.return_value = [1, 2, 3]
        mocks["adapter"].load.return_value = mock_model

        # Mock LLM Patch output
        mocks["adapter"].generate_patches.return_value = [
            PhoneticPatch(original="a cute angina", correction="acute angina")
        ]

        # Mock Repo Save
        mocks["repo"].save.side_effect = lambda x: x

        # FIX: Replaced string sub_domain with a list
        result = api.execute(transcript, "MEDICAL", ["CARDIOLOGY"])

        # Verify logic
        assert len(result.sentences) == 1
        assert result.sentences[0].text == "He has acute angina."
        # Assert patch applied
        assert len(result.audit_trail) == 1

        # Verify strict VRAM control
        mocks["adapter"].load.assert_called_once()
        mocks["adapter"].unload.assert_called_once_with(mock_model)