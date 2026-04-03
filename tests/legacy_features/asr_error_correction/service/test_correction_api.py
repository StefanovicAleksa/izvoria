# Canvas: test_correction_api_v10
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
    @patch("app.features.phonetic_correction.service.correction_api.split_massive_sentences")
    def test_execute_applies_patches_and_unloads_vram(self, mock_splitter, mock_dict_loader, api, mock_repo,
                                                      mock_orchestrator, mock_adapter):
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

        # Ensure the pre-processor utility passes the sentence through
        mock_splitter.return_value = [dummy_sentence]

        mock_loaded_model = MagicMock()
        mock_orchestrator.acquire_model.return_value = mock_loaded_model

        mock_adapter.generate_patches.return_value = [
            PhoneticPatch(original="a cute angina", correction="acute angina")
        ]

        mock_repo.save.side_effect = lambda x: x

        result = api.execute(dummy_transcript, "MEDICAL", [], release_after=True)

        # FIXED: Updated to 8,000 characters to match the current optimized implementation
        mock_splitter.assert_called_once_with(dummy_transcript.sentences, max_chars=8000)
        mock_orchestrator.acquire_model.assert_called_once()
        mock_adapter.generate_patches.assert_called_once()
        mock_orchestrator.release_model.assert_called_once_with(mock_loaded_model, mock_adapter, flush=True)

        assert "acute angina" in result.sentences[0].text
        assert len(result.audit_trail) == 1
        assert result.audit_trail[0].original_text == "I have a cute angina."
        assert result.audit_trail[0].corrected_text == "I have acute angina."

    def test_apply_patches_to_chunk_safely_handles_regex_hallucinations(self):
        """
        Proves that the re.sub lambda fix prevents Python from fatally crashing
        if the LLM hallucinates an escaped regex group like \\1.
        """
        sentence_id = uuid.uuid4()
        sentences = [
            ReconstructedSentence(
                sentence_id=sentence_id,
                start_sec=0,
                end_sec=1,
                text="The patient has acute angina.",
                speaker_id="S0",
                words=[]
            )
        ]

        # If Qwen hallucinates a regex backreference like \\1
        patches = [
            PhoneticPatch(original="acute angina", correction=r"\\1")
        ]

        # FIXED: Passed a Dictionary instead of a List to match the stateful deduplication logic
        audit_records_map = {}

        result = PhoneticCorrectionAPI._apply_patches_to_chunk(sentences, patches, audit_records_map)

        # The text was successfully updated without parsing the \\1 as a capture group
        assert result[0].text == r"The patient has \\1."
        assert len(audit_records_map) == 1
        assert audit_records_map[sentence_id].corrected_text == r"The patient has \\1."

    def test_apply_patches_deduplicates_overlapping_signatures(self):
        """
        Regression Test: Ensures that if an overlapping window generates the same patch
        for the same sentence, the API deduplicates it using signatures.
        """
        sentence_id = uuid.uuid4()
        sentences = [
            ReconstructedSentence(
                sentence_id=sentence_id,
                start_sec=0,
                end_sec=1,
                text="The patient has a cute angina.",
                speaker_id="S0",
                words=[]
            )
        ]

        # FIXED: Variable renamed to 'target_patch' to prevent shadowing 'unittest.mock.patch'
        target_patch = PhoneticPatch(original="a cute angina", correction="acute angina")
        audit_records_map = {}

        # First application
        result_1 = PhoneticCorrectionAPI._apply_patches_to_chunk(sentences, [target_patch], audit_records_map)
        assert result_1[0].text == "The patient has acute angina."
        assert len(audit_records_map[sentence_id].applied_patches) == 1

        # Second application (Simulating an overlapping chunk seeing the same error)
        # Even though we provide the patch again, the signature check should skip it.
        result_2 = PhoneticCorrectionAPI._apply_patches_to_chunk(result_1, [target_patch], audit_records_map)

        assert result_2[0].text == "The patient has acute angina."
        # Verify no duplicate patches or corrupted "acute acute angina" text
        assert len(audit_records_map[sentence_id].applied_patches) == 1