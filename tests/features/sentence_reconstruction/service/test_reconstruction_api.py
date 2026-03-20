# Canvas: test_reconstruction_api_unit_v2
# Path: tests/features/sentence_reconstruction/service/test_reconstruction_api.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.features.transcription.domain.entities import Transcript, TranscriptSegment, TranscriptWord
from app.features.sentence_reconstruction.service.reconstruction_api import SentenceReconstructionAPI
from app.features.sentence_reconstruction.domain.entities import ReconstructedTranscript, ReconstructedSentence
from app.features.sentence_reconstruction.exceptions.assembly import MissingDiarizationError
from app.features.sentence_reconstruction.exceptions.model import PunctuationExecutionError


class TestSentenceReconstructionAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "repo": MagicMock(),
            "orchestrator": MagicMock(),
            "adapter": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return SentenceReconstructionAPI(
            repository=mocks["repo"],
            orchestrator=mocks["orchestrator"],
            adapter=mocks["adapter"]
        )

    def test_reconstruct_transcript_idempotent_return(self, api, mocks):
        """Should return immediately if transcript already exists in DB."""
        source_id = uuid.uuid4()
        existing_transcript = ReconstructedTranscript(source_id=source_id, sentences=[])
        mocks["repo"].get_by_source_id.return_value = existing_transcript

        result = api.reconstruct_transcript(Transcript(source_id=source_id, segments=[]))

        assert result == existing_transcript
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_reconstruct_transcript_empty_audio(self, api, mocks):
        """Handles edge case where VAD found no human speech."""
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None

        empty_reconstructed = ReconstructedTranscript(source_id=source_id, sentences=[])
        mocks["repo"].save.return_value = empty_reconstructed

        result = api.reconstruct_transcript(Transcript(source_id=source_id, segments=[]))

        assert result == empty_reconstructed
        mocks["repo"].save.assert_called_once()
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_reconstruct_transcript_fails_without_diarization(self, api, mocks):
        """Phase 7 must enforce the Diarization Speaker Wall."""
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None

        # Transcript with an 'Unknown' speaker
        bad_transcript = Transcript(
            source_id=source_id,
            segments=[TranscriptSegment(start_sec=0, end_sec=1, text="Hello", speaker_id="Unknown", words=[])]
        )

        loaded_model_mock = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded_model_mock

        with pytest.raises(MissingDiarizationError, match="Cannot reconstruct sentences without speaker IDs"):
            api.reconstruct_transcript(bad_transcript)

        # Ensure VRAM is still released even though it failed the data validation
        mocks["orchestrator"].release_model.assert_called_once_with(
            loaded_model_mock, mocks["adapter"], flush=True
        )

    @patch("app.features.sentence_reconstruction.service.reconstruction_api.assemble_sentences")
    def test_reconstruct_transcript_success_flow(self, mock_assembler, api, mocks):
        """Happy path verifies Orchestrator loading, AI tagging, Assembly, and saving."""
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None

        # 1. Setup Input Transcript
        words = [TranscriptWord(start_sec=0, end_sec=1, word="test", probability=0.99)]
        transcript = Transcript(
            source_id=source_id,
            segments=[TranscriptSegment(start_sec=0, end_sec=1, text="test", speaker_id="Speaker_0", words=words)]
        )

        # 2. Setup GPU Mocks
        loaded_model_mock = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded_model_mock

        # AI tags the words
        tagged_words = [MagicMock()]
        mocks["adapter"].predict_punctuation.return_value = tagged_words

        # Pure Python utility builds the sentence
        assembled_sentence = ReconstructedSentence(
            sentence_id=uuid.uuid4(), text="Test.", speaker_id="Speaker_0",
            start_sec=0.0, end_sec=1.0, words=words
        )
        mock_assembler.return_value = [assembled_sentence]

        # 3. Setup DB Mock
        final_transcript = ReconstructedTranscript(source_id=source_id, sentences=[assembled_sentence])
        mocks["repo"].save.return_value = final_transcript

        # 4. Execute
        result = api.reconstruct_transcript(transcript, release_after=True)

        # 5. Assertions
        assert result == final_transcript

        # Verify GPU lifecycle
        mocks["orchestrator"].acquire_model.assert_called_once()
        mocks["adapter"].predict_punctuation.assert_called_once_with(loaded_model_mock, words)
        mocks["orchestrator"].release_model.assert_called_once_with(
            loaded_model_mock, mocks["adapter"], flush=True
        )

        # Verify DB Save
        mocks["repo"].save.assert_called_once()

    def test_reconstruct_transcript_guarantees_vram_release_on_crash(self, api, mocks):
        """Even if the Hugging Face pipeline throws an OOM or dimension error, finally block must release VRAM."""
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None

        # CRITICAL FIX: Add a dummy word so the inference block is actually executed
        dummy_word = TranscriptWord(start_sec=0.0, end_sec=1.0, word="test", probability=0.99)
        transcript = Transcript(
            source_id=source_id,
            segments=[
                TranscriptSegment(start_sec=0, end_sec=1, text="test", speaker_id="Speaker_0", words=[dummy_word])]
        )

        loaded_model_mock = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded_model_mock

        # Simulate Pipeline crash
        mocks["adapter"].predict_punctuation.side_effect = RuntimeError("Tensor dimension mismatch")

        with pytest.raises(PunctuationExecutionError):
            api.reconstruct_transcript(transcript, release_after=True)

        # VRAM MUST BE FLUSHED
        mocks["orchestrator"].release_model.assert_called_once_with(
            loaded_model_mock, mocks["adapter"], flush=True
        )
        mocks["repo"].save.assert_not_called()