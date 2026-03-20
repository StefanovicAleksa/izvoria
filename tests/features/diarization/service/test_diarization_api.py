# Canvas: test_diarization_api_unit_v1
# Path: tests/features/diarization/test_diarization_api.py

import uuid
import pytest
from unittest.mock import MagicMock, patch

from app.features.diarization.service.diarization_api import DiarizationAPI
from app.features.diarization.domain.entities import DiarizationMap
from app.features.transcription.domain.entities import Transcript
from app.features.vad.domain.entities import VadMap
from app.features.diarization.exceptions.model import DiarizationManifestError, DiarizationInferenceError


class TestDiarizationAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "repo": MagicMock(),
            "vad_repo": MagicMock(),
            "transcription_api": MagicMock(),
            "storage_api": MagicMock(),
            "orchestrator": MagicMock(),
            "adapter": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return DiarizationAPI(
            repository=mocks["repo"],
            vad_repository=mocks["vad_repo"],
            transcription_api=mocks["transcription_api"],
            storage_api=mocks["storage_api"],
            orchestrator=mocks["orchestrator"],
            adapter=mocks["adapter"]
        )

    def test_process_audio_idempotent_return(self, api, mocks):
        """Should return immediately if diarization map already exists in DB."""
        source_id = uuid.uuid4()
        existing_map = DiarizationMap(source_id=source_id, segments=[])
        mocks["repo"].get_map_by_source_id.return_value = existing_map

        result = api.process_audio(source_id, "/dummy.wav")

        assert result == existing_map
        mocks["vad_repo"].get_map_by_source_id.assert_not_called()
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_process_audio_fails_without_vad_map(self, api, mocks):
        """Cannot diarize without the VAD map prerequisites."""
        source_id = uuid.uuid4()
        mocks["repo"].get_map_by_source_id.return_value = None
        mocks["vad_repo"].get_map_by_source_id.return_value = None

        with pytest.raises(DiarizationManifestError, match="VAD Map not found"):
            api.process_audio(source_id, "/dummy.wav")

    def test_process_audio_fails_without_transcript(self, api, mocks):
        """Cannot run the alignment phase if the Transcript does not exist."""
        source_id = uuid.uuid4()
        mocks["repo"].get_map_by_source_id.return_value = None
        mocks["vad_repo"].get_map_by_source_id.return_value = VadMap(source_id=source_id, segments=[])
        mocks["transcription_api"].get_transcript.return_value = None

        with pytest.raises(DiarizationManifestError, match="Whisper Transcript not found"):
            api.process_audio(source_id, "/dummy.wav")

    @patch("app.features.diarization.service.diarization_api.align_transcript_with_diarization")
    def test_process_audio_success_flow(self, mock_aligner, api, mocks):
        """Happy path verifies Orchestrator loading, execution, saving, alignment, and VRAM flushing."""
        source_id = uuid.uuid4()
        mocks["repo"].get_map_by_source_id.return_value = None

        fake_vad_map = VadMap(source_id=source_id, segments=[])
        fake_transcript = Transcript(source_id=source_id, segments=[])

        mocks["vad_repo"].get_map_by_source_id.return_value = fake_vad_map
        mocks["transcription_api"].get_transcript.return_value = fake_transcript

        loaded_model_mock = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded_model_mock

        generated_map = DiarizationMap(source_id=source_id, segments=[])
        mocks["adapter"].generate_diarization_map.return_value = generated_map
        mocks["repo"].save_map.return_value = generated_map

        aligned_transcript_mock = MagicMock()
        mock_aligner.return_value = aligned_transcript_mock

        # Execute
        result = api.process_audio(source_id, "/dummy.wav", release_after=True)

        assert result == generated_map

        # Verify Orchestrator
        mocks["orchestrator"].acquire_model.assert_called_once()
        mocks["orchestrator"].release_model.assert_called_once_with(
            loaded_model_mock, mocks["adapter"], flush=True
        )

        # Verify DB updates
        mocks["repo"].save_map.assert_called_once_with(generated_map)

        # Verify cross-feature service communication
        mock_aligner.assert_called_once_with(fake_transcript, generated_map)
        mocks["transcription_api"].update_transcript.assert_called_once_with(aligned_transcript_mock)

    def test_process_audio_guarantees_vram_release_on_crash(self, api, mocks):
        """Even if the model throws an inference error, finally block must release VRAM."""
        source_id = uuid.uuid4()
        mocks["repo"].get_map_by_source_id.return_value = None
        mocks["vad_repo"].get_map_by_source_id.return_value = VadMap(source_id=source_id, segments=[])
        mocks["transcription_api"].get_transcript.return_value = Transcript(source_id=source_id, segments=[])

        loaded_model_mock = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded_model_mock

        # Simulate NeMo engine crash
        mocks["adapter"].generate_diarization_map.side_effect = RuntimeError("CUDA Out of memory")

        with pytest.raises(DiarizationInferenceError):
            api.process_audio(source_id, "/dummy.wav", release_after=True)

        # Flush must still happen to prevent batch lockups
        mocks["orchestrator"].release_model.assert_called_once_with(
            loaded_model_mock, mocks["adapter"], flush=True
        )
        mocks["repo"].save_map.assert_not_called()