# Canvas: test_transcription_api_unit_v3
# Path: tests/features/transcription/test_transcription_api_unit.py

import uuid
import pytest
from unittest.mock import MagicMock

from app.features.transcription.service.transcription_api import TranscriptionAPI
from app.features.transcription.domain.entities import Transcript
from app.features.vad.domain.entities import VadMap
from app.features.transcription.exceptions.model import (
    TranscriptionAudioError,
    CTranslateEngineError
)


class TestTranscriptionAPI:
    @pytest.fixture
    def mocks(self):
        return {
            "repo": MagicMock(),
            "vad_repo": MagicMock(),
            "orchestrator": MagicMock(),
            "adapter": MagicMock()
        }

    @pytest.fixture
    def api(self, mocks):
        return TranscriptionAPI(
            repository=mocks["repo"],
            vad_repository=mocks["vad_repo"],
            orchestrator=mocks["orchestrator"],
            adapter=mocks["adapter"]
        )

    def test_get_transcript(self, api, mocks):
        source_id = uuid.uuid4()
        fake_transcript = Transcript(source_id=source_id, segments=[])
        mocks["repo"].get_by_source_id.return_value = fake_transcript

        result = api.get_transcript(source_id)

        assert result == fake_transcript
        mocks["repo"].get_by_source_id.assert_called_once_with(source_id)

    def test_update_transcript(self, api, mocks):
        source_id = uuid.uuid4()
        fake_transcript = Transcript(source_id=source_id, segments=[])
        mocks["repo"].update.return_value = fake_transcript

        result = api.update_transcript(fake_transcript)

        assert result == fake_transcript
        mocks["repo"].update.assert_called_once_with(fake_transcript)

    def test_process_audio_idempotent_return(self, api, mocks):
        """Should return immediately if transcript already exists in DB."""
        source_id = uuid.uuid4()
        existing_transcript = Transcript(source_id=source_id, segments=[])
        mocks["repo"].get_by_source_id.return_value = existing_transcript

        result = api.process_audio(source_id, "/dummy.wav")

        assert result == existing_transcript
        mocks["vad_repo"].get_map_by_source_id.assert_not_called()
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_process_audio_fails_without_vad_map(self, api, mocks):
        """Cannot transcribe without the VAD map prerequisites."""
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None
        mocks["vad_repo"].get_map_by_source_id.return_value = None

        with pytest.raises(TranscriptionAudioError, match="VAD Map not found"):
            api.process_audio(source_id, "/dummy.wav")

    def test_process_audio_success_flow(self, api, mocks):
        """Happy path verifies Orchestrator loading, execution, saving, and VRAM flushing."""
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None

        fake_vad_map = VadMap(source_id=source_id, segments=[])
        mocks["vad_repo"].get_map_by_source_id.return_value = fake_vad_map

        loaded_model_mock = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded_model_mock

        generated_transcript = Transcript(source_id=source_id, segments=[])
        mocks["adapter"].transcribe.return_value = generated_transcript
        mocks["repo"].save.return_value = generated_transcript

        result = api.process_audio(source_id, "/dummy.wav", release_after=True)

        assert result == generated_transcript
        mocks["orchestrator"].acquire_model.assert_called_once()
        mocks["adapter"].transcribe.assert_called_once_with(
            model=loaded_model_mock, audio_path="/dummy.wav", vad_map=fake_vad_map, source_id=source_id
        )
        mocks["repo"].save.assert_called_once_with(generated_transcript)
        mocks["orchestrator"].release_model.assert_called_once_with(
            loaded_model_mock, mocks["adapter"], flush=True
        )

    def test_process_audio_guarantees_vram_release_on_crash(self, api, mocks):
        """Even if the model throws a CTranslate runtime error, finally block must release VRAM."""
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None
        mocks["vad_repo"].get_map_by_source_id.return_value = VadMap(source_id=source_id, segments=[])

        loaded_model_mock = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded_model_mock

        mocks["adapter"].transcribe.side_effect = RuntimeError("cublas failed")

        with pytest.raises(CTranslateEngineError):
            api.process_audio(source_id, "/dummy.wav", release_after=True)

        mocks["orchestrator"].release_model.assert_called_once_with(
            loaded_model_mock, mocks["adapter"], flush=True
        )
        mocks["repo"].save.assert_not_called()