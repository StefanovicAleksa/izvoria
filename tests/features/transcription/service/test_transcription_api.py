# Canvas: test_transcription_api_v6
# Path: tests/features/transcription/service/test_transcription_api.py

import uuid
import pytest
from unittest.mock import MagicMock

from app.features.transcription.service.transcription_api import TranscriptionAPI
from app.features.transcription.domain.entities import Transcript, TranscriptSegment, TranscriptWord
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.transcription.exceptions.model import CTranslateEngineError


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

    def test_process_audio_idempotent_return(self, api, mocks):
        source_id = uuid.uuid4()
        existing_transcript = Transcript(source_id=source_id, segments=[])
        mocks["repo"].get_by_source_id.return_value = existing_transcript

        result = api.process_audio(source_id, "/dummy.wav")

        assert result == existing_transcript
        mocks["vad_repo"].get_map_by_source_id.assert_not_called()
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_process_audio_fails_without_vad_map(self, api, mocks):
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None
        mocks["vad_repo"].get_map_by_source_id.return_value = None

        empty_transcript = Transcript(source_id=source_id, segments=[])
        mocks["repo"].save.return_value = empty_transcript

        result = api.process_audio(source_id, "/dummy.wav")
        assert result == empty_transcript
        mocks["orchestrator"].acquire_model.assert_not_called()

    def test_process_audio_success_flow_with_hints(self, api, mocks):
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None

        fake_vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=0, end_sec=5.0, confidence=0.99)])
        mocks["vad_repo"].get_map_by_source_id.return_value = fake_vad_map

        loaded_model_mock = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded_model_mock

        generated_segments = []
        mocks["adapter"].transcribe.return_value = generated_segments

        saved_transcript = Transcript(source_id=source_id, segments=[])
        mocks["repo"].save.return_value = saved_transcript

        result = api.process_audio(
            source_id,
            "/dummy.wav",
            language="fr",
            hotwords="bonjour:15.0",
            initial_prompt="French Context Primer",
            release_after=True
        )

        assert result == saved_transcript
        mocks["orchestrator"].acquire_model.assert_called_once()

        # Verifies domain adaptation payload is passed securely to the whisper engine
        mocks["adapter"].transcribe.assert_called_once_with(
            model=loaded_model_mock,
            audio_path="/dummy.wav",
            vad_map=fake_vad_map,
            language="fr",
            hotwords="bonjour:15.0",
            initial_prompt="French Context Primer"
        )
        mocks["repo"].save.assert_called_once()
        mocks["orchestrator"].release_model.assert_called_once_with(
            loaded_model_mock, mocks["adapter"], flush=True
        )

    def test_process_audio_removes_text_whisper_copied_from_its_hotwords(self, api, mocks):
        """Whisper sometimes repeats its primed hotwords as if they were spoken; that text must never be saved."""
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None
        mocks["vad_repo"].get_map_by_source_id.return_value = VadMap(
            source_id=source_id, segments=[VadSegment(start_sec=0, end_sec=9.0, confidence=0.99)]
        )
        mocks["repo"].save.side_effect = lambda transcript: transcript

        def word(text: str, start: float) -> TranscriptWord:
            return TranscriptWord(start_sec=start, end_sec=start + 0.4, word=text)

        spoken = TranscriptSegment(start_sec=0.0, end_sec=1.6, text="The ECB held rates.",
                                   words=[word(" The", 0.0), word(" ECB", 0.4), word(" held", 0.8), word(" rates.", 1.2)])
        echoed = TranscriptSegment(start_sec=5.0, end_sec=7.0, text="ECB, Lagarde, Frankfurt, Bunds, BOE.",
                                   words=[word(" ECB,", 5.0), word(" Lagarde,", 5.4), word(" Frankfurt,", 5.8),
                                          word(" Bunds,", 6.2), word(" BOE.", 6.6)])
        mocks["adapter"].transcribe.return_value = [spoken, echoed]

        result = api.process_audio(source_id, "/dummy.wav", hotwords="ECB, Lagarde, Frankfurt, Bunds, BOE, Bailey")

        assert [s.text for s in result.segments] == ["The ECB held rates."]
        mocks["repo"].save.assert_called_once_with(result)

    def test_process_audio_guarantees_vram_release_on_crash(self, api, mocks):
        source_id = uuid.uuid4()
        mocks["repo"].get_by_source_id.return_value = None

        fake_vad_map = VadMap(source_id=source_id, segments=[VadSegment(start_sec=0, end_sec=5.0, confidence=0.99)])
        mocks["vad_repo"].get_map_by_source_id.return_value = fake_vad_map

        loaded_model_mock = MagicMock()
        mocks["orchestrator"].acquire_model.return_value = loaded_model_mock

        mocks["adapter"].transcribe.side_effect = RuntimeError("cublas failed")

        with pytest.raises(CTranslateEngineError):
            api.process_audio(source_id, "/dummy.wav", release_after=True)

        mocks["orchestrator"].release_model.assert_called_once_with(
            loaded_model_mock, mocks["adapter"], flush=True
        )
        mocks["repo"].save.assert_not_called()