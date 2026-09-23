# Path: integration_tests/features/transcription/test_language_guard_integration.py

import os
import uuid
import wave
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.core.config import settings
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator
from app.features.transcription.data.faster_whisper_adapter import FasterWhisperAdapter
from app.features.transcription.exceptions import LanguageMismatchError
from app.features.transcription.service.transcription_api import TranscriptionAPI
from app.features.vad.domain.entities import VadMap, VadSegment

ENGLISH_SPEECH = Path(__file__).resolve().parents[2] / "resources" / "test_speech.wav"


@pytest.mark.integration
class TestLanguageGuardIntegration:
    """
    Real Whisper large-v3 on real English speech. Requires an NVIDIA GPU and the Whisper weights; no database
    (the repositories are mocked, since a rejected file must never reach them).
    """

    @pytest.fixture
    def speech(self):
        if not os.path.exists(settings.WHISPER_MODEL_PATH):
            pytest.skip(f"Whisper weights not found at {settings.WHISPER_MODEL_PATH}.")
        if not ENGLISH_SPEECH.exists():
            pytest.skip(f"Test audio not found at {ENGLISH_SPEECH}.")
        with wave.open(str(ENGLISH_SPEECH)) as audio:
            duration = audio.getnframes() / audio.getframerate()
        return VadMap(source_id=uuid.uuid4(), segments=[VadSegment(start_sec=0.0, end_sec=duration, confidence=0.9)])

    @pytest.fixture
    def api_parts(self, speech):
        repo = MagicMock(**{"get_by_source_id.return_value": None, "save.side_effect": lambda t: t})
        vad_repo = MagicMock(**{"get_map_by_source_id.return_value": speech})
        orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
        return TranscriptionAPI(repo, vad_repo, orchestrator, FasterWhisperAdapter()), repo, orchestrator

    def test_english_speech_is_detected_as_english(self, speech):
        orchestrator = ModelManagerOrchestrator(PyTorchVramMonitor())
        adapter = FasterWhisperAdapter()
        loaded = orchestrator.acquire_model(ModelManifest(
            model_type=ModelType.TRANSCRIPTION, model_path=settings.WHISPER_MODEL_PATH,
            required_vram_mb=settings.WHISPER_VRAM_REQUIREMENT_MB
        ), adapter)
        try:
            language, probability = adapter.detect_language(loaded, str(ENGLISH_SPEECH), speech)
        finally:
            orchestrator.release_model(loaded, adapter)

        assert language == "en"
        assert probability > 0.9

    def test_english_file_on_a_serbian_appliance_is_rejected_not_transcribed(self, api_parts, monkeypatch):
        """Forced to Serbian, Whisper would translate or invent Serbian text; the file must fail instead."""
        monkeypatch.setattr(settings, "APPLIANCE_LANGUAGE", "sr")
        api, repo, orchestrator = api_parts

        with pytest.raises(LanguageMismatchError, match="'en'"):
            api.process_audio(uuid.uuid4(), str(ENGLISH_SPEECH), release_after=True)

        repo.save.assert_not_called()
        assert not orchestrator._active_models  # released despite the rejection
