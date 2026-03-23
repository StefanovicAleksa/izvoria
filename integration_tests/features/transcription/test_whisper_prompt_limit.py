# Canvas: test_whisper_prompt_limit_integration_v1
# Path: integration_tests/features/transcription/test_whisper_prompt_limit.py

import os
import tempfile
import uuid
import pytest
import torch
import torchaudio

from app.core.config import settings
from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.features.transcription.data.faster_whisper_adapter import FasterWhisperAdapter
from app.features.vad.domain.entities import VadMap, VadSegment


class TestFasterWhisperPromptLimitIntegration:
    """
    STRICT NO-MOCKS INTEGRATION TEST.
    Physically tests the CTranslate2 engine's token limits to ensure
    massive Domain Adaptation prompts do not crash the pipeline.
    """

    @pytest.fixture(scope="module", autouse=True)
    def ensure_model_exists(self):
        if not os.path.exists(settings.WHISPER_MODEL_PATH):
            pytest.skip(f"Skipping test due to missing Whisper model at: {settings.WHISPER_MODEL_PATH}")

    def test_transcribe_massively_oversized_prompt(self):
        adapter = FasterWhisperAdapter()

        manifest = ModelManifest(
            model_type=ModelType.TRANSCRIPTION,
            model_path=settings.WHISPER_MODEL_PATH,
            required_vram_mb=settings.WHISPER_VRAM_REQUIREMENT_MB
        )

        loaded_model = None
        temp_audio_path = None

        try:
            # 1. Create a dummy 1-second audio file on disk
            sample_rate = 16000
            waveform = torch.zeros(1, sample_rate * 1)
            temp_dir = tempfile.gettempdir()
            temp_audio_path = os.path.join(temp_dir, "test_whisper_silence.wav")
            torchaudio.save(temp_audio_path, waveform, sample_rate)

            # 2. Setup VAD map for the 1-second file
            vad_map = VadMap(
                source_id=uuid.uuid4(),
                segments=[VadSegment(start_sec=0.0, end_sec=1.0, confidence=0.99)]
            )

            # 3. Create a massive 500-word prompt (Easily 1000+ tokens, way over the 224 limit)
            massive_prompt = "Quantitative Easing " * 250

            # 4. Load the real model into VRAM
            loaded_model = adapter.load(manifest)

            # 5. Execute - If the adapter fails to truncate, CTranslate2 will throw ValueError!
            segments = adapter.transcribe(
                model=loaded_model,
                audio_path=temp_audio_path,
                vad_map=vad_map,
                language="en",
                initial_prompt=massive_prompt
            )

            # If we reach here, the truncation worked perfectly!
            assert isinstance(segments, list)
            print("\n[SUCCESS] Faster-Whisper safely handled the massive prompt overflow without crashing!")

        finally:
            # Cleanup VRAM and Temp Disk
            if loaded_model:
                adapter.unload(loaded_model)
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)