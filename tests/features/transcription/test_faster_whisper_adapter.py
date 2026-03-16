# Canvas: test_faster_whisper_adapter_v4
# Path: tests/features/transcription/test_faster_whisper_adapter.py

import uuid
import pytest
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.vad.domain.entities import VadMap, VadSegment
from app.features.transcription.data.faster_whisper_adapter import FasterWhisperAdapter
from app.features.transcription.exceptions.model import TranscriptionModelFilePathError


class TestFasterWhisperAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.TRANSCRIPTION,
            model_path="/mock/path/faster-whisper-large-v3",
            required_vram_mb=3500.0
        )

    @patch("app.features.transcription.data.faster_whisper_adapter.WhisperModel")
    @patch("os.path.exists")
    def test_load_success(self, mock_exists, mock_whisper_cls, manifest):
        mock_exists.return_value = True
        mock_instance = MagicMock()
        mock_whisper_cls.return_value = mock_instance

        adapter = FasterWhisperAdapter()
        loaded_model = adapter.load(manifest)

        assert loaded_model.manifest == manifest
        assert loaded_model.model_instance == mock_instance
        mock_whisper_cls.assert_called_once_with(
            model_size_or_path=manifest.model_path,
            device="cuda",
            compute_type="float16",
            local_files_only=True
        )

    @patch("os.path.exists")
    def test_load_missing_file_raises_error(self, mock_exists, manifest):
        mock_exists.return_value = False
        adapter = FasterWhisperAdapter()

        with pytest.raises(TranscriptionModelFilePathError):
            adapter.load(manifest)

    def test_transcribe_with_vad_merging_and_hotwords(self, manifest):
        """Verifies the adapter correctly slices audio and passes domain hints to Whisper."""
        # 1. Setup VAD Map
        vad_map = VadMap(
            source_id=uuid.uuid4(),
            segments=[
                VadSegment(start_sec=1.0, end_sec=2.0, confidence=0.9),
                VadSegment(start_sec=2.5, end_sec=4.0, confidence=0.9)
            ]
        )

        # 2. Setup Mock Whisper Model
        mock_whisper_instance = MagicMock()
        mock_whisper_segment = MagicMock()
        mock_whisper_segment.start = 0.5
        mock_whisper_segment.end = 1.5
        mock_whisper_segment.text = "Hello world"

        def dummy_transcribe_generator(*args, **kwargs):
            yield mock_whisper_segment

        # FIX: Use a lambda with side_effect to spawn a fresh generator for every VAD segment loop
        mock_whisper_instance.transcribe.side_effect = lambda *args, **kwargs: (dummy_transcribe_generator(),
                                                                                MagicMock())

        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_whisper_instance)

        adapter = FasterWhisperAdapter()

        # 3. Execute
        segments_out = adapter.transcribe(
            model=loaded_model,
            audio_path="/mock/audio.wav",
            vad_map=vad_map,
            language="en",
            hotwords="EBITDA:15.0"
        )

        # 4. Assertions
        assert isinstance(segments_out, list)
        assert len(segments_out) == 2  # The second generator execution will now succeed!
        assert segments_out[0].text == "Hello world"

        mock_whisper_instance.transcribe.assert_called_with(
            audio="/mock/audio.wav",
            language="en",
            vad_filter=False,
            hotwords="EBITDA:15.0",
            clip_timestamps=[2.5, 4.0]
        )