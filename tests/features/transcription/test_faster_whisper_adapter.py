# filename: test_faster_whisper_adapter_v1.py
# path: tests/features/transcription/test_faster_whisper_adapter.py
import uuid
import pytest
from unittest.mock import patch, MagicMock
import torch

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
        # Verify INT8 and offline flags were strictly passed
        mock_whisper_cls.assert_called_once_with(
            model_size_or_path=manifest.model_path,
            device="cuda",
            compute_type="int8",
            local_files_only=True
        )

    @patch("os.path.exists")
    def test_load_missing_file_raises_error(self, mock_exists, manifest):
        mock_exists.return_value = False
        adapter = FasterWhisperAdapter()

        with pytest.raises(TranscriptionModelFilePathError):
            adapter.load(manifest)

    @patch("app.features.transcription.data.faster_whisper_adapter.torchaudio")
    def test_transcribe_with_vad_merging(self, mock_torchaudio, manifest):
        """Verifies the adapter correctly slices audio and offsets timestamps."""
        # 1. Setup Audio Mock (10 seconds of 16kHz audio)
        sample_rate = 16000
        dummy_waveform = torch.zeros(1, sample_rate * 10)
        mock_torchaudio.load.return_value = (dummy_waveform, sample_rate)

        # 2. Setup VAD Map (Two segments close enough to trigger the merge logic)
        vad_map = VadMap(
            source_id=uuid.uuid4(),
            segments=[
                VadSegment(start_sec=1.0, end_sec=2.0, confidence=0.9),
                VadSegment(start_sec=2.5, end_sec=4.0, confidence=0.9)
            ]
        )

        # 3. Setup Mock Whisper Model
        mock_whisper_instance = MagicMock()

        # Whisper yields segments as a generator
        mock_whisper_segment = MagicMock()
        mock_whisper_segment.start = 0.5  # Relative to the merged audio chunk
        mock_whisper_segment.end = 1.5
        mock_whisper_segment.text = "Hello world"

        def dummy_transcribe_generator(*args, **kwargs):
            yield mock_whisper_segment

        mock_whisper_instance.transcribe.return_value = (dummy_transcribe_generator(), MagicMock())
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_whisper_instance)

        adapter = FasterWhisperAdapter()

        # 4. Execute
        transcript = adapter.transcribe(
            model=loaded_model,
            audio_path="/mock/audio.wav",
            vad_map=vad_map,
            source_id=vad_map.source_id
        )

        # 5. Assertions
        assert transcript.source_id == vad_map.source_id
        assert len(transcript.segments) == 1
        assert transcript.segments[0].text == "Hello world"

        # Verify WhisperModel.transcribe was only called ONCE due to the < 2.0s gap merge logic
        mock_whisper_instance.transcribe.assert_called_once()