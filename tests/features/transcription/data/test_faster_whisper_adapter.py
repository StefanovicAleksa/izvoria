# Canvas: test_faster_whisper_adapter_v7
# Path: tests/features/transcription/test_faster_whisper_adapter.py

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

    @patch("app.features.transcription.data.faster_whisper_adapter.torchaudio")
    def test_transcribe_with_vad_merging_and_hotwords(self, mock_torchaudio, manifest):
        """Verifies the adapter correctly slices audio, merges VAD, requests word timestamps, and offsets them."""
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

        # Setup mock word
        mock_whisper_word = MagicMock()
        mock_whisper_word.start = 0.5  # Relative to the chunk
        mock_whisper_word.end = 1.0
        mock_whisper_word.word = "Hello"
        mock_whisper_word.probability = 0.99

        # Setup mock segment containing the word
        mock_whisper_segment = MagicMock()
        mock_whisper_segment.start = 0.5
        mock_whisper_segment.end = 1.5
        mock_whisper_segment.text = "Hello world"
        mock_whisper_segment.words = [mock_whisper_word]

        def dummy_transcribe_generator(*args, **kwargs):
            yield mock_whisper_segment

        # Because the segments merge (gap is 0.5s <= 2.0s threshold), transcribe is only called ONCE!
        mock_whisper_instance.transcribe.return_value = (dummy_transcribe_generator(), MagicMock())
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_whisper_instance)

        adapter = FasterWhisperAdapter()

        # 4. Execute
        segments_out = adapter.transcribe(
            model=loaded_model,
            audio_path="/mock/audio.wav",
            vad_map=vad_map,
            language="en",
            hotwords="EBITDA:15.0"
        )

        # 5. Assertions
        assert isinstance(segments_out, list)
        assert len(segments_out) == 1
        assert segments_out[0].text == "Hello world"

        # Start sec should be chunk_start (1.0) + segment_start (0.5) = 1.5
        assert segments_out[0].start_sec == 1.5

        # Verify word-level timestamps were properly mapped and offset
        assert len(segments_out[0].words) == 1
        assert segments_out[0].words[0].word == "Hello"
        assert segments_out[0].words[0].start_sec == 1.5  # 1.0 + 0.5
        assert segments_out[0].words[0].end_sec == 2.0    # 1.0 + 1.0
        assert segments_out[0].words[0].probability == 0.99

        # Verify CTranslate2 received our exact language, domain hints, raw audio, and word_timestamps flag
        mock_whisper_instance.transcribe.assert_called_once()
        call_kwargs = mock_whisper_instance.transcribe.call_args[1]
        assert call_kwargs["language"] == "en"
        assert call_kwargs["hotwords"] == "EBITDA:15.0"
        assert call_kwargs["word_timestamps"] is True
        assert "audio" in call_kwargs