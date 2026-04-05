# Canvas: test_faster_whisper_adapter_v6
# Path: tests/features/transcription/data/test_faster_whisper_adapter.py

import uuid
import pytest
import torch
from unittest.mock import patch, MagicMock

from app.core.model_manager.domain.entities import ModelManifest, LoadedModel
from app.core.model_manager.domain.enums import ModelType
from app.features.transcription.data.faster_whisper_adapter import FasterWhisperAdapter
from app.features.vad.domain.entities import VadMap, VadSegment


class TestFasterWhisperAdapter:
    @pytest.fixture
    def manifest(self):
        return ModelManifest(
            model_type=ModelType.TRANSCRIPTION,
            model_path="/mock/path/faster-whisper-large-v3",
            required_vram_mb=4000.0
        )

    # CRITICAL FIX: Patch the static factory method instead of the global WhisperModel
    @patch("os.path.exists", return_value=True)
    @patch.object(FasterWhisperAdapter, "_get_whisper_model")
    def test_load_success(self, mock_get_whisper, mock_exists, manifest):
        mock_instance = MagicMock()
        mock_get_whisper.return_value = mock_instance

        adapter = FasterWhisperAdapter()
        loaded_model = adapter.load(manifest)

        assert loaded_model.model_instance == mock_instance
        mock_get_whisper.assert_called_once_with(manifest.model_path)

    @patch("app.features.transcription.data.faster_whisper_adapter.torchaudio")
    def test_transcribe_bypasses_micro_segments(self, mock_torchaudio, manifest):
        """Verifies that chunks under 0.4 seconds are skipped to prevent CTranslate2 crashes."""
        sample_rate = 16000

        # Simulate 2 seconds of blank audio
        dummy_waveform = torch.zeros(1, sample_rate * 2)
        mock_torchaudio.load.return_value = (dummy_waveform, sample_rate)

        mock_whisper_instance = MagicMock()
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_whisper_instance)

        vad_map = VadMap(
            source_id=uuid.uuid4(),
            segments=[
                # 0.1 second segment (1600 samples) -> Under 6400 limit, should be skipped [cite: 1267]
                VadSegment(start_sec=0.0, end_sec=0.1, confidence=0.9),

                # 0.5 second segment (8000 samples) -> Over 6400 limit, should be processed [cite: 1268]
                VadSegment(start_sec=1.0, end_sec=1.5, confidence=0.9)
            ]
        )

        adapter = FasterWhisperAdapter()

        # Mock the engine output (we don't care about the return value for this test)
        mock_whisper_instance.transcribe.return_value = ([], MagicMock())

        adapter.transcribe(
            model=loaded_model,
            audio_path="/mock/path.wav",
            vad_map=vad_map,
            language="en"
        )

        # The whisper model should have only been called ONCE for the 0.5s segment
        assert mock_whisper_instance.transcribe.call_count == 1

    @patch("app.features.transcription.data.faster_whisper_adapter.torchaudio")
    def test_transcribe_extracts_metadata(self, mock_torchaudio, manifest):
        """Verifies that the adapter pulls hallucination metadata from the Whisper engine output."""
        sample_rate = 16000
        dummy_waveform = torch.zeros(1, sample_rate * 2)
        mock_torchaudio.load.return_value = (dummy_waveform, sample_rate)

        mock_whisper_instance = MagicMock()
        loaded_model = LoadedModel(manifest=manifest, model_instance=mock_whisper_instance)

        vad_map = VadMap(
            source_id=uuid.uuid4(),
            segments=[VadSegment(start_sec=0.0, end_sec=1.5, confidence=0.9)]
        )

        # Mock a raw Whisper Segment object with metadata
        mock_raw_segment = MagicMock()
        mock_raw_segment.start = 0.0
        mock_raw_segment.end = 1.0
        mock_raw_segment.text = "Hello world"
        mock_raw_segment.words = []
        mock_raw_segment.no_speech_prob = 0.95
        mock_raw_segment.avg_logprob = -2.0
        mock_raw_segment.compression_ratio = 3.1

        mock_whisper_instance.transcribe.return_value = ([mock_raw_segment], MagicMock())

        adapter = FasterWhisperAdapter()
        segments_out = adapter.transcribe(
            model=loaded_model,
            audio_path="/mock/path.wav",
            vad_map=vad_map,
            language="en"
        )

        # Assert metadata was mapped to the Domain object successfully
        assert len(segments_out) == 1
        assert segments_out[0].no_speech_prob == 0.95
        assert segments_out[0].avg_logprob == -2.0
        assert segments_out[0].compression_ratio == 3.1