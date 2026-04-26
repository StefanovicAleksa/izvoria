# Canvas: test_ffmpeg_adapter_v2
# Path: tests/features/audio_extractor/data/test_ffmpeg_adapter.py

import subprocess
import wave
import pytest
from pathlib import Path
from unittest.mock import patch

from app.features.audio_extractor.data.ffmpeg_adapter import FFmpegAdapter
from app.features.audio_extractor import exceptions as exc


class TestFFmpegAdapter:
    @pytest.fixture
    def adapter(self):
        return FFmpegAdapter(timeout_seconds=60)

    @patch("app.features.audio_extractor.data.ffmpeg_adapter.subprocess.run")
    def test_extract_audio_happy_path(self, mock_run, adapter, tmp_path):
        """Verifies correct FFmpeg CLI arguments and temp file generation."""
        source_file = tmp_path / "test_video.mp4"
        source_file.write_text("fake video data")

        result_path = adapter.extract_audio(source_file)

        assert isinstance(result_path, Path)
        assert result_path.suffix == ".wav"
        assert result_path.exists() 

        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]

        assert called_args[0] == "ffmpeg"
        assert "-i" in called_args

    @patch("app.features.audio_extractor.data.ffmpeg_adapter.subprocess.run")
    def test_extract_audio_no_audio_stream_fallback(self, mock_run, adapter, tmp_path):
        """Verifies that a video with no audio stream generates a silent 1-sec WAV instead of crashing."""
        source_file = tmp_path / "silent_video.mp4"
        source_file.write_text("fake silent video")

        # Simulate FFmpeg crashing because there's no audio track
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd="ffmpeg",
            stderr="Output file does not contain any stream"
        )

        result_path = adapter.extract_audio(source_file)

        # 1. Assert the adapter swallowed the error and returned a path
        assert isinstance(result_path, Path)
        assert result_path.exists()

        # 2. Assert the generated file is actually a valid WAV file
        with wave.open(str(result_path), 'rb') as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getframerate() == 16000
            assert wav_file.getnframes() == 16000  # 1 second of silence

    @patch("app.features.audio_extractor.data.ffmpeg_adapter.subprocess.run")
    def test_extract_audio_genuine_crash_raises_error(self, mock_run, adapter, tmp_path):
        """Verifies that genuine FFmpeg errors (not related to missing streams) still bubble up correctly."""
        source_file = tmp_path / "corrupted_video.mp4"
        source_file.write_text("fake corrupted video")

        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=255,
            cmd="ffmpeg",
            stderr="Invalid data found when processing input"
        )

        with pytest.raises(exc.FFmpegExecutionError, match="Invalid data found when processing input"):
            adapter.extract_audio(source_file)