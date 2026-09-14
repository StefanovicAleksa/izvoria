# Canvas: test_ffmpeg_adapter_v3
# Path: tests/features/audio_extractor/data/test_ffmpeg_adapter.py

import subprocess
import wave
import pytest
from pathlib import Path
from unittest.mock import patch

from app.features.audio_extractor.data.ffmpeg_adapter import FFmpegAdapter
from app.features.audio_extractor import exceptions as exc


def _completed(stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


class TestFFmpegAdapter:
    @pytest.fixture
    def adapter(self):
        return FFmpegAdapter(timeout_seconds=60)

    @patch("app.features.audio_extractor.data.ffmpeg_adapter.subprocess.run")
    def test_extract_audio_happy_path(self, mock_run, adapter, tmp_path):
        """Probes for audio streams first, then runs the exact 16kHz mono extraction command."""
        source_file = tmp_path / "test_video.mp4"
        source_file.write_text("fake video data")
        mock_run.side_effect = [_completed("1\n"), _completed()]

        result_path = adapter.extract_audio(source_file)

        assert isinstance(result_path, Path)
        assert result_path.suffix == ".wav"
        assert result_path.exists()

        assert mock_run.call_count == 2
        probe_args = mock_run.call_args_list[0][0][0]
        assert probe_args == [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(source_file)
        ]
        ffmpeg_args = mock_run.call_args_list[1][0][0]
        assert ffmpeg_args == [
            "ffmpeg", "-i", str(source_file), "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1", "-y", str(result_path)
        ]

        result_path.unlink()

    @patch("app.features.audio_extractor.data.ffmpeg_adapter.subprocess.run")
    def test_extract_audio_no_audio_stream_fallback(self, mock_run, adapter, tmp_path):
        """A file that ffprobe reports as having zero audio streams gets a silent 1-sec WAV."""
        source_file = tmp_path / "silent_video.mp4"
        source_file.write_text("fake silent video")
        mock_run.return_value = _completed("")

        result_path = adapter.extract_audio(source_file)

        # FFmpeg extraction must not run at all
        assert mock_run.call_count == 1
        assert isinstance(result_path, Path)
        assert result_path.exists()

        with wave.open(str(result_path), 'rb') as wav_file:
            assert wav_file.getnchannels() == 1
            assert wav_file.getframerate() == 16000
            assert wav_file.getnframes() == 16000  # 1 second of silence

        result_path.unlink()

    @patch("app.features.audio_extractor.data.ffmpeg_adapter.subprocess.run")
    def test_extract_audio_stream_error_with_audio_present_raises(self, mock_run, adapter, tmp_path):
        """FFmpeg stream errors on media that DOES have audio must raise, not become silence."""
        source_file = tmp_path / "video.mp4"
        source_file.write_text("fake video")
        mock_run.side_effect = [
            _completed("1\n"),
            subprocess.CalledProcessError(returncode=1, cmd="ffmpeg", stderr="Output file does not contain any stream")
        ]

        with pytest.raises(exc.FFmpegExecutionError, match="does not contain any stream"):
            adapter.extract_audio(source_file)

    @patch("app.features.audio_extractor.data.ffmpeg_adapter.subprocess.run")
    def test_extract_audio_genuine_crash_raises_error(self, mock_run, adapter, tmp_path):
        """A probe failure on a corrupted file bubbles up as a typed FFmpeg error."""
        source_file = tmp_path / "corrupted_video.mp4"
        source_file.write_text("fake corrupted video")

        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd="ffprobe",
            stderr="Invalid data found when processing input"
        )

        with pytest.raises(exc.FFmpegExecutionError, match="Invalid data found when processing input"):
            adapter.extract_audio(source_file)
