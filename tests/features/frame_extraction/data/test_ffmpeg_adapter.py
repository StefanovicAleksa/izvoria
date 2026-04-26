# Canvas: test_frame_ffmpeg_adapter_v2
# Path: tests/features/frame_extraction/data/test_ffmpeg_adapter.py

import pytest
import subprocess
from unittest.mock import patch

from app.features.frame_extraction.data.ffmpeg_adapter import FFmpegExtractionAdapter


class TestFFmpegExtractionAdapter:
    @pytest.fixture
    def adapter(self):
        return FFmpegExtractionAdapter()

    @patch("app.features.frame_extraction.data.ffmpeg_adapter.subprocess.run")
    def test_extract_frame_success_with_corruption_flags(self, mock_run, adapter):
        """Verifies that the adapter executes FFmpeg with the correct resilience flags."""
        # Setup mock to simulate a successful extraction
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)

        result = adapter.extract_frame("corrupted_video.mp4", 10.5, "output.jpg")

        # Assertions
        assert result is True
        mock_run.assert_called_once()

        cmd_args = mock_run.call_args[0][0]

        # Verify the new corruption resilience flags are enforced
        assert "-err_detect" in cmd_args
        assert "ignore_err" in cmd_args
        assert "-fflags" in cmd_args
        assert "+discardcorrupt" in cmd_args

        # Verify standard parameters
        assert "-ss" in cmd_args
        assert "10.5" in cmd_args
        assert "corrupted_video.mp4" in cmd_args
        assert "output.jpg" in cmd_args
        assert "-vframes" in cmd_args
        assert "1" in cmd_args

    @patch("app.features.frame_extraction.data.ffmpeg_adapter.subprocess.run")
    def test_extract_frame_handles_called_process_error(self, mock_run, adapter):
        """Verifies that unrecoverable FFmpeg errors return False safely."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd="ffmpeg",
            stderr="Fatal unrecoverable error"
        )

        result = adapter.extract_frame("video.mp4", 15.0, "output.jpg")

        assert result is False

    @patch("app.features.frame_extraction.data.ffmpeg_adapter.subprocess.run")
    def test_extract_frame_handles_timeout(self, mock_run, adapter):
        """Verifies that hanging FFmpeg processes return False safely."""
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30)

        result = adapter.extract_frame("video.mp4", 20.0, "output.jpg")

        assert result is False