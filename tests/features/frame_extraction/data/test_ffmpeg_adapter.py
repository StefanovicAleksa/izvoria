# Canvas: test_ffmpeg_extraction_adapter_v2
# Path: tests/features/frame_extraction/data/test_ffmpeg_adapter.py

import pytest
import subprocess
from pathlib import Path
from unittest.mock import patch

from app.features.contextual_retrieval.domain.entities import ContextualizedChunk
from app.features.frame_extraction.data.ffmpeg_adapter import FFmpegExtractionAdapter
from app.features.frame_extraction import exceptions as exc


class TestFFmpegExtractionAdapter:
    @pytest.fixture
    def adapter(self):
        # Initialized with our new max_dimension parameter
        return FFmpegExtractionAdapter(timeout_seconds=60, max_dimension=1024)

    @pytest.fixture
    def sample_chunk(self):
        # 2 second chunk from 10.0 to 12.0
        return ContextualizedChunk(start_sec=10.0, end_sec=12.0)

    @patch("app.features.frame_extraction.data.ffmpeg_adapter.subprocess.run")
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.glob")
    def test_extract_raw_frames_happy_path(self, mock_glob, mock_exists, mock_run, adapter, sample_chunk):
        # Simulate FFmpeg creating 2 frames
        mock_glob.return_value = [Path("frame_0001.jpg"), Path("frame_0002.jpg")]

        result = adapter.extract_raw_frames("video.mp4", "/tmp/out", sample_chunk, fps=1)

        assert len(result) == 2
        # Verify timestamp mapping calculation (starts at 10.0)
        assert result[0][0] == 10.0
        assert result[1][0] == 11.0

        # Verify FFmpeg fast-seek command structure
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        assert "-ss" in cmd
        assert "-t" in cmd
        assert "10.0" in cmd  # Start time
        assert "2.0" in cmd  # Duration

        # Verify the new video filter string includes both fps and the downscale logic
        vf_index = cmd.index("-vf")
        vf_string = cmd[vf_index + 1]

        assert "fps=1" in vf_string
        assert "scale='min(1024,iw)':'min(1024,ih)'" in vf_string
        assert "force_original_aspect_ratio=decrease" in vf_string

    def test_extract_raw_frames_missing_video(self, adapter, sample_chunk):
        with pytest.raises(exc.FFmpegProcessError):
            adapter.extract_raw_frames("nonexistent.mp4", "/tmp/out", sample_chunk)

    @patch("app.features.frame_extraction.data.ffmpeg_adapter.subprocess.run",
           side_effect=subprocess.TimeoutExpired(cmd="ffmpeg", timeout=60))
    @patch("pathlib.Path.exists", return_value=True)
    def test_extract_raw_frames_timeout(self, mock_exists, mock_run, adapter, sample_chunk):
        with pytest.raises(exc.FFmpegProcessError):
            adapter.extract_raw_frames("video.mp4", "/tmp/out", sample_chunk)