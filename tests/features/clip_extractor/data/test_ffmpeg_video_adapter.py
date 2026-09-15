# Canvas: test_ffmpeg_video_adapter_v2
# Path: tests/features/clip_extractor/data/test_ffmpeg_video_adapter.py

import subprocess
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch

from app.features.clip_extractor.data.ffmpeg_video_adapter import FFmpegVideoAdapter
from app.features.clip_extractor import exceptions as exc


class TestFFmpegVideoAdapter:
    @pytest.fixture
    def adapter(self):
        return FFmpegVideoAdapter(timeout_seconds=60)

    @patch("app.features.clip_extractor.data.ffmpeg_video_adapter.subprocess.run")
    def test_extract_clip_happy_path(self, mock_run, adapter, tmp_path):
        """Verifies input seeking plus a DURATION (-t), never an output -to position."""
        source_file = tmp_path / "test_video.mp4"
        source_file.write_text("fake video data")

        result_path = adapter.extract_clip(source_file, start_sec=15.0, end_sec=30.0)

        assert isinstance(result_path, Path)
        assert result_path.suffix == ".mp4"
        assert result_path.exists()

        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]

        assert mock_run.call_args.kwargs["stdin"] == subprocess.DEVNULL
        assert called_args == [
            "ffmpeg",
            "-nostdin",
            "-ss", "15.0",
            "-i", str(source_file),
            "-t", "15.000",
            "-c", "copy",
            "-y",
            str(result_path)
        ]
        assert "-to" not in called_args

        if result_path.exists():
            result_path.unlink()

    @patch("app.features.clip_extractor.data.ffmpeg_video_adapter.subprocess.run")
    def test_extract_clip_duration_is_relative_to_start(self, mock_run, adapter, tmp_path):
        """A late chunk (3600s-3700s) must produce a 100s clip, not a 3700s one."""
        source_file = tmp_path / "long_video.mp4"
        source_file.write_text("fake video data")

        result_path = adapter.extract_clip(source_file, start_sec=3600.0, end_sec=3700.0)

        called_args = mock_run.call_args[0][0]
        assert called_args[called_args.index("-t") + 1] == "100.000"

        result_path.unlink()

    @patch("app.features.clip_extractor.data.ffmpeg_video_adapter.subprocess.run")
    def test_extract_clip_negative_range_clamps_to_zero(self, mock_run, adapter, tmp_path):
        source_file = tmp_path / "video.mp4"
        source_file.write_text("fake video data")

        result_path = adapter.extract_clip(source_file, start_sec=10.0, end_sec=5.0)

        called_args = mock_run.call_args[0][0]
        assert called_args[called_args.index("-t") + 1] == "0.000"

        result_path.unlink()

    def test_extract_clip_source_not_found(self, adapter):
        with pytest.raises(exc.ClipExtractorError, match="Source file not found"):
            adapter.extract_clip("/path/to/nothing.mp4", 0.0, 10.0)

    @patch("app.features.clip_extractor.data.ffmpeg_video_adapter.subprocess.run")
    def test_extract_clip_cleans_up_on_crash(self, mock_run, adapter, tmp_path):
        """Verifies temp files are not leaked if FFmpeg crashes."""
        source_file = tmp_path / "broken_video.mp4"
        source_file.write_text("fake data")

        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")

        temp_dir = Path(tempfile.gettempdir())
        initial_mp4s = set(temp_dir.glob("onyx_clip_*.mp4"))

        with pytest.raises(exc.FFmpegVideoExecutionError):
            adapter.extract_clip(source_file, 0.0, 5.0)

        final_mp4s = set(temp_dir.glob("onyx_clip_*.mp4"))
        leaked_files = final_mp4s - initial_mp4s
        assert len(leaked_files) == 0
