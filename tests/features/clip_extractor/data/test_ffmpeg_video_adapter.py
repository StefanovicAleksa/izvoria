# Canvas: test_ffmpeg_video_adapter_v1
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
        """Verifies correct FFmpeg CLI arguments for fast keyframe seeking."""
        source_file = tmp_path / "test_video.mp4"
        source_file.write_text("fake video data")

        result_path = adapter.extract_clip(source_file, start_sec=15.0, end_sec=30.0)

        assert isinstance(result_path, Path)
        assert result_path.suffix == ".mp4"
        assert result_path.exists()

        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]

        # Verify command structure explicitly
        assert called_args[0] == "ffmpeg"
        assert called_args[1] == "-ss"
        assert called_args[2] == "15.0"
        assert called_args[3] == "-i"
        assert called_args[4] == str(source_file)
        assert called_args[5] == "-to"
        assert called_args[6] == "30.0"
        assert called_args[7] == "-c"
        assert called_args[8] == "copy"

        if result_path.exists():
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