# Canvas: test_ffmpeg_adapter_v2
# Path: tests/features/scene_detector/test_ffmpeg_adapter.py

import pytest
from unittest.mock import patch, MagicMock

from app.features.scene_detector.data.ffmpeg_scene_adapter import FFmpegSceneAdapter
from app.features.scene_detector.domain.entities import TimeWindow
from app.features.scene_detector import exceptions as exc


class TestFFmpegSceneAdapter:
    @pytest.fixture
    def adapter(self):
        return FFmpegSceneAdapter(timeout_seconds=60)

    @patch("app.features.scene_detector.data.ffmpeg_scene_adapter.subprocess.run")
    @patch("pathlib.Path.exists", return_value=True)
    def test_extract_scenes_global_pass(self, mock_exists, mock_run, adapter):
        """Verifies parsing of the showinfo pts_time output."""
        # Mocking FFmpeg's stderr output
        mock_result = MagicMock()
        mock_result.stderr = "[Parsed_showinfo_1 @ 0x123] n:   0 pts_time:12.5 \n[Parsed_showinfo_1 @ 0x123] n:   1 pts_time:45.0"
        mock_run.return_value = mock_result

        result = adapter.extract_scenes("video.mp4", "/tmp", threshold=0.4)

        assert len(result) == 2
        assert result[0][0] == 12.5
        assert result[1][0] == 45.0
        assert mock_exists.called  # Use the parameter to clear the warning

    @patch("app.features.scene_detector.data.ffmpeg_scene_adapter.subprocess.run")
    @patch("pathlib.Path.exists", return_value=True)
    def test_extract_scenes_with_time_windows(self, mock_exists, mock_run, adapter):
        """Verifies fast-seeking (-ss) applies global timestamp correction."""
        mock_result = MagicMock()
        # FFmpeg will reset the clock to 0 for the seeked chunk
        mock_result.stderr = "[Parsed_showinfo_1] n: 0 pts_time:2.0"
        mock_run.return_value = mock_result

        # We start looking at 10.0s. So a local pts_time of 2.0s means global 12.0s.
        windows = [TimeWindow(start_sec=10.0, end_sec=20.0)]

        result = adapter.extract_scenes("video.mp4", "/tmp", threshold=0.4, windows=windows)

        assert len(result) == 1
        assert result[0][0] == 12.0  # 10.0 + 2.0
        assert mock_exists.called  # Use the parameter to clear the warning

    def test_extract_scenes_missing_video(self, adapter):
        with pytest.raises(exc.FFmpegInvalidVideoError):
            adapter.extract_scenes("nonexistent.mp4", "/tmp")