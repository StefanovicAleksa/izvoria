import subprocess
import tempfile  # <-- FIX 1: Missing import added
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
        # Setup fake source file
        source_file = tmp_path / "test_video.mp4"
        source_file.write_text("fake video data")

        # Execute
        result_path = adapter.extract_audio(source_file)

        # Assertions
        assert isinstance(result_path, Path)
        assert result_path.suffix == ".wav"
        assert result_path.exists()  # The mkstemp should physically create an empty file

        # Verify subprocess was called with correct arguments
        mock_run.assert_called_once()
        called_args = mock_run.call_args[0][0]  # The command list

        assert called_args[0] == "ffmpeg"
        assert "-i" in called_args
        assert str(source_file) in called_args
        assert "-vn" in called_args
        assert "-acodec" in called_args
        assert "pcm_s16le" in called_args
        assert str(result_path) in called_args

        # Cleanup the successfully generated file for good measure
        if result_path.exists():
            result_path.unlink()

    def test_extract_audio_source_not_found(self, adapter):
        """Verifies missing source immediately raises error without calling subprocess."""
        with pytest.raises(exc.FFmpegNotFoundError):
            adapter.extract_audio("/path/to/nothing.mp4")

    @patch("app.features.audio_extractor.data.ffmpeg_adapter.subprocess.run")
    def test_extract_audio_cleans_up_on_crash(self, mock_run, adapter, tmp_path):
        """Verifies the temporary file is deleted if FFmpeg crashes."""
        source_file = tmp_path / "broken_video.mp4"
        source_file.write_text("fake data")

        # Simulate FFmpeg crashing
        mock_run.side_effect = subprocess.CalledProcessError(1, "ffmpeg")

        # FIX 2: Take a snapshot of the temp directory BEFORE the crash
        temp_dir = Path(tempfile.gettempdir())
        initial_wavs = set(temp_dir.glob("onyx_audio_*.wav"))

        with pytest.raises(exc.FFmpegExecutionError):
            adapter.extract_audio(source_file)

        # Take a snapshot AFTER the crash
        final_wavs = set(temp_dir.glob("onyx_audio_*.wav"))

        # Verify no NEW files were left behind by calculating the difference
        leaked_files = final_wavs - initial_wavs
        assert len(leaked_files) == 0  # The variable is now actively asserted!