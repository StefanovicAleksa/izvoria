# filename: tests/features/audio_extractor/test_error_mapper.py

import subprocess
from app.features.audio_extractor.utils.error_mapper import map_ffmpeg_error
from app.features.audio_extractor import exceptions as exc

class TestFFmpegErrorMapper:
    def test_map_file_not_found(self):
        """FileNotFoundError -> FFmpegNotFoundError"""
        raw_err = FileNotFoundError("ffprobe not found")
        mapped = map_ffmpeg_error(raw_err)
        assert isinstance(mapped, exc.FFmpegNotFoundError)
        assert "binary not found" in str(mapped).lower()

    def test_map_timeout_expired(self):
        """subprocess.TimeoutExpired -> FFmpegTimeoutError"""
        raw_err = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=3600)
        mapped = map_ffmpeg_error(raw_err)
        assert isinstance(mapped, exc.FFmpegTimeoutError)
        assert "timed out after 3600" in str(mapped).lower()

    def test_map_called_process_error_with_stderr(self):
        """subprocess.CalledProcessError -> FFmpegExecutionError (with logs)"""
        raw_err = subprocess.CalledProcessError(
            returncode=1,
            cmd="ffmpeg",
            stderr="Invalid data found when processing input"
        )
        mapped = map_ffmpeg_error(raw_err)
        assert isinstance(mapped, exc.FFmpegExecutionError)
        assert "Invalid data found" in str(mapped)

    def test_map_os_error(self):
        """OSError -> AudioExtractorError"""
        raw_err = OSError("No space left on device")
        mapped = map_ffmpeg_error(raw_err)
        # Should fall back to the base error, but not the specific FFmpeg ones
        assert type(mapped) is exc.AudioExtractorError
        assert "System I/O error" in str(mapped)

    def test_map_unknown_exception(self):
        """Exception -> AudioExtractorError"""
        raw_err = Exception("Random cosmic ray bit flip")
        mapped = map_ffmpeg_error(raw_err)
        assert type(mapped) is exc.AudioExtractorError