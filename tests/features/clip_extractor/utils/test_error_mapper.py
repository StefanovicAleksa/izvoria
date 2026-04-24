# Canvas: test_clip_extractor_error_mapper_v1
# Path: tests/features/clip_extractor/utils/test_error_mapper.py

import subprocess
from sqlalchemy.exc import SQLAlchemyError
from app.features.clip_extractor.utils.error_mapper import map_clip_extractor_error
from app.features.clip_extractor import exceptions as exc

class TestClipExtractorErrorMapper:
    def test_map_sqlalchemy_error(self):
        """SQLAlchemyError -> ClipExtractorDatabaseError"""
        raw_err = SQLAlchemyError("Connection dropped")
        mapped = map_clip_extractor_error(raw_err)
        assert isinstance(mapped, exc.ClipExtractorDatabaseError)
        assert "database operation failed" in str(mapped).lower()

    def test_map_ffmpeg_not_found(self):
        """FileNotFoundError (with 'ffmpeg') -> FFmpegVideoNotFoundError"""
        raw_err = FileNotFoundError("ffmpeg binary is missing")
        mapped = map_clip_extractor_error(raw_err)
        assert isinstance(mapped, exc.FFmpegVideoNotFoundError)
        assert "binary not found" in str(mapped).lower()

    def test_map_timeout_expired(self):
        """subprocess.TimeoutExpired -> FFmpegVideoTimeoutError"""
        raw_err = subprocess.TimeoutExpired(cmd="ffmpeg", timeout=300)
        mapped = map_clip_extractor_error(raw_err)
        assert isinstance(mapped, exc.FFmpegVideoTimeoutError)
        assert "timed out after 300" in str(mapped).lower()

    def test_map_called_process_error(self):
        """subprocess.CalledProcessError -> FFmpegVideoExecutionError"""
        raw_err = subprocess.CalledProcessError(
            returncode=1,
            cmd="ffmpeg",
            stderr="moov atom not found"
        )
        mapped = map_clip_extractor_error(raw_err)
        assert isinstance(mapped, exc.FFmpegVideoExecutionError)
        assert "moov atom not found" in str(mapped)

    def test_map_os_error(self):
        """OSError -> ClipExtractorError"""
        raw_err = OSError("No space left on device")
        mapped = map_clip_extractor_error(raw_err)
        assert type(mapped) is exc.ClipExtractorError
        assert "system i/o error" in str(mapped).lower()