# Canvas: test_extraction_error_mapper_v1
# Path: tests/features/frame_extraction/utils/test_error_mapper.py

import pytest
import subprocess
from PIL import UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError

from app.features.frame_extraction.utils.error_mapper import map_extraction_error
from app.features.frame_extraction import exceptions as exc


@pytest.mark.parametrize(
    "input_exception, context, expected_output_type",
    [
        # FFmpeg Subprocess Errors
        (subprocess.CalledProcessError(returncode=1, cmd="ffmpeg", stderr="Crash"), "", exc.FFmpegProcessError),
        (subprocess.TimeoutExpired(cmd="ffmpeg", timeout=60), "video.mp4", exc.FFmpegProcessError),
        (FileNotFoundError("ffmpeg binary not found"), "ffmpeg", exc.FFmpegProcessError),
        (FileNotFoundError("video missing"), "target video", exc.FFmpegProcessError),

        # PIL / I-O Image Errors
        (UnidentifiedImageError("Not a JPEG"), "Checking image quality", exc.FrameQualityError),
        (OSError("Disk full"), "Reading image for blur detection", exc.FrameQualityError),
        (OSError("Permission denied"), "Hashing image frame", exc.FrameDeduplicationError),

        # Database Errors
        (SQLAlchemyError("Connection lost"), "Saving chunk frames", exc.ExtractionPersistenceError),
        (SQLAlchemyError("Syntax error"), "Generic query", exc.ExtractionDatabaseError),

        # Fallback
        (Exception("Random unexpected error"), "", exc.FrameExtractionError),
    ]
)
def test_map_extraction_error_translates_correctly(input_exception, context, expected_output_type):
    # Act
    mapped_error = map_extraction_error(input_exception, context=context)

    # Assert
    assert isinstance(mapped_error, expected_output_type)
    assert mapped_error.original_exception == input_exception