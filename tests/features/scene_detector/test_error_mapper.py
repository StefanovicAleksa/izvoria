# Canvas: test_error_mapper_v1.py
# Path: tests/features/scene_detector/test_error_mapper.py

import pytest
import subprocess
from sqlalchemy.exc import SQLAlchemyError

from app.features.scene_detector.utils.error_mapper import map_scene_detector_error
from app.features.scene_detector import exceptions as exc


@pytest.mark.parametrize(
    "input_exception, context, expected_output_type",
    [
        (FileNotFoundError("ffmpeg not found"), "", exc.FFmpegNotFoundError),
        (FileNotFoundError("video.mp4 not found"), "video.mp4", exc.FFmpegInvalidVideoError),
        (SQLAlchemyError("Connection failed"), "", exc.SceneDetectorDatabaseError),
        (subprocess.TimeoutExpired(cmd="ffmpeg", timeout=60), "video.mp4", exc.FFmpegTimeoutError),
        (subprocess.CalledProcessError(returncode=1, cmd="ffmpeg", stderr="Crash"), "", exc.FFmpegExecutionError),
        (ValueError("invalid time window"), "", exc.SceneDetectionWindowError),
        (OSError("Disk full"), "", exc.SceneDeduplicationError),
        (Exception("Random unexpected error"), "", exc.SceneDetectorError),
    ]
)
def test_map_scene_detector_error_translates_correctly(input_exception, context, expected_output_type):
    # Act
    mapped_error = map_scene_detector_error(input_exception, context=context)

    # Assert
    assert isinstance(mapped_error, expected_output_type)
    assert mapped_error.original_exception == input_exception