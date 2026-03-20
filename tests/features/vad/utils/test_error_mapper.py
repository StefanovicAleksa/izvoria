# File: tests/features/vad/test_error_mapper.py
import pytest
import soundfile as sf
from sqlalchemy.exc import SQLAlchemyError

from app.features.vad.utils.error_mapper import map_vad_error
from app.features.vad import exceptions as exc


@pytest.mark.parametrize(
    "input_exception, expected_output_type",
    [
        (FileNotFoundError("No such file or directory"), exc.VadModelFilePathError),
        (SQLAlchemyError("Connection failed"), exc.VadDatabaseError),
        (RuntimeError("Expected all tensors to be on the same device"), exc.VadDeviceError),
        (Exception("Audio sample rate must be 16000"), exc.AudioSampleRateError),
        # FIXED: Using ValueError with "format" avoids the C-library instantiation crash
        (ValueError("Audio format not recognized"), exc.AudioFormatError),
        (AttributeError("omegaconf missing key"), exc.VadModelConfigError),
        (RuntimeError("nemo model forward pass failed"), exc.VadInferenceError),
        (ValueError("Random unexpected error"), exc.VadError),
    ]
)
def test_map_vad_error_translates_correctly(input_exception, expected_output_type):
    # Act
    mapped_error = map_vad_error(input_exception)

    # Assert
    assert isinstance(mapped_error, expected_output_type)
    assert mapped_error.original_exception == input_exception