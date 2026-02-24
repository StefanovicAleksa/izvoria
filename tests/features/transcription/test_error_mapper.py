# filename: test_error_mapper_v1.py
# path: tests/features/transcription/test_error_mapper.py
import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.features.transcription.utils.error_mapper import map_transcription_error
from app.features.transcription import exceptions as exc


class TestTranscriptionErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, expected_output_type",
        [
            (FileNotFoundError("No model path found"), exc.TranscriptionModelFilePathError),
            (SQLAlchemyError("DB connection dropped"), exc.TranscriptionDatabaseError),
            (ValueError("Timestamp out of bounds"), exc.TranscriptionAudioError),
            (RuntimeError("ctranslate2 compute type int8 not supported"), exc.CTranslateEngineError),
            (RuntimeError("whisper inference failed"), exc.TranscriptionInferenceError),
            (Exception("Random cosmic ray"), exc.TranscriptionError),
        ]
    )
    def test_map_transcription_error_translates_correctly(self, input_exception, expected_output_type):
        """Verifies that the error mapper correctly buckets strings and types."""
        mapped_error = map_transcription_error(input_exception)

        assert isinstance(mapped_error, expected_output_type)
        assert mapped_error.original_exception == input_exception