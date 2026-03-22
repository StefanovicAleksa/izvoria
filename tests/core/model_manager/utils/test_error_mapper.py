# Canvas: test_model_error_mapper_v1
# Path: tests/core/model_manager/utils/test_error_mapper.py

import pytest
from app.core.model_manager.utils.error_mapper import map_model_manager_error
from app.core.model_manager import exceptions as exc


class TestModelManagerErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, expected_output_type",
        [
            (MemoryError("Cannot allocate memory"), exc.OutOfVramError),
            (RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB"), exc.OutOfVramError),
            (RuntimeError("CUDA error: device-side assert triggered"), exc.CudaCrashError),
            (RuntimeError("CUDA error: invalid device ordinal"), exc.CudaCrashError),
            (FileNotFoundError("No such file or directory: 'models/whisper'"), exc.ModelNotFoundError),
            (EOFError("Ran out of input"), exc.ModelLoadError),
            (TypeError("invalid load key, '\\x00'"), exc.ModelLoadError),
            (ValueError("Some unknown random framework error"), exc.ModelManagerError),
        ]
    )
    def test_map_model_manager_error_translates_correctly(self, input_exception, expected_output_type):
        """Verifies that the error mapper correctly buckets strings and system types."""
        mapped_error = map_model_manager_error(input_exception)

        assert isinstance(mapped_error, expected_output_type)
        assert mapped_error.original_exception == input_exception

    def test_map_model_manager_error_does_not_double_wrap(self):
        """Ensures that Domain exceptions aren't recursively wrapped if thrown internally."""
        domain_exc = exc.OutOfVramError("Already a domain error")
        mapped_error = map_model_manager_error(domain_exc)

        assert mapped_error is domain_exc