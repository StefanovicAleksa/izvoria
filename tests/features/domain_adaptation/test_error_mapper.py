# Canvas: test_domain_error_mapper_v1
# Path: tests/features/domain_adaptation/test_error_mapper.py

import pytest
import subprocess
import json
from sqlalchemy.exc import SQLAlchemyError

from app.features.domain_adaptation.utils.error_mapper import map_domain_adaptation_error
from app.features.domain_adaptation import exceptions as exc

class TestDomainAdaptationErrorMapper:
    @pytest.mark.parametrize(
        "input_exception, expected_output_type",
        [
            (SQLAlchemyError("DB issue"), exc.DomainDatabaseError),
            (subprocess.TimeoutExpired(cmd="ffmpeg", timeout=30), exc.AudioSamplingError),
            (subprocess.CalledProcessError(returncode=1, cmd="ffmpeg"), exc.AudioSamplingError),
            (json.JSONDecodeError("Expecting value", "", 0), exc.GlossaryManifestError),
            (FileNotFoundError("finance.json missing"), exc.GlossaryManifestError),
            (FileNotFoundError("model bin missing"), exc.CpuModelLoadError),
            (RuntimeError("ctranslate2 memory allocation failed"), exc.ScoutInferenceError), # if "whisper" in context
            (Exception("Random cosmic ray"), exc.DomainAdaptationError),
        ]
    )
    def test_map_error_translates_correctly(self, input_exception, expected_output_type):
        context = "whisper model execution" if expected_output_type == exc.ScoutInferenceError else "test context"
        mapped_error = map_domain_adaptation_error(input_exception, context=context)

        assert isinstance(mapped_error, expected_output_type)
        assert mapped_error.original_exception == input_exception