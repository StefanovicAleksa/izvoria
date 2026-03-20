# Canvas: test_sentence_error_mapper_v1
# Path: tests/features/sentence_reconstruction/utils/test_error_mapper.py

import pytest
import torch

from app.core.database import exceptions as db_exc
from app.features.sentence_reconstruction.utils.error_mapper import map_sentence_error
from app.features.sentence_reconstruction import exceptions as exc


@pytest.mark.parametrize(
    "input_exception, expected_output_type",
    [
        (db_exc.DBForeignKeyError("Missing source"), exc.SentencePersistenceError),
        (db_exc.DataError("Bad JSONB"), exc.SentenceSerializationError),
        (db_exc.DatabaseError("Generic DB Drop"), exc.SentenceDatabaseError),
        (OSError("Model identifier not found"), exc.PunctuationModelLoadError),
        (torch.cuda.OutOfMemoryError("CUDA out of memory"), exc.PunctuationOOMError),
        (RuntimeError("Tensor dimension mismatch in pipeline"), exc.PunctuationExecutionError),
        (Exception("Random cosmic ray"), exc.SentenceReconstructionError),
    ]
)
def test_map_sentence_error_translates_correctly(input_exception, expected_output_type):
    mapped_error = map_sentence_error(input_exception)
    assert isinstance(mapped_error, expected_output_type)