# Canvas: test_error_mapper_v3
# Path: tests/features/sentence_reconstruction/test_error_mapper.py

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
        # FIXED: Test the DB exception coming from the core Manager, not raw SQLAlchemy
        (db_exc.DatabaseError("Generic DB Drop"), exc.SentenceDatabaseError),
        (OSError("Model identifier sat-3l-sm not found"), exc.SaTModelLoadError),
        (torch.cuda.OutOfMemoryError("CUDA out of memory"), exc.SaTOOMError),
        (ZeroDivisionError("division by zero"), exc.ZeroDurationSegmentError),
        (RuntimeError("Tensor dimension mismatch"), exc.SaTExecutionError),
        (Exception("Random cosmic ray"), exc.SentenceReconstructionError),
    ]
)
def test_map_sentence_error_translates_correctly(input_exception, expected_output_type):
    # Act
    mapped_error = map_sentence_error(input_exception)

    # Assert
    assert isinstance(mapped_error, expected_output_type)