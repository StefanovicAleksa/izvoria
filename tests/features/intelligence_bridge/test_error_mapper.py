# Canvas: test_error_mapper_v1
# Path: tests/features/intelligence_bridge/test_error_mapper.py

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.features.intelligence_bridge.utils.error_mapper import map_bridge_error
from app.features.intelligence_bridge import exceptions as exc


@pytest.mark.parametrize(
    "input_exception, context, expected_output_type",
    [
        (SQLAlchemyError("Connection dropped"), "DB Save", exc.BridgeDatabaseError),
        (ValueError("Failed to merge timelines"), "Merge Process", exc.TimelineMergeError),
        (Exception("Unknown sub-process fault"), "Unknown", exc.IntelligenceBridgeError),
    ]
)
def test_map_bridge_error_translates_correctly(input_exception, context, expected_output_type):
    # Act
    mapped_error = map_bridge_error(input_exception, context=context)

    # Assert
    assert isinstance(mapped_error, expected_output_type)
    assert mapped_error.original_exception == input_exception
    assert context in str(mapped_error)