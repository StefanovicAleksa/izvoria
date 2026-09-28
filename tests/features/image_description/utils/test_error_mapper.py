# Path: tests/features/image_description/utils/test_error_mapper.py

import pytest
from PIL import Image, UnidentifiedImageError
from sqlalchemy.exc import SQLAlchemyError

from app.core.database import exceptions as core_db_exc
from app.core.model_manager.exceptions.vram import OutOfVramError
from app.features.image_description import exceptions as exc
from app.features.image_description.utils.error_mapper import map_image_description_error
from app.features.visual_analysis.exceptions import ImageDecodeError


class TestImageDescriptionErrorMapper:
    @pytest.mark.parametrize("error, expected_type", [
        (SQLAlchemyError("down"), exc.ImageDescriptionPersistenceError),
        (core_db_exc.DatabaseError("down"), exc.ImageDescriptionPersistenceError),
        (FileNotFoundError("gone.jpg"), exc.SourceImageMissingError),
        (UnidentifiedImageError("not an image"), exc.ImageReadError),
        (Image.DecompressionBombError("too big"), exc.ImageReadError),
        (ImageDecodeError("truncated"), exc.ImageReadError),
        (RuntimeError("anything else"), exc.ImageDescriptionError),
    ])
    def test_translates_to_feature_exceptions(self, error, expected_type):
        mapped = map_image_description_error(error, context="test")

        assert type(mapped) is expected_type
        assert mapped.original_exception is error

    def test_does_not_double_wrap(self):
        original = exc.EmptyImageError("nothing there")

        assert map_image_description_error(original) is original

    def test_model_manager_errors_pass_through(self):
        with pytest.raises(OutOfVramError):
            map_image_description_error(OutOfVramError("no room"))
