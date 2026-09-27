# Path: tests/features/image_description/utils/test_exif.py

from unittest.mock import MagicMock

import pytest

from app.features.image_description.utils.exif import date_taken


def _image(original=None, fallback=None):
    image = MagicMock()
    exif = MagicMock()
    exif.get_ifd.return_value = {36867: original} if original else {}
    exif.get.side_effect = lambda tag: fallback if tag == 306 else None
    image.getexif.return_value = exif
    return image


class TestDateTaken:
    def test_prefers_date_time_original(self):
        image = _image(original="2025:03:14 10:02:59", fallback="2025:04:01 08:00:00")

        assert date_taken(image) == "2025-03-14 10:02"
        image.getexif.return_value.get_ifd.assert_called_once_with(0x8769)

    def test_falls_back_to_date_time(self):
        assert date_taken(_image(fallback="2024:12:31 23:59:00")) == "2024-12-31 23:59"

    @pytest.mark.parametrize("value", [None, "", "0000:00:00 00:00:00", "yesterday"])
    def test_missing_or_malformed_is_none(self, value):
        assert date_taken(_image(original=value)) is None

    def test_unreadable_exif_is_none(self):
        image = MagicMock()
        image.getexif.side_effect = OSError("corrupt EXIF")

        assert date_taken(image) is None
