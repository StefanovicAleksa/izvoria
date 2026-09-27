# Path: app/features/image_description/utils/exif.py

import re
from typing import Optional

from PIL import Image

_EXIF_IFD = 0x8769
_DATE_TIME_ORIGINAL = 36867
_DATE_TIME = 306
_EXIF_DATE = re.compile(r"^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2})")


def date_taken(image: Image.Image) -> Optional[str]:
    """
    When the photo was taken ("2025-03-14 10:02"), from EXIF DateTimeOriginal, else DateTime; None when absent or
    malformed. Nothing else is read from EXIF: GPS in particular is sensitive and no citation needs it.
    """
    try:
        exif = image.getexif()
        value = exif.get_ifd(_EXIF_IFD).get(_DATE_TIME_ORIGINAL) or exif.get(_DATE_TIME)
    except Exception:
        return None
    match = _EXIF_DATE.match(str(value or "").strip())
    if not match or match.group(1) == "0000":
        return None
    year, month, day, hour, minute = match.groups()
    return f"{year}-{month}-{day} {hour}:{minute}"
