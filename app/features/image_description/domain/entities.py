# Path: app/features/image_description/domain/entities.py

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ImageDescription(BaseModel):
    """
    What is known about one standalone image: the vision model's description, OCR's reading of its text, and which
    of the text the vision model claimed to read OCR confirmed. An image has no words of its own, so this -- rendered
    by `render_excerpt` -- is what its citation quotes, and every part of it says who wrote it.
    """
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID
    file_name: str
    # The vision model's description ("A photographed whiteboard with a timeline"); empty when it gave none.
    # Quoted text and numbers OCR did not read are already replaced by a marker.
    shows: str = ""
    marks: Optional[str] = None
    # OCR's reading, in reading order: the verbatim text of the image. Empty without OCR or when it read nothing.
    ocr_text: str = ""
    ocr_lines_detected: int = 0
    ocr_lines_read: int = 0
    # OCR ran on this image; False on an appliance without OCR (Serbian), whose text is then unverified
    ocr_checked: bool = False
    # Text items the vision model read that OCR confirmed
    verified_items: List[str] = Field(default_factory=list)
    # Text items the vision model read with no OCR to check them against: shown, and labelled unverified
    unverified_items: List[str] = Field(default_factory=list)
    # Text items the vision model read that OCR did not: never indexed
    dropped_items: List[str] = Field(default_factory=list)
    # EXIF date taken ("2025-03-14 10:02"), when the file has one
    taken_at: Optional[str] = None
    width: int = 0
    height: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=True)

    @property
    def ocr_incomplete(self) -> bool:
        """OCR detected lines it then failed to read: the text reading is missing words."""
        return self.ocr_checked and self.ocr_lines_read < self.ocr_lines_detected
