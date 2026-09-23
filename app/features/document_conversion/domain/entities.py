# Path: app/features/document_conversion/domain/entities.py

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.features.document_conversion.domain.enums import DocumentKind, ExtractionMethod


class OcrPage(BaseModel):
    """What OCR read on one page image."""
    # Recognised text in reading order; a blank line separates paragraphs
    text: str = ""
    mean_confidence: Optional[float] = None
    # RapidOCR silently drops every line whose recognition comes back empty. The detector's own count is kept
    # beside the count of lines actually read, so a page that lost text is known to have lost it.
    lines_detected: int = 0
    lines_read: int = 0

    model_config = ConfigDict(frozen=True)


class PageContent(BaseModel):
    """One page of a PDF (page_no from 1), or a whole DOCX (page_no None)."""
    page_no: Optional[int] = None
    # Markdown text of the page, verbatim apart from whitespace
    text: str = ""
    method: ExtractionMethod = ExtractionMethod.TEXT_LAYER
    ocr_confidence: Optional[float] = None
    ocr_lines_detected: int = 0
    ocr_lines_read: int = 0
    # Why a page has no text, when it has none
    note: Optional[str] = None

    model_config = ConfigDict(frozen=True)

    @property
    def ocr_incomplete(self) -> bool:
        """OCR detected lines on this page that it then failed to read: the page text is missing words."""
        return self.method is ExtractionMethod.OCR and self.ocr_lines_read < self.ocr_lines_detected


class ConvertedDocument(BaseModel):
    """A document turned into page-tagged Markdown: the input of document structuring."""
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    source_id: uuid.UUID
    kind: DocumentKind
    pages: List[PageContent] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    model_config = ConfigDict(frozen=True)

    @property
    def pages_ocr(self) -> int:
        return sum(1 for page in self.pages if page.method is ExtractionMethod.OCR)

    @property
    def pages_ocr_incomplete(self) -> int:
        return sum(1 for page in self.pages if page.ocr_incomplete)

    @property
    def pages_unreadable(self) -> int:
        return sum(1 for page in self.pages if page.method is ExtractionMethod.NONE)
