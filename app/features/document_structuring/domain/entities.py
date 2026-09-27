# Path: app/features/document_structuring/domain/entities.py

import uuid
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

from app.core.anchors.domain.enums import AnchorKind
from app.features.document_structuring.domain.enums import BlockKind


class Block(BaseModel):
    """One structural unit of a page: a heading, a paragraph, or a Markdown table."""
    kind: BlockKind
    text: str
    page_no: Optional[int] = None
    # First and last line of the block in its page's text (from 1); for a source without pages, the file's lines
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    # Heading depth (1 = document title, 2 = clause/article, 3+ = sub-clause); 0 for body blocks
    level: int = 0

    model_config = ConfigDict(frozen=True)


class StructuredChunk(BaseModel):
    """A retrieval unit cut along the document's own sections. `text` is verbatim document text."""
    ordinal: int
    text: str
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    # The chunk's lines in its source file; cited only for sources without pages (plain text, Markdown)
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    # Headings enclosing the chunk: "Article 7 Termination › 7.2 Notice"
    section_path: Optional[str] = None

    model_config = ConfigDict(frozen=True)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


class StructuredDocument(BaseModel):
    source_id: uuid.UUID
    # PAGE for a PDF, LINE for plain text and Markdown, SECTION for other documents without pages (DOCX, EPUB...)
    anchor_kind: AnchorKind
    chunks: List[StructuredChunk] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True)
