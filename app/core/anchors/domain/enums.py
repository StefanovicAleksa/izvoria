# Path: app/core/anchors/domain/enums.py

from enum import Enum


class AnchorKind(str, Enum):
    """
    What a chunk's location in its source is measured in -- what a citation points at.

    A transcript chunk is a time range, a PDF chunk a page range, a plain-text or Markdown chunk a line range, and
    a document without pages or checkable lines (DOCX, HTML, RTF, EPUB) a section path, and a still image the
    image as a whole. Chunks of every kind share
    `vector_chunks`, whose `start_sec`/`end_sec` are 0.0 for anything that is not TIME, so a reader must check the
    kind before it treats those fields as times.

    A LINE anchor keeps its first and last line in `page_start`/`page_end`: the same two numbers every layer
    already carries, with the kind saying what they count.
    """
    TIME = "time"
    PAGE = "page"
    LINE = "line"
    SECTION = "section"
    IMAGE = "image"
