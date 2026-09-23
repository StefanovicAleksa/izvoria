# Path: app/core/anchors/domain/enums.py

from enum import Enum


class AnchorKind(str, Enum):
    """
    What a chunk's location in its source is measured in -- what a citation points at.

    A transcript chunk is a time range, a PDF chunk a page range, and a document without pages (DOCX) a section
    path. Chunks of every kind share `vector_chunks`, whose `start_sec`/`end_sec` are 0.0 for anything that is not
    TIME, so a reader must check the kind before it treats those fields as times.
    """
    TIME = "time"
    PAGE = "page"
    SECTION = "section"
