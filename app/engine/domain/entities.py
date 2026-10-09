# Canvas: engine_entities_v1
# Path: app/engine/domain/entities.py

from typing import Dict
from pydantic import BaseModel, Field, ConfigDict


class IngestionReport(BaseModel):
    """The structured result of pointing the Onyx Engine at a directory."""
    target_path: str
    total_files_scanned: int = 0
    successful_ingestions: int = 0
    failed_ingestions: int = 0

    # Maps file paths to error messages (e.g., {"/vault/video.mp4": "CUDA OOM"})
    errors: Dict[str, str] = Field(default_factory=dict)

    # Telemetry
    total_duration_seconds: float = 0.0
    vectors_created: int = 0
    clips_extracted: int = 0

    # Document ingestion: pages converted, pages read by OCR, OCR pages that lost lines (indexed with text
    # missing), and scanned pages with no text at all (not indexed). The last two are listed in `errors` too.
    pages_converted: int = 0
    pages_ocr: int = 0
    pages_ocr_incomplete: int = 0
    pages_unreadable: int = 0

    # Image ingestion: images described, images whose text no OCR could check (an appliance without OCR), and
    # text items the vision model read that OCR did not find -- left out of the index, and listed in `errors`
    images_described: int = 0
    images_text_unverified: int = 0
    # Low-confidence OCR lines (handwriting) the vision model did not confirm: not indexed
    image_ocr_lines_withheld: int = 0
    image_text_items_dropped: int = 0

    # Freed by dropping indexed recordings' WAVs and swapping their originals for compact copies
    storage_bytes_saved: int = 0

    model_config = ConfigDict(frozen=True)


class StorageCompactionReport(BaseModel):
    """What `OnyxEngine.compact_storage` freed (or, as a dry run, would free) across the indexed recordings."""
    dry_run: bool = False
    sources_checked: int = 0
    clip_bytes: int = 0
    audio_bytes: int = 0
    originals_replaced: int = 0
    original_bytes_saved: int = 0
    # Dry run only: originals that would be re-encoded, and their current size (the saving is known only after)
    originals_to_compact: int = 0
    original_bytes_to_compact: int = 0
    errors: Dict[str, str] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)

    @property
    def bytes_saved(self) -> int:
        return self.clip_bytes + self.audio_bytes + self.original_bytes_saved