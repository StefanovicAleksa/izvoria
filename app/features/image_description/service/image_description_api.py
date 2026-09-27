# Path: app/features/image_description/service/image_description_api.py

import logging
import uuid
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

from app.core.config import settings
from app.core.language.service.language_api import get_language_pack
from app.core.model_manager.domain.entities import LoadedModel
from app.features.document_conversion.domain.entities import OcrPage
from app.features.document_conversion.service.conversion_api import DocumentConversionAPI
from app.features.image_description.domain.entities import ImageDescription
from app.features.image_description.domain.interfaces import IImageDescriptionRepository
from app.features.image_description.exceptions import EmptyImageError, SourceImageMissingError
from app.features.image_description.utils.error_mapper import map_image_description_error
from app.features.image_description.utils.excerpt_renderer import render_context, render_excerpt
from app.features.image_description.utils.exif import date_taken
from app.features.image_description.utils.text_verification import (
    OcrReading, select_ocr_lines, strip_unconfirmed, verify_items,
)
from app.features.storage.service.storage_api import StorageAPI
from app.features.visual_analysis.domain.entities import FrameObservation
from app.features.visual_analysis.service.analysis_api import VisualAnalysisAPI

logger = logging.getLogger(__name__)


class ImageDescriptionAPI:
    """
    Describes a standalone image for the index: the vision model says what it shows and what text it reads, OCR
    reads the text on its own, and only text both saw is kept. Where the appliance has no OCR (Serbian), the vision
    model's text is kept but labelled unverified. The rendered excerpt names who wrote each part of it.

    The VLM is loaded by the caller (a phase, once per batch) and passed in; this API never acquires or releases it.
    """

    def __init__(
            self,
            repository: IImageDescriptionRepository,
            storage_api: StorageAPI,
            visual_api: VisualAnalysisAPI,
            conversion_api: DocumentConversionAPI
    ):
        self._repo = repository
        self._storage = storage_api
        self._visual = visual_api
        self._conversion = conversion_api

    def execute(self, source_id: uuid.UUID, vlm_model: LoadedModel) -> ImageDescription:
        try:
            existing = self._repo.get_by_source_id(source_id)
            if existing:
                logger.info(f"Image description exists for Source {source_id}. Skipping.")
                return existing

            path = self._source_path(source_id)
            width, height, taken_at = self._metadata(path)
            observation = self._visual.describe_image(path, vlm_model)
            ocr = self._conversion.read_image(path, self._line_selector(observation))
            description = self._describe(source_id, path.name, observation, ocr, width, height, taken_at)

            if not (description.shows or description.ocr_text.strip() or description.unverified_items):
                raise EmptyImageError(f"Nothing was recognised in {path.name}: no description and no text.")
            if description.ocr_withheld_lines:
                logger.warning(f"{path.name}: {len(description.ocr_withheld_lines)} low-confidence OCR line(s) the "
                               f"vision model did not read are not indexed: {description.ocr_withheld_lines}")
            if description.dropped_items:
                logger.warning(f"{path.name}: {len(description.dropped_items)} text item(s) the vision model read "
                               f"were not found by OCR and are not indexed: {description.dropped_items}")
            return self._repo.save(description)

        except Exception as e:
            raise map_image_description_error(e, context=f"ImageDescriptionAPI.execute for Source {source_id}") from e

    @staticmethod
    def render(description: ImageDescription) -> str:
        """The image's citable excerpt, labelled in the appliance's language."""
        return render_excerpt(description, get_language_pack().images)

    @staticmethod
    def render_context(description: ImageDescription) -> str:
        return render_context(description)

    def release(self) -> None:
        """Drops the OCR models once a batch is done."""
        self._conversion.release()

    def _source_path(self, source_id: uuid.UUID) -> Path:
        source = self._storage.get_source(source_id)
        if source is None or source.file is None:
            raise SourceImageMissingError(f"Source {source_id} has no file record.")
        path = Path(source.file.file_path)
        if not path.is_file():
            raise SourceImageMissingError(f"Image file missing from the vault: {path}")
        return path

    @staticmethod
    def _metadata(path: Path) -> Tuple[int, int, Optional[str]]:
        # Opening the image here also rejects a file PIL can't decode before the VLM is asked about it
        with Image.open(path) as image:
            return image.width, image.height, date_taken(image)

    @staticmethod
    def _line_selector(observation: Optional[FrameObservation]):
        """Keeps an OCR line when OCR was confident, or when the vision model read the same words."""
        vision_reading = OcrReading(" \n ".join(observation.legible_text) if observation else "",
                                    get_language_pack().transcription.script)
        return lambda lines: select_ocr_lines(
            lines, vision_reading,
            min_coverage=settings.IMAGE_TEXT_MIN_OCR_COVERAGE,
            min_confidence=settings.IMAGE_OCR_MIN_LINE_CONFIDENCE,
            unreliable_share=settings.IMAGE_OCR_UNRELIABLE_SHARE,
            unreliable_min_lines=settings.IMAGE_OCR_UNRELIABLE_MIN_LINES,
            unreliable_confidence=settings.IMAGE_OCR_UNRELIABLE_LINE_CONFIDENCE,
        )

    @staticmethod
    def _describe(source_id: uuid.UUID, file_name: str, observation: Optional[FrameObservation],
                  ocr: Optional[OcrPage], width: int, height: int, taken_at: Optional[str]) -> ImageDescription:
        items = list(observation.legible_text) if observation else []
        shows = observation.shows if observation else ""
        marks = observation.marks if observation else None
        common = dict(source_id=source_id, file_name=file_name, width=width, height=height, taken_at=taken_at)

        if ocr is None:
            # No OCR on this appliance: the vision model's reading stands alone, and says so
            return ImageDescription(shows=shows, marks=marks, unverified_items=items, **common)

        reading = OcrReading(ocr.text, get_language_pack().transcription.script)
        coverage = settings.IMAGE_TEXT_MIN_OCR_COVERAGE
        verified, dropped = verify_items(items, reading, coverage)
        return ImageDescription(
            shows=strip_unconfirmed(shows, reading, coverage) or "",
            marks=strip_unconfirmed(marks, reading, coverage),
            ocr_text=ocr.text,
            ocr_lines_detected=ocr.lines_detected,
            ocr_lines_read=ocr.lines_read,
            ocr_withheld_lines=list(ocr.withheld_lines),
            ocr_checked=True,
            verified_items=verified,
            dropped_items=dropped,
            **common,
        )
