# Path: app/features/document_conversion/service/conversion_api.py

import logging
import uuid
from pathlib import Path
from typing import List, Optional

from PIL import Image

from app.core.config import settings
from app.features.document_conversion.domain.entities import ConvertedDocument, OcrPage, PageContent
from app.features.document_conversion.domain.enums import DocumentKind, ExtractionMethod
from app.features.document_conversion.domain.interfaces import (
    IConvertedDocumentRepository, IMarkdownConverter, IOcrEngine, IPageLayoutExtractor, IPdfDocument, IPdfReader,
    IRtfReader, ITextDecoder,
)
from app.features.document_conversion.exceptions import (
    OcrModelsMissingError, SourceDocumentMissingError, UnsupportedDocumentError,
)
from app.features.document_conversion.utils.error_mapper import map_conversion_error
from app.features.document_conversion.utils.text import count_letters, normalize_extracted_text, normalize_text_lines
from app.features.storage.service.storage_api import StorageAPI

logger = logging.getLogger(__name__)

# Every suffix the document pipeline ingests; the pipeline's own filter is built from this
DOCUMENT_KINDS = {
    ".pdf": DocumentKind.PDF,
    ".docx": DocumentKind.DOCX,
    ".txt": DocumentKind.TXT,
    ".md": DocumentKind.MARKDOWN,
    ".markdown": DocumentKind.MARKDOWN,
    ".html": DocumentKind.HTML,
    ".htm": DocumentKind.HTML,
    ".rtf": DocumentKind.RTF,
    ".epub": DocumentKind.EPUB,
}
# Read line for line, so a chunk can cite the file's own line numbers
LINE_KINDS = {DocumentKind.TXT, DocumentKind.MARKDOWN}
# Converted to Markdown by MarkItDown, whose headings (`#`) become the sections a citation names
MARKDOWN_CONVERTED_KINDS = {DocumentKind.DOCX: ".docx", DocumentKind.HTML: ".html", DocumentKind.EPUB: ".epub"}


class DocumentConversionAPI:
    """
    Turns an ingested document -- PDF, DOCX, plain text, Markdown, HTML, RTF or an EPUB book -- into page-tagged
    Markdown.

    A PDF is converted one page at a time: MarkItDown on a whole PDF joins the pages and drops the empty ones, so
    neither page numbers nor scanned pages would survive. A page is treated as scanned only when its text layer
    has next to no letters AND an image covers most of it; it is then handed to OCR -- or, on an appliance whose
    language has no OCR model, recorded as unreadable rather than indexed with guessed text. A sparse
    born-digital page keeps its exact text layer.

    Everything else has no pages and becomes one `PageContent(page_no=None)`. Plain text and Markdown bypass
    MarkItDown: it collapses runs of blank lines, which would shift every line number a citation shows. They are
    decoded (any encoding) and cleaned line by line instead, keeping line N of the result on line N of the file.
    """

    def __init__(
            self,
            repository: IConvertedDocumentRepository,
            storage_api: StorageAPI,
            pdf_reader: IPdfReader,
            markdown_converter: IMarkdownConverter,
            ocr_engine: Optional[IOcrEngine],
            layout_extractor: Optional[IPageLayoutExtractor] = None,
            text_decoder: Optional[ITextDecoder] = None,
            rtf_reader: Optional[IRtfReader] = None
    ):
        self._repo = repository
        self._storage = storage_api
        self._pdf_reader = pdf_reader
        self._converter = markdown_converter
        self._ocr = ocr_engine
        self._layout = layout_extractor
        self._text_decoder = text_decoder
        self._rtf_reader = rtf_reader

    def execute(self, source_id: uuid.UUID) -> ConvertedDocument:
        try:
            existing = self._repo.get_by_source_id(source_id)
            if existing:
                logger.info(f"Converted document exists for Source {source_id}. Skipping.")
                return existing

            path = self._source_path(source_id)
            kind = DOCUMENT_KINDS.get(path.suffix.lower())
            if kind is None:
                raise UnsupportedDocumentError(f"Not a supported document type: {path.name}")

            if kind is DocumentKind.PDF:
                pages = self._convert_pdf(path)
            else:
                pages = [PageContent(page_no=None, text=self._convert_whole(kind, path.read_bytes()))]

            document = ConvertedDocument(source_id=source_id, kind=kind, pages=pages)
            logger.info(
                f"Converted {path.name}: {len(pages)} page(s), {document.pages_ocr} by OCR, "
                f"{document.pages_ocr_incomplete} with OCR lines lost, {document.pages_unreadable} unreadable."
            )
            return self._repo.save(document)

        except Exception as e:
            raise map_conversion_error(e, context=f"DocumentConversionAPI.execute for Source {source_id}") from e

    def read_image(self, image_path: Path) -> Optional[OcrPage]:
        """
        OCR of a standalone image file, for features that index images. None when this appliance has no OCR
        model for its language, or its models are not installed: the caller then has no reading to check against,
        which is its to report. The image is downscaled to `IMAGE_OCR_MAX_SIDE` on its long side first.
        """
        if self._ocr is None:
            return None
        try:
            with Image.open(image_path) as image:
                image.seek(0)  # a multi-page TIFF: its first page
                rgb = image.convert("RGB")
            longest = max(rgb.size)
            if longest > settings.IMAGE_OCR_MAX_SIDE:
                scale = settings.IMAGE_OCR_MAX_SIDE / float(longest)
                rgb = rgb.resize((max(1, int(rgb.width * scale)), max(1, int(rgb.height * scale))),
                                 Image.Resampling.LANCZOS)
            return self._ocr.recognize(rgb)
        except OcrModelsMissingError as e:
            logger.error(str(e))
            return None
        except Exception as e:
            raise map_conversion_error(e, context=f"DocumentConversionAPI.read_image for {image_path.name}") from e

    def release(self) -> None:
        """Drops the OCR engine's models; called once a batch is done."""
        if self._ocr is not None:
            self._ocr.close()

    def _source_path(self, source_id: uuid.UUID) -> Path:
        source = self._storage.get_source(source_id)
        if source is None or source.file is None:
            raise SourceDocumentMissingError(f"Source {source_id} has no file record.")
        path = Path(source.file.file_path)
        if not path.is_file():
            raise SourceDocumentMissingError(f"Document file missing from the vault: {path}")
        return path

    def _convert_whole(self, kind: DocumentKind, data: bytes) -> str:
        if kind in LINE_KINDS:
            if self._text_decoder is None:
                raise UnsupportedDocumentError("No text decoder is configured for plain-text documents.")
            return normalize_text_lines(self._text_decoder.decode(data))
        if kind is DocumentKind.RTF:
            if self._rtf_reader is None:
                raise UnsupportedDocumentError("No RTF reader is configured.")
            return normalize_extracted_text(self._rtf_reader.to_text(data))
        return normalize_extracted_text(self._converter.convert(data, MARKDOWN_CONVERTED_KINDS[kind]))

    def _convert_pdf(self, path: Path) -> List[PageContent]:
        with self._pdf_reader.open(str(path)) as pdf:
            return [self._convert_page(pdf, index, path.name) for index in range(pdf.page_count())]

    def _convert_page(self, pdf: IPdfDocument, index: int, file_name: str) -> PageContent:
        page_no = index + 1
        page_pdf = pdf.page_as_pdf(index)
        text = normalize_extracted_text(self._converter.convert(page_pdf, ".pdf"))
        if (count_letters(text) >= settings.DOC_MIN_TEXT_LETTERS_PER_PAGE
                or pdf.image_coverage(index) < settings.DOC_SCANNED_IMAGE_COVERAGE):
            # An exact text layer, even a sparse one, is never replaced by OCR's reading of the same page.
            # A page with tables is re-read with its tables as Markdown: MarkItDown reads a borderless table
            # column by column, which separates every value from the row it belongs to.
            with_tables = self._layout.extract(page_pdf) if self._layout is not None else None
            if with_tables:
                text = normalize_extracted_text(with_tables)
            return PageContent(page_no=page_no, text=text, method=ExtractionMethod.TEXT_LAYER)

        # Next to no text layer over a page-sized image: a scan
        if self._ocr is None:
            logger.warning(f"{file_name} page {page_no} has no text layer and this appliance has no OCR.")
            return PageContent(page_no=page_no, text=text, method=ExtractionMethod.NONE,
                               note="scanned page; no OCR model for this appliance's language")
        try:
            ocr = self._ocr.recognize(pdf.render_page(index, settings.DOC_OCR_DPI))
        except OcrModelsMissingError as e:
            # A deployment error, reported on every page it affects rather than failing the document
            logger.error(str(e))
            return PageContent(page_no=page_no, text=text, method=ExtractionMethod.NONE,
                               note="scanned page; OCR models are not installed")

        if ocr.lines_detected == 0:
            # A blank page (a separator, the back of a sheet): OCR ran and there was nothing to read
            return PageContent(page_no=page_no, text=text, method=ExtractionMethod.OCR)
        if not ocr.text.strip():
            return PageContent(page_no=page_no, text=text, method=ExtractionMethod.NONE,
                               ocr_lines_detected=ocr.lines_detected, note="OCR could not read any line")
        if ocr.lines_read < ocr.lines_detected:
            logger.warning(
                f"OCR read {ocr.lines_read} of {ocr.lines_detected} detected lines on {file_name} page {page_no}; "
                f"the page text is incomplete."
            )
        return PageContent(
            page_no=page_no,
            text=ocr.text,
            method=ExtractionMethod.OCR,
            ocr_confidence=ocr.mean_confidence,
            ocr_lines_detected=ocr.lines_detected,
            ocr_lines_read=ocr.lines_read,
        )
