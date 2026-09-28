# Path: app/features/document_conversion/data/pdfium_reader.py

import io

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image

from app.features.document_conversion.domain.interfaces import IPdfDocument, IPdfReader


class PdfiumDocument(IPdfDocument):
    """A PDF opened with pypdfium2 (PDFium, Apache/BSD). Pages are indexed from 0."""

    def __init__(self, pdf: pdfium.PdfDocument):
        self._pdf = pdf

    def page_count(self) -> int:
        return len(self._pdf)

    def page_as_pdf(self, index: int) -> bytes:
        single = pdfium.PdfDocument.new()
        try:
            single.import_pages(self._pdf, [index])
            buffer = io.BytesIO()
            single.save(buffer)
            return buffer.getvalue()
        finally:
            single.close()

    def image_coverage(self, index: int) -> float:
        page = self._pdf[index]
        try:
            width, height = page.get_size()
            if width <= 0 or height <= 0:
                return 0.0
            covered = 0.0
            for image in page.get_objects(filter=[pdfium_c.FPDF_PAGEOBJ_IMAGE], max_depth=3):
                left, bottom, right, top = image.get_bounds()
                # Clip to the page: a scan is often placed slightly larger than the media box
                covered += max(0.0, min(right, width) - max(left, 0.0)) * max(0.0, min(top, height) - max(bottom, 0.0))
            return min(1.0, covered / (width * height))
        finally:
            page.close()

    def render_page(self, index: int, dpi: int) -> Image.Image:
        page = self._pdf[index]
        try:
            bitmap = page.render(scale=dpi / 72)
            try:
                return bitmap.to_pil().convert("RGB")
            finally:
                bitmap.close()
        finally:
            page.close()

    def close(self) -> None:
        self._pdf.close()


class PdfiumReader(IPdfReader):
    def open(self, path: str) -> IPdfDocument:
        return PdfiumDocument(pdfium.PdfDocument(path))
