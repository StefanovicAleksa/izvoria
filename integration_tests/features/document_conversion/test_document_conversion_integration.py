# Path: integration_tests/features/document_conversion/test_document_conversion_integration.py

"""
Real files through the real adapters: pypdfium2 page splitting, MarkItDown per page, RapidOCR on a scanned page,
and the converted document stored in and read back from Postgres, then structured into page-anchored chunks.

Fixtures are generated here rather than checked in: a born-digital PDF from Qt's own PDF writer (PySide6 ships
with the desktop app), a scanned page as an image-only PDF, and a minimal DOCX zip.
"""

import os
import sys
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFont

from app.core.anchors.domain.enums import AnchorKind
from app.core.config import settings
from app.features.document_conversion.data.markitdown_converter import MarkItDownConverter
from app.features.document_conversion.data.pdfium_reader import PdfiumReader
from app.features.document_conversion.data.pdfplumber_layout import PdfplumberLayoutExtractor
from app.features.document_conversion.data.rapidocr_engine import (
    CLASSIFIER_FILE, DETECTOR_FILE, RECOGNIZER_FILES, RapidOcrEngine,
)
from app.features.document_conversion.data.repositories import ConvertedDocumentRepository
from app.features.document_conversion.domain.enums import DocumentKind, ExtractionMethod
from app.features.document_conversion.service.conversion_api import DocumentConversionAPI
from app.features.document_structuring.service.structuring_api import DocumentStructuringAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import SourceType
from app.features.storage.service.storage_api import StorageAPI

pytestmark = pytest.mark.integration

OCR_DIR = settings.BASE_DIR / settings.DOC_OCR_MODEL_DIR
FONT = Path("/usr/share/fonts/truetype/noto/NotoSerif-Regular.ttf")

LEASE_PAGES = [
    "<h1>COMMERCIAL LEASE AGREEMENT</h1><p>This Agreement is made between Acme Properties Ltd (the Landlord) "
    "and Beta Consulting d.o.o. (the Tenant), for the office space at 12 Main Street, Ljubljana.</p>"
    "<h2>Article 1 Definitions</h2><p>In this Agreement the Premises means the office space described above, "
    "including the parking place allotted to it.</p>",
    "<h2>Article 7 Termination</h2><p>7.1 The Tenant may terminate this Agreement by giving the Landlord not less "
    "than thirty (30) days' prior written notice.</p><p>7.2 The Landlord may terminate this Agreement immediately "
    "if the Tenant fails to pay rent for two consecutive months.</p>",
    "<h2>Article 9 Rent</h2><table border=1><tr><th>Period</th><th>Monthly rent</th></tr>"
    "<tr><td>2025</td><td>EUR 1,250.00</td></tr><tr><td>2026</td><td>EUR 1,300.00</td></tr></table>",
]
INVOICE_HTML = """
<h2>INVOICE No. INV-2025-0412</h2>
<p>Issued by Acme Properties Ltd to Beta Consulting d.o.o. on 5 November 2025.</p>
<table border=0 cellpadding=10>
<tr><th align=left>Item</th><th align=left>Quantity</th><th align=left>Unit price</th><th align=left>Amount</th></tr>
<tr><td>Office rent November</td><td>1</td><td>EUR 1,250.00</td><td>EUR 1,250.00</td></tr>
<tr><td>Parking place</td><td>2</td><td>EUR 40.00</td><td>EUR 80.00</td></tr>
<tr><td>Cleaning service</td><td>4</td><td>EUR 25.00</td><td>EUR 100.00</td></tr>
</table>
<p>Payment details:</p>
<table border=0 cellpadding=10>
<tr><td>IBAN</td><td>SI56 1910 0000 0123 438</td></tr>
<tr><td>Due date</td><td>20 November 2025</td></tr>
<tr><td>Total due</td><td>EUR 1,430.00</td></tr>
</table>
<p>Late payments accrue statutory default interest.</p>
"""
LINE_ITEMS = ("| Item | Quantity | Unit price | Amount |\n| --- | --- | --- | --- |\n"
              "| Office rent November | 1 | EUR 1,250.00 | EUR 1,250.00 |\n"
              "| Parking place | 2 | EUR 40.00 | EUR 80.00 |\n"
              "| Cleaning service | 4 | EUR 25.00 | EUR 100.00 |")
PAYMENT = ("| IBAN | SI56 1910 0000 0123 438 |\n| --- | --- |\n| Due date | 20 November 2025 |\n"
           "| Total due | EUR 1,430.00 |")
SCANNED_LINES = ["POGODBA O NAJEMU", "3. člen", "Najemnik plača najemnino do petega dne v mesecu.",
                 "Za čas zamude se zaračunajo zakonske zamudne obresti."]


def _born_digital_pdf(path: Path, html: str = None) -> None:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication, QPageSize, QPdfWriter, QTextDocument

    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841 -- Qt needs one alive
    writer = QPdfWriter(str(path))
    writer.setPageSize(QPageSize(QPageSize.A4))
    # 72 dpi: Qt then writes text at its real point size (at 300 it came out at ~3 pt)
    writer.setResolution(72)
    document = QTextDocument()
    document.setHtml(html or "<p style='page-break-after:always'></p>".join(LEASE_PAGES))
    document.setPageSize(writer.pageLayout().paintRectPixels(writer.resolution()).size().toSizeF())
    document.print_(writer)
    del writer


def _scanned_pdf(path: Path) -> None:
    font = ImageFont.truetype(str(FONT), 40)
    image = Image.new("RGB", (2480, 3508), "white")
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(SCANNED_LINES):
        draw.text((180, 250 + i * 80), line, font=font, fill="black")
    image.save(path, "PDF", resolution=300)


def _docx(path: Path) -> None:
    body = (
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Hearing memo</w:t></w:r></w:p>'
        '<w:p><w:r><w:t>The hearing in case P-1234/2025 moved to Monday 3 November.</w:t></w:r></w:p>'
    )
    ns = 'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                   '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
                   '</Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                   '</Relationships>')
        z.writestr("word/_rels/document.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                   '</Relationships>')
        z.writestr("word/styles.xml",
                   f'<?xml version="1.0" encoding="UTF-8"?><w:styles {ns}><w:style w:type="paragraph" w:styleId="Heading1">'
                   '<w:name w:val="heading 1"/></w:style></w:styles>')
        z.writestr("word/document.xml",
                   f'<?xml version="1.0" encoding="UTF-8"?><w:document {ns}><w:body>{body}</w:body></w:document>')


def _ocr_models_present() -> bool:
    return all((OCR_DIR / name).is_file() for name in (DETECTOR_FILE, CLASSIFIER_FILE, RECOGNIZER_FILES["latin"]))


@pytest.fixture
def storage_api(db_manager):
    return StorageAPI(StorageRepository(db_manager))


@pytest.fixture
def conversion_api(db_manager, storage_api):
    ocr = RapidOcrEngine(str(OCR_DIR), "latin")
    api = DocumentConversionAPI(ConvertedDocumentRepository(db_manager), storage_api, PdfiumReader(),
                                MarkItDownConverter(), ocr, PdfplumberLayoutExtractor())
    yield api
    api.release()


class TestDocumentConversionIntegration:
    def test_born_digital_pdf_keeps_every_page_and_exact_text(self, tmp_path, storage_api, conversion_api):
        path = tmp_path / "lease.pdf"
        _born_digital_pdf(path)
        source = storage_api.ingest_local_file(str(path), SourceType.DOCUMENT)

        document = conversion_api.execute(source.id)

        assert document.kind is DocumentKind.PDF
        assert [p.page_no for p in document.pages] == [1, 2, 3]
        assert all(p.method is ExtractionMethod.TEXT_LAYER for p in document.pages)
        # Ligatures expanded ("fi" in Definitions), tabs gone, the sparse table page not replaced by OCR
        assert "Article 1 Definitions" in document.pages[0].text
        # Lines wrap at the page width; structuring joins them
        assert "thirty (30) days' prior written notice" in " ".join(document.pages[1].text.split())
        assert "2025" in document.pages[2].text and "EUR 1,300.00" in document.pages[2].text

    def test_conversion_round_trips_through_postgres_and_is_not_redone(self, tmp_path, db_manager, storage_api,
                                                                      conversion_api):
        path = tmp_path / "lease.pdf"
        _born_digital_pdf(path)
        source = storage_api.ingest_local_file(str(path), SourceType.DOCUMENT)
        first = conversion_api.execute(source.id)

        stored = ConvertedDocumentRepository(db_manager).get_by_source_id(source.id)

        assert stored.id == first.id and stored.pages == first.pages
        assert conversion_api.execute(source.id).id == first.id

    def test_chunks_carry_the_page_they_came_from(self, tmp_path, storage_api, conversion_api, monkeypatch):
        monkeypatch.setattr(settings, "DOC_SHORT_DOCUMENT_WORDS", 10)
        path = tmp_path / "lease.pdf"
        _born_digital_pdf(path)
        source = storage_api.ingest_local_file(str(path), SourceType.DOCUMENT)

        structured = DocumentStructuringAPI().execute(conversion_api.execute(source.id))

        assert structured.anchor_kind is AnchorKind.PAGE
        termination = next(c for c in structured.chunks if "thirty (30) days" in c.text)
        assert (termination.page_start, termination.page_end) == (2, 2)
        assert termination.section_path.endswith("Article 7 Termination")
        rent = next(c for c in structured.chunks if "EUR 1,250.00" in c.text)
        assert rent.page_start == 3 and "2025" in rent.text

    @pytest.mark.skipif(not FONT.is_file(), reason="Noto Serif is needed to render the synthetic scan")
    def test_scanned_slovenian_page_is_read_by_ocr(self, tmp_path, storage_api, conversion_api):
        if not _ocr_models_present():
            pytest.skip(f"RapidOCR models not in {OCR_DIR}; run `python -m scripts.download_ocr_models`")
        path = tmp_path / "scan.pdf"
        _scanned_pdf(path)
        source = storage_api.ingest_local_file(str(path), SourceType.DOCUMENT)

        [page] = conversion_api.execute(source.id).pages

        assert page.method is ExtractionMethod.OCR
        assert page.ocr_lines_read == page.ocr_lines_detected == len(SCANNED_LINES)
        assert "Najemnik plača najemnino do petega dne v mesecu." in page.text
        assert "zaračunajo" in page.text

    def test_docx_is_one_section_anchored_document(self, tmp_path, storage_api, conversion_api):
        path = tmp_path / "memo.docx"
        _docx(path)
        source = storage_api.ingest_local_file(str(path), SourceType.DOCUMENT)

        document = conversion_api.execute(source.id)
        structured = DocumentStructuringAPI().execute(document)

        assert document.kind is DocumentKind.DOCX and document.pages[0].page_no is None
        assert structured.anchor_kind is AnchorKind.SECTION
        [chunk] = structured.chunks
        assert chunk.section_path == "Hearing memo"
        assert "P-1234/2025 moved to Monday 3 November" in chunk.text


class TestTablesIntegration:
    def test_born_digital_tables_keep_their_rows(self, tmp_path, storage_api, conversion_api):
        path = tmp_path / "invoice.pdf"
        _born_digital_pdf(path, INVOICE_HTML)
        source = storage_api.ingest_local_file(str(path), SourceType.DOCUMENT)

        [page] = conversion_api.execute(source.id).pages

        assert LINE_ITEMS in page.text and PAYMENT in page.text
        assert "Payment details:\n\n| IBAN |" in page.text

    def test_ruled_table_is_exact(self, tmp_path, storage_api, conversion_api):
        path = tmp_path / "lease.pdf"
        _born_digital_pdf(path)
        source = storage_api.ingest_local_file(str(path), SourceType.DOCUMENT)

        pages = conversion_api.execute(source.id).pages

        assert ("| Period | Monthly rent |\n| --- | --- |\n| 2025 | EUR 1,250.00 |\n| 2026 | EUR 1,300.00 |"
                in pages[2].text)
        assert "|" not in pages[0].text and "|" not in pages[1].text

    def test_a_scanned_invoice_keeps_its_rows(self, tmp_path, storage_api, conversion_api):
        if not _ocr_models_present():
            pytest.skip(f"RapidOCR models not in {OCR_DIR}; run `python -m scripts.download_ocr_models`")
        digital = tmp_path / "invoice.pdf"
        _born_digital_pdf(digital, INVOICE_HTML)
        with PdfiumReader().open(str(digital)) as pdf:
            scan = tmp_path / "invoice_scan.pdf"
            pdf.render_page(0, 300).save(scan, "PDF", resolution=300)
        source = storage_api.ingest_local_file(str(scan), SourceType.DOCUMENT)

        [page] = conversion_api.execute(source.id).pages

        assert page.method is ExtractionMethod.OCR
        assert LINE_ITEMS in page.text and PAYMENT in page.text

    def test_table_survives_structuring_as_one_block(self, tmp_path, storage_api, conversion_api):
        path = tmp_path / "invoice.pdf"
        _born_digital_pdf(path, INVOICE_HTML)
        source = storage_api.ingest_local_file(str(path), SourceType.DOCUMENT)

        [chunk] = DocumentStructuringAPI().execute(conversion_api.execute(source.id)).chunks

        assert LINE_ITEMS in chunk.text
