# Path: integration_tests/features/document_conversion/test_books_and_texts_integration.py

"""
Books and texts through the real adapters and Postgres: charset detection on a cp1250 text file, MarkItDown's
EPUB and HTML converters, striprtf, then structuring into line- or section-anchored chunks.

A plain-text chunk's line range is checked against the file itself: open the file, read those lines, find the
chunk's text there. That is the promise a "lines 120–164" citation makes.

Fixtures are generated here rather than checked in, like the PDF/DOCX ones.
"""

import zipfile
from pathlib import Path

import pytest

from app.core.anchors.domain.enums import AnchorKind
from app.features.document_conversion.data.markitdown_converter import MarkItDownConverter
from app.features.document_conversion.data.pdfium_reader import PdfiumReader
from app.features.document_conversion.data.plain_text_decoder import CharsetTextDecoder
from app.features.document_conversion.data.repositories import ConvertedDocumentRepository
from app.features.document_conversion.data.rtf_reader import StripRtfReader
from app.features.document_conversion.domain.enums import DocumentKind
from app.features.document_conversion.service.conversion_api import DocumentConversionAPI
from app.features.document_structuring.service.structuring_api import DocumentStructuringAPI
from app.features.storage.data.repositories import StorageRepository
from app.features.storage.domain.entities import SourceType
from app.features.storage.service.storage_api import StorageAPI

pytestmark = pytest.mark.integration

# Hard-wrapped like a Project Gutenberg file, with its blank-line runs, and in cp1250 like many Slovenian texts
NOVEL = (
    "\n"
    "THE LIGHTHOUSE KEEPER\n"
    "\n"
    "\n"
    "CHAPTER I.\n"
    "\n"
    "The keeper of the northern light was a man called Tomaž, who had not\n"
    "left the island in eleven winters. Every evening at dusk he climbed\n"
    "the hundred and twelve steps and lit the lamp.\n"
    "\n"
    "\n"
    "\n"
    "CHAPTER II.\n"
    "\n"
    "In the spring a ship ran aground on the reef below the tower. Its\n"
    "captain, a Frenchman named Čedomir Lacroix, swore the lamp had been dark.\n"
)


def _epub(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container version="1.0" '
                   'xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles>'
                   '<rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>'
                   '</rootfiles></container>')
        z.writestr("OEBPS/content.opf",
                   '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>The Lighthouse</dc:title>'
                   '<dc:creator>A. Keeper</dc:creator></metadata><manifest>'
                   '<item id="c1" href="c1.xhtml" media-type="application/xhtml+xml"/>'
                   '<item id="c2" href="c2.xhtml" media-type="application/xhtml+xml"/></manifest>'
                   '<spine><itemref idref="c1"/><itemref idref="c2"/></spine></package>')
        for name, title, body in (("c1", "Chapter 1", "The keeper lit the lamp at dusk."),
                                  ("c2", "Chapter 2", "A ship ran aground on the northern reef.")):
            z.writestr(f"OEBPS/{name}.xhtml",
                       f'<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>{title}</h1><p>{body}</p></body></html>')


@pytest.fixture
def storage_api(db_manager):
    return StorageAPI(StorageRepository(db_manager))


@pytest.fixture
def conversion_api(db_manager, storage_api):
    return DocumentConversionAPI(ConvertedDocumentRepository(db_manager), storage_api, PdfiumReader(),
                                 MarkItDownConverter(), ocr_engine=None, text_decoder=CharsetTextDecoder(),
                                 rtf_reader=StripRtfReader())


def _structure(path: Path, storage_api, conversion_api):
    source = storage_api.ingest_local_file(str(path), SourceType.DOCUMENT)
    document = conversion_api.execute(source.id)
    return document, DocumentStructuringAPI().execute(document)


class TestBooksAndTextsIntegration:
    def test_cp1250_text_is_decoded_and_cites_the_files_own_lines(self, tmp_path, storage_api, conversion_api,
                                                                  monkeypatch):
        from app.core.config import settings
        monkeypatch.setattr(settings, "DOC_SHORT_DOCUMENT_WORDS", 10)
        path = tmp_path / "novel.txt"
        path.write_bytes(NOVEL.encode("cp1250"))

        document, structured = _structure(path, storage_api, conversion_api)

        assert document.kind is DocumentKind.TXT
        assert structured.anchor_kind is AnchorKind.LINE
        second = next(c for c in structured.chunks if "ran aground" in c.text)
        assert second.section_path == "THE LIGHTHOUSE KEEPER › CHAPTER II."
        assert "Čedomir Lacroix" in second.text

        # The cited lines, read back from the file a reader would open, hold the chunk's text
        file_lines = path.read_bytes().decode("cp1250").split("\n")
        cited = " ".join(file_lines[second.line_start - 1:second.line_end])
        assert (second.line_start, second.line_end) == (13, 16)
        assert " ".join(second.text.split()) == " ".join(cited.split())

    def test_markdown_is_line_anchored_under_its_headings(self, tmp_path, storage_api, conversion_api):
        path = tmp_path / "notes.md"
        path.write_text("# Case notes\n\n\n\n## Witnesses\n\nThe witness saw the car at 9 pm.\n", encoding="utf-8")

        _, structured = _structure(path, storage_api, conversion_api)

        assert structured.anchor_kind is AnchorKind.LINE
        [chunk] = structured.chunks
        assert (chunk.line_start, chunk.line_end) == (1, 7)
        assert chunk.section_path == "Case notes › Witnesses"

    def test_epub_chapters_become_sections_in_spine_order(self, tmp_path, storage_api, conversion_api, monkeypatch):
        from app.core.config import settings
        monkeypatch.setattr(settings, "DOC_SHORT_DOCUMENT_WORDS", 5)
        path = tmp_path / "lighthouse.epub"
        _epub(path)

        document, structured = _structure(path, storage_api, conversion_api)

        assert document.kind is DocumentKind.EPUB and document.pages[0].page_no is None
        assert structured.anchor_kind is AnchorKind.SECTION
        assert [c.section_path for c in structured.chunks if "ship" in c.text or "lamp" in c.text] == [
            "Chapter 1", "Chapter 2"]
        assert "The Lighthouse" in structured.chunks[0].text

    def test_html_and_rtf_are_section_anchored(self, tmp_path, storage_api, conversion_api):
        html = tmp_path / "page.html"
        html.write_text("<html><head><script>track()</script></head><body><h2>Section A</h2>"
                        "<p>The deposit is refundable.</p></body></html>", encoding="utf-8")
        rtf = tmp_path / "knjiga.rtf"
        rtf.write_bytes(rb"{\rtf1\ansi\ansicpg1250{\fonttbl{\f0 Times;}}\f0 Glava 1\par Najemnik pla\'e8a.\par}")

        html_doc, html_structured = _structure(html, storage_api, conversion_api)
        rtf_doc, rtf_structured = _structure(rtf, storage_api, conversion_api)

        assert html_doc.kind is DocumentKind.HTML and "track()" not in html_doc.pages[0].text
        assert html_structured.chunks[0].section_path == "Section A"
        assert rtf_doc.kind is DocumentKind.RTF and "Najemnik plača." in rtf_doc.pages[0].text
        assert rtf_structured.anchor_kind is AnchorKind.SECTION
