# Path: tests/features/document_structuring/service/test_structuring_api.py

import uuid

import pytest

from app.core.anchors.domain.enums import AnchorKind
from app.features.document_conversion.domain.entities import ConvertedDocument, PageContent
from app.features.document_conversion.domain.enums import DocumentKind, ExtractionMethod
from app.features.document_structuring.exceptions import EmptyDocumentError
from app.features.document_structuring.service.structuring_api import DocumentStructuringAPI


def _pdf(*texts):
    return ConvertedDocument(source_id=uuid.uuid4(), kind=DocumentKind.PDF,
                             pages=[PageContent(page_no=i + 1, text=t) for i, t in enumerate(texts)])


class TestDocumentStructuringAPI:
    def test_pdf_chunks_are_page_anchored(self):
        structured = DocumentStructuringAPI().execute(_pdf("Article 1 Terms\n\nThe rent is due monthly.",
                                                           "Article 2 Notice\n\nNotice must be written."))

        assert structured.anchor_kind is AnchorKind.PAGE
        assert structured.chunks[0].page_start == 1 and structured.chunks[-1].page_end == 2

    def test_docx_chunks_are_section_anchored(self):
        document = ConvertedDocument(source_id=uuid.uuid4(), kind=DocumentKind.DOCX,
                                     pages=[PageContent(text="# Memo\n\nThe hearing moved.")])

        structured = DocumentStructuringAPI().execute(document)

        assert structured.anchor_kind is AnchorKind.SECTION
        assert structured.chunks[0].section_path == "Memo"

    def test_unreadable_pages_are_skipped(self):
        document = ConvertedDocument(source_id=uuid.uuid4(), kind=DocumentKind.PDF, pages=[
            PageContent(page_no=1, text="", method=ExtractionMethod.NONE),
            PageContent(page_no=2, text="Page two text."),
        ])

        [chunk] = DocumentStructuringAPI().execute(document).chunks

        assert (chunk.page_start, chunk.text) == (2, "Page two text.")

    def test_document_with_no_text_raises(self):
        with pytest.raises(EmptyDocumentError):
            DocumentStructuringAPI().execute(_pdf("", "  "))

    def test_slovenian_clauses_split_the_document(self, appliance_language, monkeypatch):
        appliance_language("sl")
        monkeypatch.setattr("app.features.document_structuring.service.structuring_api.settings.DOC_SHORT_DOCUMENT_WORDS", 5)

        structured = DocumentStructuringAPI().execute(_pdf(
            "POGODBA O NAJEMU\n1. člen\nNajemodajalec odda najemniku poslovni prostor.\n"
            "2. člen\nNajemnik plača najemnino do petega v mesecu."))

        assert [c.section_path for c in structured.chunks] == [
            "POGODBA O NAJEMU › 1. člen", "POGODBA O NAJEMU › 2. člen"]
        assert structured.chunks[1].text == "2. člen\n\nNajemnik plača najemnino do petega v mesecu."

    def test_serbian_cyrillic_is_indexed_in_latin(self, appliance_language):
        appliance_language("sr")

        [chunk] = DocumentStructuringAPI().execute(_pdf("Закупац плаћа закупнину.")).chunks

        assert chunk.text == "Zakupac plaća zakupninu."
