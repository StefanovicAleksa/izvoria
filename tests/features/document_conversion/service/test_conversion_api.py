# Path: tests/features/document_conversion/service/test_conversion_api.py

import uuid
from unittest.mock import MagicMock

import pytest
from PIL import Image

from app.features.document_conversion.domain.entities import ConvertedDocument, OcrPage, PageContent
from app.features.document_conversion.domain.enums import DocumentKind, ExtractionMethod
from app.features.document_conversion.exceptions import (
    DocumentConversionError, OcrModelsMissingError, SourceDocumentMissingError, UndecodableTextError,
    UnsupportedDocumentError,
)
from app.features.document_conversion.service.conversion_api import DocumentConversionAPI

BODY = "The Tenant may terminate this Agreement by giving the Landlord thirty days' written notice."


class FakePdf:
    """A PDF whose pages are (text layer, image coverage) pairs."""

    def __init__(self, pages):
        self.pages = pages
        self.closed = False

    def page_count(self):
        return len(self.pages)

    def page_as_pdf(self, index):
        return f"page-{index}".encode()

    def image_coverage(self, index):
        return self.pages[index][1]

    def render_page(self, index, dpi):
        return Image.new("RGB", (10, 10))

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


@pytest.fixture
def document_file(tmp_path):
    def make(name):
        path = tmp_path / name
        path.write_bytes(b"%PDF-1.7")
        return path
    return make


def _api(path, pdf=None, ocr=None, existing=None, layout=None, text_decoder=None, rtf_reader=None):
    repo = MagicMock()
    repo.get_by_source_id.return_value = existing
    repo.save.side_effect = lambda document: document
    storage = MagicMock()
    storage.get_source.return_value.file.file_path = str(path)
    reader = MagicMock()
    reader.open.return_value = pdf
    converter = MagicMock()
    if pdf is not None:
        converter.convert.side_effect = lambda data, ext: pdf.pages[int(data.decode().split("-")[1])][0]
    api = DocumentConversionAPI(repo, storage, reader, converter, ocr, layout, text_decoder, rtf_reader)
    return api, repo, converter


class TestPdfRouting:
    def test_text_layer_pages_keep_their_page_numbers(self, document_file):
        pdf = FakePdf([(BODY, 0.0), ("Article 7 Termination\n\n" + BODY, 0.0)])
        api, repo, converter = _api(document_file("lease.pdf"), pdf)

        document = api.execute(uuid.uuid4())

        assert [p.page_no for p in document.pages] == [1, 2]
        assert all(p.method is ExtractionMethod.TEXT_LAYER for p in document.pages)
        assert document.pages[1].text.startswith("Article 7 Termination")
        converter.convert.assert_any_call(b"page-0", ".pdf")
        converter.convert.assert_any_call(b"page-1", ".pdf")
        assert pdf.closed
        repo.save.assert_called_once_with(document)

    def test_sparse_born_digital_page_keeps_its_text_layer(self, document_file):
        """A heading and a small table: few letters, but no page-sized image, so it is not a scan."""
        ocr = MagicMock()
        pdf = FakePdf([("Article 9 Rent\n| 2025 | EUR 1,250 |", 0.0)])
        api, _, _ = _api(document_file("lease.pdf"), pdf, ocr)

        [page] = api.execute(uuid.uuid4()).pages

        assert page.method is ExtractionMethod.TEXT_LAYER
        assert page.text == "Article 9 Rent\n| 2025 | EUR 1,250 |"
        ocr.recognize.assert_not_called()

    def test_scanned_page_is_ocrd(self, document_file):
        ocr = MagicMock()
        ocr.recognize.return_value = OcrPage(text="Člen 3\nNajemnik plača najemnino.", mean_confidence=0.97,
                                             lines_detected=2, lines_read=2)
        pdf = FakePdf([("", 0.98)])
        api, _, _ = _api(document_file("scan.pdf"), pdf, ocr)

        [page] = api.execute(uuid.uuid4()).pages

        assert page.method is ExtractionMethod.OCR
        assert page.text == "Člen 3\nNajemnik plača najemnino."
        assert (page.ocr_confidence, page.ocr_lines_detected, page.ocr_lines_read) == (0.97, 2, 2)
        assert not page.ocr_incomplete

    def test_page_that_lost_ocr_lines_is_flagged_incomplete(self, document_file):
        ocr = MagicMock()
        ocr.recognize.return_value = OcrPage(text="Only one line", mean_confidence=0.9, lines_detected=6,
                                             lines_read=4)
        api, _, _ = _api(document_file("scan.pdf"), FakePdf([("", 1.0)]), ocr)

        document = api.execute(uuid.uuid4())

        assert document.pages[0].ocr_incomplete
        assert document.pages_ocr_incomplete == 1

    def test_scanned_page_without_ocr_is_unreadable_not_guessed(self, document_file):
        api, _, _ = _api(document_file("scan.pdf"), FakePdf([("3", 1.0)]), ocr=None)

        document = api.execute(uuid.uuid4())

        assert document.pages[0].method is ExtractionMethod.NONE
        assert "no OCR model" in document.pages[0].note
        assert document.pages_unreadable == 1

    def test_missing_ocr_models_mark_pages_unreadable_instead_of_failing(self, document_file):
        ocr = MagicMock()
        ocr.recognize.side_effect = OcrModelsMissingError("missing")
        api, _, _ = _api(document_file("scan.pdf"), FakePdf([("", 1.0), (BODY, 0.0)]), ocr)

        document = api.execute(uuid.uuid4())

        assert [p.method for p in document.pages] == [ExtractionMethod.NONE, ExtractionMethod.TEXT_LAYER]
        assert document.pages[0].note == "scanned page; OCR models are not installed"

    def test_blank_scanned_page_is_empty_not_unreadable(self, document_file):
        ocr = MagicMock()
        ocr.recognize.return_value = OcrPage()
        api, _, _ = _api(document_file("scan.pdf"), FakePdf([("", 1.0)]), ocr)

        document = api.execute(uuid.uuid4())

        assert document.pages[0].method is ExtractionMethod.OCR and document.pages[0].text == ""
        assert document.pages_unreadable == 0

    def test_text_layer_is_normalized(self, document_file):
        pdf = FakePdf([("Article\t1\tDeﬁnitions\n\n\n\nThe Premises means the oﬃce space in Ljubljana.", 0.0)])
        api, _, _ = _api(document_file("lease.pdf"), pdf)

        [page] = api.execute(uuid.uuid4()).pages

        assert page.text == "Article 1 Definitions\n\nThe Premises means the office space in Ljubljana."


class TestTables:
    def test_a_page_with_tables_uses_the_layout_text(self, document_file):
        layout = MagicMock()
        layout.extract.return_value = "INVOICE\n\n| Item | Amount |\n| --- | --- |\n| Parking |\tEUR 80.00 |"
        api, _, _ = _api(document_file("invoice.pdf"), FakePdf([(BODY, 0.0)]), layout=layout)

        [page] = api.execute(uuid.uuid4()).pages

        layout.extract.assert_called_once_with(b"page-0")
        assert page.text == "INVOICE\n\n| Item | Amount |\n| --- | --- |\n| Parking | EUR 80.00 |"
        assert page.method is ExtractionMethod.TEXT_LAYER

    def test_a_page_without_tables_keeps_markitdown_text(self, document_file):
        layout = MagicMock()
        layout.extract.return_value = None
        api, _, _ = _api(document_file("lease.pdf"), FakePdf([(BODY, 0.0)]), layout=layout)

        [page] = api.execute(uuid.uuid4()).pages

        assert page.text == BODY

    def test_scanned_pages_are_not_sent_to_the_layout_extractor(self, document_file):
        layout, ocr = MagicMock(), MagicMock()
        ocr.recognize.return_value = OcrPage(text="| A | B |", lines_detected=1, lines_read=1)
        api, _, _ = _api(document_file("scan.pdf"), FakePdf([("", 1.0)]), ocr=ocr, layout=layout)

        api.execute(uuid.uuid4())

        layout.extract.assert_not_called()


class TestOtherInputs:
    def test_docx_is_one_unpaged_block(self, document_file):
        api, _, converter = _api(document_file("memo.docx"))
        converter.convert.return_value = "# Memo\n\nThe hearing moved to Monday."

        document = api.execute(uuid.uuid4())

        assert document.kind is DocumentKind.DOCX
        assert document.pages == [PageContent(page_no=None, text="# Memo\n\nThe hearing moved to Monday.")]
        assert converter.convert.call_args.args[1] == ".docx"

    def test_existing_conversion_is_returned_without_reconverting(self, document_file):
        existing = ConvertedDocument(source_id=uuid.uuid4(), kind=DocumentKind.PDF)
        api, repo, converter = _api(document_file("lease.pdf"), existing=existing)

        assert api.execute(existing.source_id) is existing
        converter.convert.assert_not_called()
        repo.save.assert_not_called()

    def test_unsupported_extension_raises(self, document_file):
        api, _, _ = _api(document_file("budget.xlsx"))

        with pytest.raises(UnsupportedDocumentError):
            api.execute(uuid.uuid4())

    def test_missing_vault_file_raises(self, tmp_path):
        api, _, _ = _api(tmp_path / "gone.pdf")

        with pytest.raises(SourceDocumentMissingError):
            api.execute(uuid.uuid4())

    def test_converter_failure_is_mapped(self, document_file):
        pdf = FakePdf([(BODY, 0.0)])
        api, _, converter = _api(document_file("lease.pdf"), pdf)
        converter.convert.side_effect = RuntimeError("boom")

        with pytest.raises(DocumentConversionError):
            api.execute(uuid.uuid4())

    def test_release_closes_the_ocr_engine(self, document_file):
        ocr = MagicMock()
        api, _, _ = _api(document_file("lease.pdf"), ocr=ocr)

        api.release()

        ocr.close.assert_called_once_with()


@pytest.fixture
def text_file(tmp_path):
    def make(name, data):
        path = tmp_path / name
        path.write_bytes(data)
        return path
    return make


class TestBooksAndTexts:
    @pytest.mark.parametrize("name, kind", [
        ("novel.txt", DocumentKind.TXT), ("notes.md", DocumentKind.MARKDOWN), ("notes.markdown", DocumentKind.MARKDOWN),
    ])
    def test_plain_text_is_decoded_not_sent_to_markitdown(self, text_file, name, kind):
        decoder = MagicMock()
        decoder.decode.return_value = "CHAPTER I\n\nIt was a dark night."
        api, _, converter = _api(text_file(name, b"raw bytes"), text_decoder=decoder)

        document = api.execute(uuid.uuid4())

        decoder.decode.assert_called_once_with(b"raw bytes")
        converter.convert.assert_not_called()
        assert document.kind is kind
        assert document.pages == [PageContent(page_no=None, text="CHAPTER I\n\nIt was a dark night.")]

    def test_plain_text_keeps_every_line_so_cited_line_numbers_hold(self, text_file):
        decoder = MagicMock()
        decoder.decode.return_value = "\n\nTitle\r\n\n\n\nBody\tline  \n"
        api, _, _ = _api(text_file("book.txt", b"x"), text_decoder=decoder)

        [page] = api.execute(uuid.uuid4()).pages

        assert page.text.split("\n") == ["", "", "Title", "", "", "", "Body line", ""]

    @pytest.mark.parametrize("name, kind, extension", [
        ("book.epub", DocumentKind.EPUB, ".epub"),
        ("page.html", DocumentKind.HTML, ".html"),
        ("page.htm", DocumentKind.HTML, ".html"),
    ])
    def test_epub_and_html_go_through_markitdown(self, text_file, name, kind, extension):
        api, _, converter = _api(text_file(name, b"zip"))
        converter.convert.return_value = "# Chapter 1\n\n\n\nCall me Ishmael."

        document = api.execute(uuid.uuid4())

        converter.convert.assert_called_once_with(b"zip", extension)
        assert document.kind is kind
        assert document.pages[0].text == "# Chapter 1\n\nCall me Ishmael."

    def test_rtf_is_read_by_the_rtf_reader(self, text_file):
        rtf_reader = MagicMock()
        rtf_reader.to_text.return_value = "Glava 1\nTekst."
        api, _, converter = _api(text_file("knjiga.rtf", b"{\\rtf1 x}"), rtf_reader=rtf_reader)

        document = api.execute(uuid.uuid4())

        rtf_reader.to_text.assert_called_once_with(b"{\\rtf1 x}")
        converter.convert.assert_not_called()
        assert document.kind is DocumentKind.RTF
        assert document.pages[0].text == "Glava 1\nTekst."

    def test_plain_text_without_a_decoder_is_unsupported(self, text_file):
        api, _, _ = _api(text_file("novel.txt", b"x"))

        with pytest.raises(UnsupportedDocumentError):
            api.execute(uuid.uuid4())

    def test_undecodable_text_keeps_its_type(self, text_file):
        decoder = MagicMock()
        decoder.decode.side_effect = UndecodableTextError("binary")
        api, _, _ = _api(text_file("novel.txt", b"x"), text_decoder=decoder)

        with pytest.raises(UndecodableTextError):
            api.execute(uuid.uuid4())


class TestReadImage:
    def _api(self, ocr):
        return DocumentConversionAPI(MagicMock(), MagicMock(), MagicMock(), MagicMock(), ocr)

    def test_reads_the_image_with_ocr(self, tmp_path):
        path = tmp_path / "sign.png"
        Image.new("L", (300, 200), 255).save(path)
        ocr = MagicMock()
        ocr.recognize.return_value = OcrPage(text="EXIT", lines_detected=1, lines_read=1)

        assert self._api(ocr).read_image(path) == OcrPage(text="EXIT", lines_detected=1, lines_read=1)
        [image] = ocr.recognize.call_args.args
        assert image.mode == "RGB" and image.size == (300, 200)

    def test_a_huge_photo_is_downscaled_on_its_long_side(self, tmp_path, monkeypatch):
        from app.core.config import settings
        monkeypatch.setattr(settings, "IMAGE_OCR_MAX_SIDE", 100)
        path = tmp_path / "big.png"
        Image.new("RGB", (400, 200), "white").save(path)
        ocr = MagicMock()
        ocr.recognize.return_value = OcrPage()

        self._api(ocr).read_image(path)

        assert ocr.recognize.call_args.args[0].size == (100, 50)

    def test_no_ocr_on_this_appliance_is_none(self, tmp_path):
        assert self._api(None).read_image(tmp_path / "anything.png") is None

    def test_missing_ocr_models_are_none_not_a_failure(self, tmp_path):
        path = tmp_path / "sign.png"
        Image.new("RGB", (10, 10)).save(path)
        ocr = MagicMock()
        ocr.recognize.side_effect = OcrModelsMissingError("models/rapidocr is empty")

        assert self._api(ocr).read_image(path) is None

    def test_undecodable_file_is_mapped(self, tmp_path):
        path = tmp_path / "fake.png"
        path.write_bytes(b"nope")

        with pytest.raises(DocumentConversionError):
            self._api(MagicMock()).read_image(path)
