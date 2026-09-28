# Path: tests/features/document_conversion/data/test_plain_text_decoder.py

import codecs
from unittest.mock import patch

import pytest

from app.features.document_conversion.data.plain_text_decoder import CharsetTextDecoder
from app.features.document_conversion.exceptions import UndecodableTextError

SLOVENIAN = "Najemnik plača za čas najema, ki ga določa ta pogodba o najemu poslovnega prostora.\n"
SERBIAN = "Закупац плаћа закупнину за време закупа које одређује овај уговор о закупу пословног простора.\n"


class TestCharsetTextDecoder:
    def test_utf8(self):
        assert CharsetTextDecoder().decode(SLOVENIAN.encode("utf-8")) == SLOVENIAN

    def test_utf8_bom_is_dropped(self):
        assert CharsetTextDecoder().decode(codecs.BOM_UTF8 + b"Chapter 1\n") == "Chapter 1\n"

    def test_cp1250_slovenian(self):
        assert CharsetTextDecoder().decode((SLOVENIAN * 5).encode("cp1250")) == SLOVENIAN * 5

    def test_cp1251_serbian_cyrillic(self):
        assert CharsetTextDecoder().decode((SERBIAN * 5).encode("cp1251")) == SERBIAN * 5

    def test_line_breaks_are_untouched(self):
        assert CharsetTextDecoder().decode(b"a\r\n\r\n\r\nb") == "a\r\n\r\n\r\nb"

    def test_no_recognisable_encoding_raises(self):
        with patch("app.features.document_conversion.data.plain_text_decoder.from_bytes") as from_bytes:
            from_bytes.return_value.best.return_value = None

            with pytest.raises(UndecodableTextError):
                CharsetTextDecoder().decode(b"\xff\xfe\x00\x81")
