# Path: tests/features/document_conversion/data/test_rtf_reader.py

from app.features.document_conversion.data.rtf_reader import StripRtfReader


class TestStripRtfReader:
    def test_reads_paragraphs_as_lines(self):
        rtf = rb"{\rtf1\ansi {\b Chapter 1}\par It was night.\par}"

        text = StripRtfReader().to_text(rtf)

        assert [line.strip() for line in text.splitlines() if line.strip()] == ["Chapter 1", "It was night."]

    def test_honours_the_files_code_page(self):
        # \'e8 is "č" in cp1250, not "è" as the cp1252 default would read it
        rtf = rb"{\rtf1\ansi\ansicpg1250 Najemnik pla\'e8a.\par}"

        assert "Najemnik plača." in StripRtfReader().to_text(rtf)
