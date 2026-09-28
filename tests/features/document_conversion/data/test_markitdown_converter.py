# Path: tests/features/document_conversion/data/test_markitdown_converter.py

from unittest.mock import patch

from app.features.document_conversion.data.markitdown_converter import MarkItDownConverter


class TestMarkItDownConverter:
    @patch("app.features.document_conversion.data.markitdown_converter.MarkItDown")
    def test_is_built_without_plugins_llm_or_document_intelligence(self, markitdown_cls):
        MarkItDownConverter()

        markitdown_cls.assert_called_once_with(enable_plugins=False)

    @patch("app.features.document_conversion.data.markitdown_converter.MarkItDown")
    def test_converts_bytes_through_convert_stream_only(self, markitdown_cls):
        instance = markitdown_cls.return_value
        instance.convert_stream.return_value.markdown = "# Page"

        text = MarkItDownConverter().convert(b"%PDF-1.7", ".pdf")

        assert text == "# Page"
        stream = instance.convert_stream.call_args.args[0]
        assert stream.read() == b"%PDF-1.7"
        assert instance.convert_stream.call_args.kwargs["stream_info"].extension == ".pdf"
        # convert() would fetch a URL string with requests; convert_local() reads arbitrary paths
        instance.convert.assert_not_called()
        instance.convert_local.assert_not_called()
        instance.convert_uri.assert_not_called()

    @patch("app.features.document_conversion.data.markitdown_converter.MarkItDown")
    def test_empty_conversion_is_an_empty_string(self, markitdown_cls):
        markitdown_cls.return_value.convert_stream.return_value.markdown = None

        assert MarkItDownConverter().convert(b"x", ".pdf") == ""
