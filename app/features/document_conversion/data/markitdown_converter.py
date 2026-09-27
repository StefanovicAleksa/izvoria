# Path: app/features/document_conversion/data/markitdown_converter.py

import io

from markitdown import MarkItDown, StreamInfo

from app.features.document_conversion.domain.interfaces import IMarkdownConverter


class MarkItDownConverter(IMarkdownConverter):
    """
    Microsoft MarkItDown, used on bytes only.

    MarkItDown can reach the network in exactly two ways, both avoided here: `convert()` fetches an http(s) string
    with `requests`, and its audio converter sends audio to Google's speech API. This adapter only calls
    `convert_stream()`, only hands it PDF pages, DOCX, HTML and EPUB files (all converted locally: EPUB is
    a zip of XHTML read with zipfile and defusedxml), and never passes an `llm_client` or a Document
    Intelligence endpoint. Plugins are off, so nothing installed later can register a converter.
    """

    def __init__(self):
        self._markitdown = MarkItDown(enable_plugins=False)

    def convert(self, data: bytes, extension: str) -> str:
        result = self._markitdown.convert_stream(io.BytesIO(data), stream_info=StreamInfo(extension=extension))
        return result.markdown or ""
