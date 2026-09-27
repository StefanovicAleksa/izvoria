# Path: app/features/document_conversion/data/rtf_reader.py

from striprtf.striprtf import rtf_to_text

from app.features.document_conversion.domain.interfaces import IRtfReader


class StripRtfReader(IRtfReader):
    """
    striprtf: pure Python, no dependencies. RTF is 7-bit ASCII with escaped characters, and the file's own
    `\\ansicpg` code page decides what those escapes mean, so the bytes are read as Latin-1 (lossless) and the
    code page is left to striprtf. A character it cannot decode shows as U+FFFD rather than vanishing.
    """

    def to_text(self, data: bytes) -> str:
        return rtf_to_text(data.decode("latin-1"), errors="replace")
