# Path: app/features/document_conversion/data/plain_text_decoder.py

import codecs

from charset_normalizer import from_bytes

from app.features.document_conversion.domain.interfaces import ITextDecoder
from app.features.document_conversion.exceptions import UndecodableTextError


class CharsetTextDecoder(ITextDecoder):
    """
    UTF-8 when the bytes are UTF-8 (most files, and the only answer for a BOM), otherwise whatever
    charset-normalizer finds most likely: Serbian and Slovenian texts still arrive as cp1250 and cp1251.
    Decoding never touches line breaks, so the line numbers a citation shows are the file's own.
    """

    def decode(self, data: bytes) -> str:
        if data.startswith(codecs.BOM_UTF8):
            return data[len(codecs.BOM_UTF8):].decode("utf-8")
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            pass
        best = from_bytes(data).best()
        if best is None:
            raise UndecodableTextError("The file is in no recognisable text encoding.")
        return str(best)
