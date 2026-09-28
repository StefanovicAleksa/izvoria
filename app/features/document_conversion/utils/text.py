# Path: app/features/document_conversion/utils/text.py

import re

# Typographic ligatures PDF text layers carry as single characters. "Deﬁnitions" (with U+FB01) matches neither a
# keyword search nor the embedder's vocabulary for "Definitions". Expanding them changes no letter a reader sees.
_LIGATURES = str.maketrans({
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "ﬅ": "st", "ﬆ": "st",
})
# Soft hyphens are invisible line-break hints; non-breaking and other Unicode spaces become plain spaces
_INVISIBLE = str.maketrans({"­": None, " ": " ", " ": " ", " ": " "})
_HORIZONTAL_SPACE = re.compile(r"[ \t\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")


def count_letters(text: str) -> int:
    """Letters in any script. Digits, punctuation and Markdown table pipes don't make a page's text layer real."""
    return sum(1 for character in text if character.isalpha())


def normalize_extracted_text(text: str) -> str:
    """
    Cleans a text layer without changing its wording: expands ligatures, turns tabs and odd spaces into single
    spaces (pdfminer separates positioned words with tabs), trims each line, and keeps at most one blank line
    between paragraphs. Line breaks are kept -- document structuring reads them.
    """
    text = text.translate(_LIGATURES).translate(_INVISIBLE)
    lines = [_HORIZONTAL_SPACE.sub(" ", line).strip() for line in text.replace("\r\n", "\n").split("\n")]
    return _BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


def normalize_text_lines(text: str) -> str:
    """
    `normalize_extracted_text` for a source whose line numbers are cited: the same cleanup inside each line, but
    every line is kept -- no blank lines collapsed, none trimmed from the start or end -- so line N of the result
    is line N of the file a reader opens.
    """
    text = text.translate(_LIGATURES).translate(_INVISIBLE)
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    return "\n".join(_HORIZONTAL_SPACE.sub(" ", line).strip() for line in lines)
