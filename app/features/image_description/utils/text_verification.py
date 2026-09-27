# Path: app/features/image_description/utils/text_verification.py

"""
Checks the text a vision model says it read against what OCR read on the same image.

The two readers fail differently. A vision model reads stylised and handwritten text OCR can't, but it also writes
plausible text that isn't there -- a date, a figure, a name. OCR misreads characters but never invents a line.
Text only the vision model saw is therefore not indexed: an image's citation must not quote words the image doesn't
contain.
"""

import re
import unicodedata
from typing import List, Optional, Sequence, Set, Tuple

from app.core.language.utils.script import normalize_script

_WORD = re.compile(r"\w+", re.UNICODE)
_QUOTED = re.compile(r'"([^"]+)"|“([^”]+)”')
# A number of two or more digits (with its separators): the kind of detail a vision model invents
_MULTI_DIGIT = re.compile(r"\d[\d.,:/-]*\d")

UNCONFIRMED_TEXT = "[text not confirmed]"
UNCONFIRMED_NUMBER = "[number not confirmed]"


def tokens(text: str, script: Optional[str] = None) -> List[str]:
    """Words for comparison: NFKC, case-folded, in the appliance's script, punctuation dropped."""
    return _WORD.findall(normalize_script(unicodedata.normalize("NFKC", text), script).casefold())


class OcrReading:
    """OCR's text as a set of words and as one normalised string, to test items against."""

    def __init__(self, ocr_text: str, script: Optional[str] = None):
        self.script = script
        words = tokens(ocr_text, script)
        self.words: Set[str] = set(words)
        self.joined = f" {' '.join(words)} "

    def confirms(self, item: str, min_coverage: float) -> bool:
        """
        OCR read `item` too: as a whole, word for word; or at least `min_coverage` of its words, including every
        word with a digit in it. A one-word item needs that exact word. OCR splitting a line differently from the
        vision model is the common near-miss the coverage rule allows.
        """
        item_words = tokens(item, self.script)
        if not item_words:
            return False
        if f" {' '.join(item_words)} " in self.joined:
            return True
        if any(word not in self.words for word in item_words if any(ch.isdigit() for ch in word)):
            return False
        if len(item_words) == 1:
            return item_words[0] in self.words
        return sum(word in self.words for word in item_words) / len(item_words) >= min_coverage


def verify_items(items: List[str], reading: OcrReading, min_coverage: float) -> Tuple[List[str], List[str]]:
    """(confirmed, dropped), each in the vision model's order."""
    confirmed, dropped = [], []
    for item in items:
        (confirmed if reading.confirms(item, min_coverage) else dropped).append(item)
    return confirmed, dropped


def strip_unconfirmed(phrase: Optional[str], reading: OcrReading, min_coverage: float) -> Optional[str]:
    """
    A description or mark with every quoted span and multi-digit number OCR did not read replaced by a visible
    marker, so "a sign reading "EXIT 12"" never cites text the image may not hold -- and the reader sees that
    something was there.
    """
    if not phrase:
        return phrase

    def quoted(match: re.Match) -> str:
        return match.group(0) if reading.confirms(match.group(1) or match.group(2), min_coverage) else UNCONFIRMED_TEXT

    without_quotes = _QUOTED.sub(quoted, phrase)
    return _MULTI_DIGIT.sub(
        lambda m: m.group(0) if all(word in reading.words for word in tokens(m.group(0), reading.script))
        else UNCONFIRMED_NUMBER,
        without_quotes)


def select_ocr_lines(
        lines: Sequence[Tuple[str, float]],
        vision_reading: OcrReading,
        min_coverage: float,
        min_confidence: float,
        unreliable_share: float,
        unreliable_min_lines: int,
        unreliable_confidence: float
) -> List[bool]:
    """
    Which OCR lines to keep: the reverse of `verify_items`. A line OCR read with confidence stands on its own; a
    low-confidence one is kept only when the vision model read the same words (`vision_reading` is its text).
    On an image where most lines are low-confidence -- handwriting -- OCR is guessing throughout, so the bar
    for standing alone rises to `unreliable_confidence`.
    """
    low = sum(1 for _, score in lines if score < min_confidence)
    unreliable = len(lines) >= unreliable_min_lines and low >= unreliable_share * len(lines)
    floor = unreliable_confidence if unreliable else min_confidence
    return [score >= floor or vision_reading.confirms(text, min_coverage) for text, score in lines]
