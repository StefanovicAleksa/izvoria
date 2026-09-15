# Path: app/features/domain_phonetic_correction/utils/span_replacer.py

import logging
import re
from typing import Sequence

from app.features.domain_phonetic_correction.domain.entities import PhoneticCandidate

logger = logging.getLogger(__name__)


def replace_spans(text: str, candidates: Sequence[PhoneticCandidate]) -> str:
    """
    Writes each confirmed candidate's replacement over its exact character span. Everything outside the spans is
    copied unchanged, so punctuation, articles and neighbouring words can't be touched.

    A candidate whose span no longer holds its heard text, or that overlaps an earlier one, is skipped. If the result
    would change any digit, the text is returned unchanged.
    """
    pieces = []
    cursor = 0
    for candidate in sorted(candidates, key=lambda c: c.start):
        if candidate.start < cursor or text[candidate.start:candidate.end] != candidate.heard:
            logger.warning(f"Skipping phonetic replacement '{candidate.heard}' -> '{candidate.replacement}': span moved.")
            continue
        pieces.append(text[cursor:candidate.start])
        pieces.append(candidate.replacement)
        cursor = candidate.end
    pieces.append(text[cursor:])
    new_text = "".join(pieces)

    if re.findall(r"\d", new_text) != re.findall(r"\d", text):
        logger.warning("Rejected phonetic replacements for a sentence: they would change its digits.")
        return text
    return new_text
