# Path: app/features/domain_phonetic_correction/utils/phonetic_key.py

import re
from typing import List

from app.core.language.domain.entities import PhoneticRules
from app.core.language.service.language_api import get_language_pack

# Spoken letter names (so a spelled-out acronym, "E C B" / "tee el tee ar oh", can be compared with the acronym),
# the alphabet and the spelling-to-sound rewrites all come from the appliance's language pack ([phonetic]).


def _rules() -> PhoneticRules:
    return get_language_pack().phonetic


def letters_only(text: str) -> str:
    return re.sub(f"[^{_rules().alphabet}0-9]", "", text.lower())


def phonetic_key(text: str) -> str:
    """
    A small metaphone-style key: spelling is rewritten towards sound, voiced/voiceless pairs merge, and vowels
    (plus, in English, h, w and y) are dropped except at the start. "Bund" and "bunt" share a key; "Draghi" and
    "draggy" differ only in length.
    """
    rules = _rules()
    word = text.lower()
    for letter, folded in rules.fold.items():
        word = word.replace(letter, folded)
    word = re.sub(f"[^{rules.alphabet}]", "", word)
    if not word:
        return ""
    for pattern, replacement in rules.rewrites:
        word = re.sub(pattern, replacement, word)
    head, tail = word[0], re.sub(f"[{rules.silent_letters}]", "", word[1:])
    if head in rules.head_vowels:
        head = "a"
    key = head + tail
    return re.sub(r"(.)\1+", r"\1", key)


def spelled_letters(tokens: List[str]) -> str:
    """Joins tokens that are all spoken letter names ("E", "see", "bee") into letters; empty if any token isn't one."""
    letter_names = _rules().letter_names
    letters = [letter_names.get(token.lower().strip(".")) for token in tokens]
    return "" if not letters or any(letter is None for letter in letters) else "".join(letters)
