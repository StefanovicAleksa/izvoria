# Path: app/core/language/utils/script.py

"""
Script normalisation. Serbian is written in both Cyrillic and Latin, and Whisper emits either (sometimes both in
one file), so a Serbian appliance normalises everything -- transcript and queries alike -- to Latin.

Only Cyrillic -> Latin is done because only that direction is exact: each Cyrillic letter has one Latin
spelling. The reverse is ambiguous ("injekcija" is n+j, not њ), so it would put words in a witness's mouth.
"""

import re
import unicodedata
from typing import Dict, Optional

_SERBIAN_CYRILLIC_TO_LATIN: Dict[str, str] = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ђ": "đ", "е": "e", "ж": "ž", "з": "z", "и": "i",
    "ј": "j", "к": "k", "л": "l", "љ": "lj", "м": "m", "н": "n", "њ": "nj", "о": "o", "п": "p", "р": "r",
    "с": "s", "т": "t", "ћ": "ć", "у": "u", "ф": "f", "х": "h", "ц": "c", "ч": "č", "џ": "dž", "ш": "š",
}

_CYRILLIC = re.compile(r"[Ѐ-ӿ]")


def cyrillic_to_latin(text: str) -> str:
    """
    Serbian Cyrillic transliterated to Latin; anything else passes through untouched.

    The three digraph letters take their case from the word: "Љубав" -> "Ljubav", but "ЉУБАВ" -> "LJUBAV".
    """
    if not _CYRILLIC.search(text):
        return text

    out = []
    for index, char in enumerate(text):
        lower = char.lower()
        latin = _SERBIAN_CYRILLIC_TO_LATIN.get(lower)
        if latin is None:
            out.append(char)
        elif char == lower:
            out.append(latin)
        elif len(latin) > 1 and _in_uppercase_word(text, index):
            out.append(latin.upper())
        else:
            out.append(latin[0].upper() + latin[1:])
    return "".join(out)


def _in_uppercase_word(text: str, index: int) -> bool:
    """True when a neighbouring letter of the same word is also uppercase: the word is written in capitals."""
    for neighbour in (index + 1, index - 1):
        if 0 <= neighbour < len(text) and text[neighbour].isalpha():
            return text[neighbour].isupper()
    return False


def normalize_script(text: str, script: Optional[str]) -> str:
    """Text in the pack's script. `None` means the language has only one script and nothing is changed."""
    if script == "latin":
        return cyrillic_to_latin(text)
    return text


def fold_to_ascii(text: str) -> str:
    """Diacritics dropped ("Lečenje" -> "Lecenje", "đ" -> "dj"), for filenames and other ASCII-only uses."""
    text = text.replace("đ", "dj").replace("Đ", "Dj")
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(char for char in decomposed if not unicodedata.combining(char))
