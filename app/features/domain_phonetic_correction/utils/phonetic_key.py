# Path: app/features/domain_phonetic_correction/utils/phonetic_key.py

import re
from typing import List

# Spoken letter names, so a spelled-out acronym ("E C B", "tee el tee ar oh") can be compared with the acronym
LETTER_NAMES = {
    "a": "a", "ay": "a", "b": "b", "bee": "b", "be": "b", "c": "c", "see": "c", "sea": "c", "cee": "c",
    "d": "d", "dee": "d", "e": "e", "ee": "e", "f": "f", "eff": "f", "g": "g", "gee": "g", "h": "h", "aitch": "h",
    "i": "i", "eye": "i", "j": "j", "jay": "j", "k": "k", "kay": "k", "l": "l", "el": "l", "ell": "l",
    "m": "m", "em": "m", "n": "n", "en": "n", "o": "o", "oh": "o", "p": "p", "pee": "p", "pea": "p",
    "q": "q", "cue": "q", "queue": "q", "r": "r", "ar": "r", "are": "r", "s": "s", "ess": "s",
    "t": "t", "tee": "t", "tea": "t", "u": "u", "you": "u", "v": "v", "vee": "v", "w": "w",
    "x": "x", "ex": "x", "y": "y", "why": "y", "z": "z", "zed": "z", "zee": "z",
}

# Ordered spelling-to-sound rewrites; later rules see the output of earlier ones
_REWRITES = [
    (r"sch", "sk"), (r"ph", "f"), (r"gh", "g"), (r"ck", "k"), (r"dg", "j"), (r"wr", "r"), (r"kn", "n"),
    (r"c(?=[eiy])", "s"), (r"c", "k"), (r"q", "k"), (r"x", "ks"), (r"z", "s"), (r"sh", "s"), (r"th", "t"),
    # Voiced and voiceless pairs are the commonest mis-hearings, so each pair shares one symbol
    (r"d", "t"), (r"b", "p"), (r"g", "k"), (r"v", "f"),
]


def letters_only(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def phonetic_key(text: str) -> str:
    """
    A small metaphone-style key: spelling is rewritten towards sound, voiced/voiceless pairs merge, and vowels
    (plus h, w, y) are dropped except at the start. "Bund" and "bunt" share a key; "Draghi" and "draggy" differ only
    in length.
    """
    word = re.sub(r"[^a-z]", "", text.lower())
    if not word:
        return ""
    for pattern, replacement in _REWRITES:
        word = re.sub(pattern, replacement, word)
    head, tail = word[0], re.sub(r"[aeiouhwy]", "", word[1:])
    if head in "eiouy":
        head = "a"
    key = head + tail
    return re.sub(r"(.)\1+", r"\1", key)


def spelled_letters(tokens: List[str]) -> str:
    """Joins tokens that are all spoken letter names ("E", "see", "bee") into letters; empty if any token isn't one."""
    letters = [LETTER_NAMES.get(token.lower().strip(".")) for token in tokens]
    return "" if not letters or any(letter is None for letter in letters) else "".join(letters)
