# Path: scripts/eval/cross_language/language_id.py

"""
Which language a model's answer is written in, from the language packs' own stop words.

Only words distinctive to one pack count ("in" is a Slovenian stop word, "i" a Serbian one, "the" an English one),
so the related Serbian and Slovenian aren't told apart by the words they share. Good enough to check that an
answer is in the language asked for; not a general language identifier.
"""

import re
from typing import Dict, Iterable, Optional

_WORD = re.compile(r"\w+", re.UNICODE)
# Citation tags and quoted excerpts are the material's language, not the answer's
_CITATION = re.compile(r"\[\s*Docs?\b[^\[\]]*\]", re.IGNORECASE)
_QUOTED = re.compile(r"[\"“„«][^\"”“«»]{0,300}[\"”»]")


def distinctive_stopwords(stopwords: Dict[str, Iterable[str]]) -> Dict[str, frozenset]:
    lowered = {code: {w.lower() for w in words} for code, words in stopwords.items()}
    return {code: frozenset(words - set().union(*(o for c, o in lowered.items() if c != code)))
            for code, words in lowered.items()}


def detect_language(text: str, distinctive: Dict[str, frozenset], min_hits: int = 2) -> Optional[str]:
    """The pack whose distinctive stop words the text uses most; None when fewer than `min_hits` are found."""
    body = _QUOTED.sub(" ", _CITATION.sub(" ", text))
    words = [w.lower() for w in _WORD.findall(body)]
    scores = {code: sum(w in vocabulary for w in words) for code, vocabulary in distinctive.items()}
    best = max(scores, key=scores.get, default=None)
    if best is None or scores[best] < min_hits or list(scores.values()).count(scores[best]) > 1:
        return None
    return best
