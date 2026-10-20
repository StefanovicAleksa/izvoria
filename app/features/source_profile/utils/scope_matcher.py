# Path: app/features/source_profile/utils/scope_matcher.py

"""
Which sources a question might name, by the words it shares with their titles and file names. This only shortlists:
it costs no model, so a question that names nothing ("what is conditional probability?") usually stops here, and the
answer model decides among what it finds. Language-neutral on purpose: tokens are Unicode words, numbers compare
as numbers ("module_01" is "module 1"), and a word matches its own abbreviation ("stat" ~ "statistics").
"""

import math
import re
from collections import Counter
from typing import Iterable, List, Set, Tuple

from app.features.source_profile.domain.entities import ProfiledSource

MAX_CANDIDATES = 5
# An abbreviation must be this long to match a longer word, so "in" never matches "introduction"
_MIN_PREFIX = 4
_WORD = re.compile(r"[^\W_]+")


def tokens(text: str) -> Set[str]:
    """The words and numbers of `text`, lowercased; numbers without leading zeros."""
    found: Set[str] = set()
    for word in _WORD.findall(text.lower()):
        if word.isdigit():
            found.add(word.lstrip("0") or "0")
        elif len(word) >= 2:
            found.add(word)
    return found


def _source_tokens(source: ProfiledSource) -> Set[str]:
    names = [source.identity.name, source.identity.file_name]
    if source.profile is not None:
        names.append(source.profile.title)
    return set().union(*(tokens(name) for name in names if name))


def _matches(question_token: str, source_token: str) -> bool:
    if question_token == source_token:
        return True
    if question_token.isdigit() or source_token.isdigit():
        return False
    short, long = sorted((question_token, source_token), key=len)
    return len(short) >= _MIN_PREFIX and long.startswith(short)


def _matched(question: Set[str], source: Iterable[str]) -> Set[str]:
    return {s for s in source if any(_matches(q, s) for q in question)}


def _is_distinctive(token: str, frequency: int, limit: int) -> bool:
    return (token.isdigit() or len(token) >= _MIN_PREFIX) and frequency <= limit


def shortlist(question: str, sources: List[ProfiledSource], limit: int = MAX_CANDIDATES) -> List[ProfiledSource]:
    """
    Up to `limit` sources the question could be naming, best first. A source is listed only if the question shares a
    distinctive word with its names: a number, or a word of four letters or more, that at most half of the sources
    carry (or the only source does). Short words ("of", "the") count toward the score but never list a source.
    Shared words are weighted by rarity, so "lecture 4 of stat 110" puts Lecture 4 above the other Stat 110 lectures.
    """
    question_tokens = tokens(question)
    if not question_tokens or not sources:
        return []

    source_tokens = [_source_tokens(source) for source in sources]
    document_frequency = Counter(token for names in source_tokens for token in names)
    total = len(sources)
    distinctive_limit = max(1, total // 2)

    scored: List[Tuple[float, int]] = []
    for index, names in enumerate(source_tokens):
        matched = _matched(question_tokens, names)
        if not any(_is_distinctive(token, document_frequency[token], distinctive_limit) for token in matched):
            continue
        score = sum(math.log(1 + total / document_frequency[token]) for token in matched)
        scored.append((score, index))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [sources[index] for _, index in scored[:limit]]
