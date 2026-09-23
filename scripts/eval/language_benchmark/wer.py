# Path: scripts/eval/language_benchmark/wer.py

"""
Word error rate with its breakdown, computed here rather than with jiwer: one small function, no new dependency.

Both sides are normalised the same way first -- lowercase, Serbian Cyrillic to Latin, punctuation removed -- so the
number measures words, not formatting. What normalisation deliberately does NOT do is rewrite numbers: a reference
that spells "dvadeset" against a transcript that writes "20" counts as an error, and `digit_tokens` reports how many
such tokens the hypothesis has so the reader can judge how much of the WER that is.
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from app.core.language.utils.script import cyrillic_to_latin

_NON_WORD = re.compile(r"[^\w\s]|_")


def normalize(text: str) -> List[str]:
    text = cyrillic_to_latin(text).lower()
    # Apostrophes inside words ("don't") are joined rather than split, the way both sides would write them
    text = re.sub(r"(?<=\w)['’](?=\w)", "", text)
    return _NON_WORD.sub(" ", text).split()


@dataclass
class Alignment:
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    reference_words: int = 0
    # For every reference word, whether it was matched exactly (used to score reference utterances one by one)
    reference_correct: List[bool] = field(default_factory=list)
    # (reference word or None, hypothesis word or None) for every edit
    edits: List[Tuple[Optional[str], Optional[str]]] = field(default_factory=list)

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.reference_words if self.reference_words else 0.0


def align(reference: Sequence[str], hypothesis: Sequence[str]) -> Alignment:
    """Levenshtein alignment over words, with a backtrace so each error is attributed to a reference word."""
    n, m = len(reference), len(hypothesis)
    # cost[i][j]: edits to turn reference[:i] into hypothesis[:j]
    cost = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost[i][0] = i
    for j in range(1, m + 1):
        cost[0][j] = j
    for i in range(1, n + 1):
        row, previous, ref_word = cost[i], cost[i - 1], reference[i - 1]
        for j in range(1, m + 1):
            row[j] = min(
                previous[j - 1] + (ref_word != hypothesis[j - 1]),
                previous[j] + 1,
                row[j - 1] + 1,
            )

    result = Alignment(reference_words=n, reference_correct=[False] * n)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + (reference[i - 1] != hypothesis[j - 1]):
            if reference[i - 1] == hypothesis[j - 1]:
                result.reference_correct[i - 1] = True
            else:
                result.substitutions += 1
                result.edits.append((reference[i - 1], hypothesis[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            result.deletions += 1
            result.edits.append((reference[i - 1], None))
            i -= 1
        else:
            result.insertions += 1
            result.edits.append((None, hypothesis[j - 1]))
            j -= 1
    result.edits.reverse()
    return result


def utterance_error_rates(alignment: Alignment, utterance_word_counts: Sequence[int]) -> List[float]:
    """
    Share of each reference utterance's words that were not matched exactly. Insertions belong to no reference
    word, so this is a per-utterance miss rate rather than a WER; it is for finding the worst utterances.
    """
    rates, start = [], 0
    for count in utterance_word_counts:
        words = alignment.reference_correct[start:start + count]
        rates.append(1.0 - sum(words) / count if count else 0.0)
        start += count
    return rates


def digit_tokens(words: Sequence[str]) -> int:
    return sum(1 for word in words if any(ch.isdigit() for ch in word))
