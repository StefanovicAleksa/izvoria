# Path: app/features/domain_phonetic_correction/utils/term_candidate_finder.py

import re
from difflib import SequenceMatcher
from functools import lru_cache
from typing import List, NamedTuple, Optional, Sequence

from app.core.config import settings
from app.core.language.service.language_api import get_language_pack
from app.features.domain_phonetic_correction.domain.entities import PhoneticCandidate
from app.features.domain_phonetic_correction.utils.phonetic_key import letters_only, phonetic_key, spelled_letters
from app.features.transcription.domain.entities import TranscriptWord

MAX_SPAN_TOKENS = 5
# Every token of a span must supply at least this share of its letters to the match, so a span can't absorb
# neighbouring words ("the bund" -> "Bunds") while real split mis-hearings ("a cute angina") still match
MIN_TOKEN_CONTRIBUTION = 0.6
# Letters must stay this similar however alike the sound keys are: keys alone merge "highs" with "hawkish"
MIN_LETTER_SIMILARITY = 0.6


class _Token(NamedTuple):
    text: str
    start: int
    end: int


class _Term(NamedTuple):
    text: str
    letters: str
    key: str
    parts: List[str]
    acronym: bool


@lru_cache(maxsize=16)
def _token_pattern(alphabet: str) -> re.Pattern:
    """A word, number or contraction; inner dots/hyphens/apostrophes stay inside the token ("13.50.45", "pre-planning")."""
    return re.compile(f"[{alphabet}0-9](?:[{alphabet}0-9'’.\\-]*[{alphabet}0-9])?", re.IGNORECASE)


def _plural_suffix() -> str:
    return get_language_pack().phonetic.plural_suffix


def _singular(word: str) -> str:
    suffix = _plural_suffix()
    return word[:-len(suffix)] if suffix and len(word) > 3 and word.endswith(suffix) else word


def _word_parts(text: str) -> List[str]:
    return [p for p in (letters_only(part) for part in re.split(r"[\s\-]+", text)) if p]


def _same_words(heard_parts: List[str], term_parts: List[str]) -> bool:
    """Same words up to case, hyphens, inner punctuation and a trailing plural 's': already the term, not a mis-hearing."""
    return (len(heard_parts) == len(term_parts)
            and [_singular(p) for p in heard_parts] == [_singular(p) for p in term_parts])


def _prepare_term(term: str) -> Optional[_Term]:
    letters = letters_only(term)
    if not letters:
        return None
    acronym = 2 <= len(letters) <= 6 and term.strip().isupper() and term.strip().isalpha()
    return _Term(term.strip(), letters, phonetic_key(letters), _word_parts(term), acronym)


def _tokenize(text: str) -> List[_Token]:
    pattern = _token_pattern(get_language_pack().phonetic.alphabet)
    return [_Token(m.group(0), m.start(), m.end()) for m in pattern.finditer(text)]


def _token_probabilities(tokens: List[_Token], words: Sequence[TranscriptWord]) -> List[Optional[float]]:
    """
    Maps each text token to the lowest probability of the Whisper words its letters came from. The mapping needs the
    letters of the text and of the words to be identical; otherwise (text edited upstream) probabilities are unknown.
    """
    if not words:
        return [None] * len(tokens)
    owners: List[int] = []
    for index, word in enumerate(words):
        owners.extend([index] * len(letters_only(word.word)))
    token_letters = [letters_only(token.text) for token in tokens]
    if "".join(token_letters) != "".join(letters_only(word.word) for word in words):
        return [None] * len(tokens)

    probabilities: List[Optional[float]] = []
    position = 0
    for letters in token_letters:
        word_indexes = set(owners[position:position + len(letters)])
        position += len(letters)
        probabilities.append(min(words[i].probability for i in word_indexes) if word_indexes else None)
    return probabilities


def _match_number(term: str, heard_tokens: List[_Token], acronym: bool) -> Optional[str]:
    """
    Keeps the singular/plural the speaker used: "burn" -> "Bund" and "burns" -> "Bunds" for the term "Bunds".
    A plural heard for a singular term, or an -ed/-ing form heard for a term without it, returns None: the replacement
    would have to guess a spelling nobody confirmed ("hawkishs") or change the grammar ("burned" -> "Bund").
    """
    if acronym:
        return term
    rules = get_language_pack().phonetic
    heard_last = letters_only(heard_tokens[-1].text)
    term_last = letters_only(term.split()[-1])
    # A heard verb form never becomes a noun term ("burned" -> "Bund"): that would change the grammar, not the spelling
    if any(heard_last.endswith(suffix) and not term_last.endswith(suffix) for suffix in rules.verb_suffixes):
        return None
    suffix = rules.plural_suffix
    if not suffix:
        # An inflected language: the term's own form is the only spelling anyone confirmed
        return term
    heard_plural = heard_last.endswith(suffix)
    term_plural = len(term) > 3 and term.endswith(suffix) and not term.endswith(suffix * 2)
    if term_plural and not heard_plural:
        return term[:-len(suffix)]
    if heard_plural and not term_plural:
        return None
    return term


@lru_cache(maxsize=200_000)
def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _key_similarity(heard_letters: str, term_key: str) -> float:
    """Sound-key similarity; keys shorter than 3 symbols are too coarse to compare ("bm" vs "pnt")."""
    return _cached_key_similarity(heard_letters, term_key, settings.APPLIANCE_LANGUAGE)


@lru_cache(maxsize=200_000)
def _cached_key_similarity(heard_letters: str, term_key: str, language: str) -> float:
    # `language` is only a cache key: the sound key comes from the active pack, which a runtime switch changes
    heard_key = phonetic_key(heard_letters)
    return _similarity(heard_key, term_key) if min(len(heard_key), len(term_key)) >= 3 else 0.0


def _spans_cover_each_other(heard_tokens: List[_Token], term_parts: List[str], term_letters: str) -> bool:
    """
    Every heard token must supply, and every term word must receive, at least MIN_TOKEN_CONTRIBUTION of its letters in
    the letter alignment. The first stops a span absorbing a neighbouring word ("the bund" -> "Bunds"); the second stops
    a term replacing only part of what was heard for it ("depot sit rates" -> "Deposit Rate rates").
    """
    token_letters = [letters_only(token.text) for token in heard_tokens]
    token_owner = [i for i, letters in enumerate(token_letters) for _ in letters]
    term_owner = [i for i, part in enumerate(term_parts) for _ in part]
    token_matched = [0] * len(token_letters)
    term_matched = [0] * len(term_parts)
    for block in SequenceMatcher(None, "".join(token_letters), term_letters, autojunk=False).get_matching_blocks():
        for offset in range(block.size):
            token_matched[token_owner[block.a + offset]] += 1
            if block.b + offset < len(term_owner):
                term_matched[term_owner[block.b + offset]] += 1
    tokens_ok = all(letters and token_matched[i] >= MIN_TOKEN_CONTRIBUTION * len(letters)
                    for i, letters in enumerate(token_letters))
    terms_ok = all(term_matched[i] >= MIN_TOKEN_CONTRIBUTION * len(part) for i, part in enumerate(term_parts))
    return tokens_ok and terms_ok


def find_term_candidates(
        text: str,
        words: Sequence[TranscriptWord],
        terms: Sequence[str],
        min_similarity: float,
        length_ratio_min: float,
        length_ratio_max: float,
        max_word_probability: float
) -> List[PhoneticCandidate]:
    """
    Finds non-overlapping spans of `text` that sound like a dictionary term but aren't already that term.

    Never a candidate: a span containing a digit ("73s", "13.50.45"), a span that already is the term up to case,
    hyphens and plural ("bund", "E.C.B.", "deposit rates"), a span containing a token that alone is the term
    ("the bund"), a span whose words don't each supply letters to the match, and a span whose Whisper words were all
    heard with probability of at least `max_word_probability` (unknown probabilities don't block).
    """
    prepared = [t for t in (_prepare_term(term) for term in dict.fromkeys(terms)) if t]
    tokens = _tokenize(text)
    if not prepared or not tokens:
        return []

    probabilities = _token_probabilities(tokens, words)
    # A correctly written single-word term is never part of another term's mis-hearing ("ECB meeting")
    single_word_terms = {_singular(t.letters) for t in prepared if len(t.parts) == 1}
    found: List[PhoneticCandidate] = []

    for first in range(len(tokens)):
        for last in range(first, min(first + MAX_SPAN_TOKENS, len(tokens))):
            span_tokens = tokens[first:last + 1]
            heard = text[span_tokens[0].start:span_tokens[-1].end]
            if any(ch.isdigit() for ch in heard):
                break

            known = [p for p in probabilities[first:last + 1] if p is not None]
            min_probability = min(known) if known else None
            if min_probability is not None and min_probability >= max_word_probability:
                continue

            heard_letters = letters_only(heard)
            heard_parts = _word_parts(heard)
            token_letter_sets = [_singular(letters_only(t.text)) for t in span_tokens]
            if any(letters in single_word_terms for letters in token_letter_sets):
                continue
            span_count = len(span_tokens)

            for term in prepared:
                word_count = len(term.parts)
                max_tokens = min(len(term.letters), MAX_SPAN_TOKENS) if term.acronym else word_count + 2
                # Fewer tokens than the term has words would match part of a correctly written term ("Philip" of "Philip Lane")
                if not word_count <= span_count <= max_tokens:
                    continue
                if _same_words(heard_parts, term.parts) or _singular(term.letters) in token_letter_sets:
                    continue

                spelled = spelled_letters([t.text for t in span_tokens]) if term.acronym and span_count > 1 else ""
                if spelled:
                    score = _similarity(spelled, term.letters)
                elif term.acronym and not (span_count == 1 and span_tokens[0].text.isupper()):
                    # An acronym is heard as letter names or as another acronym ("ECP"), never as an ordinary word ("from")
                    continue
                else:
                    ratio = len(heard_letters) / len(term.letters)
                    if not length_ratio_min <= ratio <= length_ratio_max:
                        continue
                    # A singular heard word is compared with the singular of a plural term ("burn" with "Bund")
                    term_letters, term_key = term.letters, term.key
                    suffix = _plural_suffix()
                    if suffix and _match_number(term.text, span_tokens, term.acronym) == term.text[:-len(suffix)]:
                        term_letters = term.letters[:-len(suffix)]
                        term_key = phonetic_key(term_letters)
                    letter_similarity = _similarity(heard_letters, term_letters)
                    if letter_similarity < MIN_LETTER_SIMILARITY:
                        continue
                    score = (letter_similarity + _key_similarity(heard_letters, term_key)) / 2
                    # A plural term compared in its singular form loses the final "s" of its last word
                    term_parts = (term.parts if term_letters == term.letters
                                  else term.parts[:-1] + [term.parts[-1][:-len(suffix)]])
                    if score >= min_similarity and not _spans_cover_each_other(span_tokens, term_parts, term_letters):
                        continue

                if score < min_similarity:
                    continue

                replacement = _match_number(term.text, span_tokens, term.acronym)
                if replacement is None:
                    continue
                if span_tokens[0].start == 0 and replacement[:1].islower():
                    replacement = replacement[0].upper() + replacement[1:]
                found.append(PhoneticCandidate(
                    start=span_tokens[0].start, end=span_tokens[-1].end, heard=heard, term=term.text,
                    replacement=replacement, score=round(score, 3), min_probability=min_probability
                ))

    # Best match first; among equals the shortest span, so no neighbouring word is swallowed
    found.sort(key=lambda c: (-c.score, c.end - c.start, c.start))
    selected: List[PhoneticCandidate] = []
    for candidate in found:
        if all(candidate.end <= other.start or candidate.start >= other.end for other in selected):
            selected.append(candidate)
    return sorted(selected, key=lambda c: c.start)
