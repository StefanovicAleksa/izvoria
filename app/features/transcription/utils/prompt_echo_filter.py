# Path: app/features/transcription/utils/prompt_echo_filter.py

import re
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from app.features.transcription.domain.entities import TranscriptSegment, TranscriptWord

_NON_WORD = re.compile(r"[^\w]")


def remove_prompt_echoes(
        segments: List[TranscriptSegment],
        injected_texts: Iterable[Optional[str]],
        min_tokens: int
) -> Tuple[List[TranscriptSegment], int]:
    """
    Removes text Whisper copied from what was injected into its decoder prompt (hotwords, initial prompt).
    faster-whisper feeds that text to the model as "previous transcript", and Whisper sometimes repeats it as if it
    had been spoken ("This recording discusses Legal, Medical, including terms such as affidavit, ...").

    A run of at least `min_tokens` consecutive words that appears, in the same order, in an injected text is removed.
    Single glossary terms ("Bank of England") are real speech and stay. A segment keeps its remaining words, with
    text and timestamps rebuilt from them; a segment with nothing left is dropped. A segment without word timestamps
    is dropped only when the echo covers all of its text.

    Returns the cleaned segments and the number of words removed.
    """
    injected = [tokens for tokens in (_tokens(text) for text in injected_texts) if tokens]
    if not injected or min_tokens < 1:
        return list(segments), 0

    cleaned: List[TranscriptSegment] = []
    removed = 0

    for segment in segments:
        if not segment.words:
            tokens = _tokens(segment.text)
            if tokens and len(_echoed_positions(tokens, injected, min_tokens)) == len(tokens):
                removed += len(tokens)
                continue
            cleaned.append(segment)
            continue

        word_tokens = [_token(word.word) for word in segment.words]
        content_indices = [index for index, token in enumerate(word_tokens) if token]
        echoed = {content_indices[position] for position in
                  _echoed_positions([word_tokens[i] for i in content_indices], injected, min_tokens)}
        if not echoed:
            cleaned.append(segment)
            continue

        echoed |= _enclosed_punctuation(word_tokens, echoed)
        remaining = [word for index, word in enumerate(segment.words) if index not in echoed]
        removed += len(segment.words) - len(remaining)

        if not any(_token(word.word) for word in remaining):
            continue

        cleaned.append(segment.model_copy(update={
            "text": _join_words(remaining),
            "words": remaining,
            "start_sec": remaining[0].start_sec,
            "end_sec": remaining[-1].end_sec,
        }))

    return cleaned, removed


def _token(text: str) -> str:
    return _NON_WORD.sub("", text.lower())


def _tokens(text: Optional[str]) -> List[str]:
    return [token for token in (_token(part) for part in (text or "").split()) if token]


def _echoed_positions(tokens: Sequence[str], injected: Sequence[Sequence[str]], min_tokens: int) -> Set[int]:
    """Positions of `tokens` covered by runs of at least `min_tokens` that appear contiguously in an injected text."""
    echoed: Set[int] = set()
    position = 0
    while position < len(tokens):
        length = _longest_injected_run(tokens, position, injected)
        if length >= min_tokens:
            echoed.update(range(position, position + length))
            position += length
        else:
            position += 1
    return echoed


def _longest_injected_run(tokens: Sequence[str], position: int, injected: Sequence[Sequence[str]]) -> int:
    best = 0
    for sequence in injected:
        for start, token in enumerate(sequence):
            if token != tokens[position]:
                continue
            length = 0
            while (position + length < len(tokens) and start + length < len(sequence)
                   and tokens[position + length] == sequence[start + length]):
                length += 1
            best = max(best, length)
    return best


def _enclosed_punctuation(word_tokens: Sequence[str], echoed: Set[int]) -> Set[int]:
    """Punctuation-only words sitting inside a removed run go with it."""
    enclosed: Set[int] = set()
    for index, token in enumerate(word_tokens):
        if token:
            continue
        previous = next((i for i in range(index - 1, -1, -1) if word_tokens[i]), None)
        following = next((i for i in range(index + 1, len(word_tokens)) if word_tokens[i]), None)
        if previous in echoed and (following is None or following in echoed):
            enclosed.add(index)
    return enclosed


def _join_words(words: List[TranscriptWord]) -> str:
    """Whisper words carry their own leading space; continuations such as ".99" have none."""
    if sum(1 for word in words if word.word[:1].isspace()) * 2 >= len(words):
        return "".join(word.word for word in words).strip()
    return " ".join(word.word.strip() for word in words if word.word.strip())
