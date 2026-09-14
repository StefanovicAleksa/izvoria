# Canvas: sentence_splitter_v3
# Path: app/features/domain_phonetic_correction/utils/sentence_splitter.py

import uuid
import logging
from typing import List

from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence

logger = logging.getLogger(__name__)


def _pack_tokens(tokens: List[str], max_chars: int) -> List[List[str]]:
    """Greedily packs whitespace tokens into groups whose joined text fits `max_chars`."""
    groups: List[List[str]] = []
    current: List[str] = []
    current_len = 0

    for token in tokens:
        # A single token longer than the limit can only be split by characters
        pieces = [token[i:i + max_chars] for i in range(0, len(token), max_chars)] if len(token) > max_chars else [token]
        for piece in pieces:
            added = len(piece) + (1 if current else 0)
            if current and current_len + added > max_chars:
                groups.append(current)
                current, current_len = [], 0
                added = len(piece)
            current.append(piece)
            current_len += added

    if current:
        groups.append(current)
    return groups


def split_massive_sentences(
        sentences: List[ReconstructedSentence],
        max_chars: int = 1500  # LOWERED from 10000: Protects the LLM from runaway generations
) -> List[ReconstructedSentence]:
    """
    Splits runaway sentences that exceed max_chars to guarantee the LLM never OOMs.

    Text is split at whitespace (never mid-word, unless a single token exceeds the limit), and the
    acoustic word array is divided in proportion to the text tokens, so each piece's words and
    timestamps still correspond to its text for the UI audit trail.
    """
    if not sentences:
        return []

    safe_sentences = []

    for sentence in sentences:
        if len(sentence.text) <= max_chars:
            safe_sentences.append(sentence)
            continue

        logger.warning(f"Massive sentence detected ({len(sentence.text)} chars). Splitting at {max_chars} to protect LLM.")

        groups = _pack_tokens(sentence.text.split(), max_chars)
        total_tokens = sum(len(g) for g in groups)
        words = sentence.words
        duration = sentence.end_sec - sentence.start_sec

        consumed = 0
        for group_idx, group in enumerate(groups):
            is_last = group_idx == len(groups) - 1
            start_frac = consumed / total_tokens
            consumed += len(group)
            end_frac = 1.0 if is_last else consumed / total_tokens

            chunk_words = []
            if words:
                word_start = round(start_frac * len(words))
                word_end = len(words) if is_last else round(end_frac * len(words))
                chunk_words = words[word_start:word_end]

            if chunk_words:
                start_sec, end_sec = chunk_words[0].start_sec, chunk_words[-1].end_sec
            else:
                start_sec = sentence.start_sec + start_frac * duration
                end_sec = sentence.start_sec + end_frac * duration

            safe_sentences.append(
                ReconstructedSentence(
                    sentence_id=uuid.uuid4(),
                    text=" ".join(group),
                    speaker_id=sentence.speaker_id,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    words=chunk_words
                )
            )

    return safe_sentences
