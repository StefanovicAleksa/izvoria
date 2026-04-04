# Canvas: sentence_splitter_v2
# Path: app/features/domain_phonetic_correction/utils/sentence_splitter.py

import uuid
import logging
from typing import List

from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence

logger = logging.getLogger(__name__)

def split_massive_sentences(
        sentences: List[ReconstructedSentence],
        max_chars: int = 1500  # LOWERED from 10000: Protects the LLM from runaway generations
) -> List[ReconstructedSentence]:
    """
    Ruthlessly splits runaway sentences that exceed max_chars to guarantee the LLM never OOMs.
    Interpolates timestamps and divides the word array proportionally to maintain the UI audit trail.
    """
    if not sentences:
        return []

    safe_sentences = []

    for sentence in sentences:
        text_len = len(sentence.text)
        if text_len <= max_chars:
            safe_sentences.append(sentence)
            continue

        logger.warning(f"Massive sentence detected ({text_len} chars). Splitting at {max_chars} to protect LLM.")

        text = sentence.text
        words = sentence.words

        num_chunks = (text_len // max_chars) + 1
        chars_per_chunk = text_len // num_chunks

        words_per_chunk = max(1, len(words) // num_chunks) if words else 0
        duration = sentence.end_sec - sentence.start_sec
        duration_per_chunk = duration / num_chunks

        for i in range(num_chunks):
            start_idx = i * chars_per_chunk
            end_idx = text_len if i == num_chunks - 1 else (i + 1) * chars_per_chunk

            chunk_words = []
            if words:
                if i == num_chunks - 1:
                    chunk_words = words[i * words_per_chunk:]
                else:
                    chunk_words = words[i * words_per_chunk: (i + 1) * words_per_chunk]

            safe_sentences.append(
                ReconstructedSentence(
                    sentence_id=uuid.uuid4(),
                    text=text[start_idx:end_idx].strip(),
                    speaker_id=sentence.speaker_id,
                    start_sec=sentence.start_sec + (i * duration_per_chunk),
                    end_sec=sentence.start_sec + ((i + 1) * duration_per_chunk),
                    words=chunk_words
                )
            )

    return safe_sentences