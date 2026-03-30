# Canvas: contextual_chunker_v1
# Path: app/features/contextual_retrieval/utils/chunker.py

from typing import List
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence


def build_rag_chunks(
        sentences: List[ReconstructedSentence],
        max_words: int = 250,
        overlap_sentences: int = 1
) -> List[List[ReconstructedSentence]]:
    """
    Deterministically groups sentences into RAG-optimized chunks.
    Treats the ReconstructedSentence as the atomic unit to preserve timestamp integrity.

    Args:
        sentences: The list of hallucination-free sentences from Phase 8.
        max_words: The target maximum word count per chunk.
        overlap_sentences: The number of sentences to repeat at the start of the next chunk
                           to maintain narrative continuity.
    """
    if not sentences:
        return []

    chunks: List[List[ReconstructedSentence]] = []
    i = 0
    n = len(sentences)

    while i < n:
        current_chunk: List[ReconstructedSentence] = []
        current_word_count = 0
        idx = i

        while idx < n:
            sentence = sentences[idx]

            # Use the exact acoustic word array length if available, otherwise fallback to split
            sentence_word_count = len(sentence.words) if sentence.words else len(sentence.text.split())

            # If adding this sentence blows past the limit AND we already have at least one sentence
            if current_word_count + sentence_word_count > max_words and current_chunk:
                break

            current_chunk.append(sentence)
            current_word_count += sentence_word_count
            idx += 1

        chunks.append(current_chunk)

        # Reached the end of the document
        if idx == n:
            break

        # Slide the window back by `overlap_sentences`, but ensure we always advance by at least 1
        # to prevent infinite loops if a single massive sentence exceeds max_words
        next_start = idx - overlap_sentences
        i = max(i + 1, next_start)

    return chunks