# Canvas: chunker_v5
# Path: app/features/phonetic_correction/utils/chunker.py

from typing import List, Callable, Any
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.phonetic_correction.exceptions.pipeline import PhoneticPipelineError


def chunk_sentences_by_exact_tokens(
        sentences: List[ReconstructedSentence],
        tokenize_func: Callable[[str], List[Any]],
        max_tokens_per_chunk: int = 4000,
        overlap_sentences: int = 1  # NEW: Context sliding window
) -> List[List[ReconstructedSentence]]:
    """
    Safely groups sentences into chunks using the exact GGUF tokenizer.
    Includes an overlapping window so the LLM retains context across chunk boundaries.
    """
    if not sentences:
        return []

    chunks: List[List[ReconstructedSentence]] = []
    current_chunk: List[ReconstructedSentence] = []
    current_token_count = 0

    for idx, sentence in enumerate(sentences):
        try:
            tokens = tokenize_func(sentence.text)
        except Exception as e:
            raise PhoneticPipelineError(f"Failed to tokenize sentence: {e}") from e

        sentence_token_count = len(tokens)

        if current_token_count + sentence_token_count > max_tokens_per_chunk and current_chunk:
            chunks.append(current_chunk)

            # Sliding Window: Start the new chunk with the last N sentences of the previous chunk
            overlap_slice = current_chunk[-overlap_sentences:] if overlap_sentences > 0 else []
            current_chunk = overlap_slice.copy()

            # Recalculate token count for the carried-over overlap
            current_token_count = sum(len(tokenize_func(s.text)) for s in current_chunk)

        current_chunk.append(sentence)
        current_token_count += sentence_token_count

    if current_chunk:
        chunks.append(current_chunk)

    # Deduplicate sentences at the API level during final assembly
    return chunks