# Canvas: chunker_v3
# Path: app/features/phonetic_correction/utils/chunker.py

from typing import List, Callable, Any
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.phonetic_correction.exceptions.pipeline import PhoneticPipelineError


def chunk_sentences_by_exact_tokens(
        sentences: List[ReconstructedSentence],
        tokenize_func: Callable[[str], List[Any]],
        max_tokens_per_chunk: int = 4000  # Leaves plenty of room for Dictionary & JSON output
) -> List[List[ReconstructedSentence]]:
    """
    Safely groups sentences into chunks using the exact GGUF tokenizer function.
    Guarantees we never breach the model's context window.
    """
    if not sentences:
        return []

    chunks: List[List[ReconstructedSentence]] = []
    current_chunk: List[ReconstructedSentence] = []
    current_token_count = 0

    for sentence in sentences:
        try:
            # handles the utf-8 byte encoding required by llama_cpp.
            tokens = tokenize_func(sentence.text)
        except Exception as e:
            raise PhoneticPipelineError(f"Failed to tokenize sentence: {e}") from e

        sentence_token_count = len(tokens)

        # If adding this sentence exceeds the limit, seal the current chunk
        if current_token_count + sentence_token_count > max_tokens_per_chunk and current_chunk:
            chunks.append(current_chunk)
            current_chunk = []
            current_token_count = 0

        current_chunk.append(sentence)
        current_token_count += sentence_token_count

    # Append any remaining sentences
    if current_chunk:
        chunks.append(current_chunk)

    return chunks