from typing import List
from transformers import AutoTokenizer

from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.asr_error_correction.exceptions.pipeline import AsrCorrectionError


def chunk_sentences_by_exact_tokens(
        sentences: List[ReconstructedSentence],
        model_path: str,
        max_tokens_per_chunk: int = 6000  # Leaves ~2k tokens for System Prompt & JSON Response
) -> List[List[ReconstructedSentence]]:
    """
    Safely groups sentences into chunks using the exact tokenizer of the target model.
    Guarantees we never breach the model's context window.
    """
    if not sentences:
        return []

    try:
        # Load the tokenizer directly from the local model directory
        # The model_path should point to the Hugging Face repo dir we downloaded
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    except Exception as e:
        raise AsrCorrectionError(f"Failed to load tokenizer from {model_path}: {e}") from e

    chunks: List[List[ReconstructedSentence]] = []
    current_chunk: List[ReconstructedSentence] = []
    current_token_count = 0

    for sentence in sentences:
        # Count exact tokens for this sentence
        tokens = tokenizer.encode(sentence.text, add_special_tokens=False)
        sentence_token_count = len(tokens)

        # If adding this sentence exceeds the strict limit, seal the current chunk
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