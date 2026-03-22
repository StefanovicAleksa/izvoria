# Canvas: test_chunker_v2
# Path: tests/features/phonetic_correction/utils/test_chunker.py

import uuid
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.phonetic_correction.utils.chunker import chunk_sentences_by_exact_tokens


def dummy_tokenizer(text: str) -> list:
    """Mock tokenizer that treats every word as 1 token."""
    return text.split()


def test_chunk_sentences_respects_token_limits_no_overlap():
    sentences = [
        ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=0, end_sec=1, text="One two three four five",
                              speaker_id="S0", words=[]),
        ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=1, end_sec=2, text="Six seven", speaker_id="S0",
                              words=[]),
        ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=2, end_sec=3, text="Eight nine ten", speaker_id="S0",
                              words=[])
    ]

    # Max 6 tokens per chunk. overlap = 0.
    # S1 (5) fits. S2 (2) pushes it to 7, so it seals chunk 1. S3 (3) goes to chunk 2.
    chunks = chunk_sentences_by_exact_tokens(sentences, dummy_tokenizer, max_tokens_per_chunk=6, overlap_sentences=0)

    assert len(chunks) == 2
    assert len(chunks[0]) == 1
    assert chunks[0][0].text == "One two three four five"
    assert len(chunks[1]) == 2
    assert chunks[1][0].text == "Six seven"
    assert chunks[1][1].text == "Eight nine ten"


def test_chunk_sentences_with_sliding_window_overlap():
    sentences = [
        ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=0, end_sec=1, text="One two three four",
                              speaker_id="S0", words=[]),
        ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=1, end_sec=2, text="Five six seven", speaker_id="S0",
                              words=[]),
        ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=2, end_sec=3, text="Eight nine ten eleven",
                              speaker_id="S0", words=[]),
        ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=3, end_sec=4, text="Twelve", speaker_id="S0",
                              words=[])
    ]

    # Max 10 tokens per chunk. overlap = 1.
    # S1 (4) + S2 (3) = 7 (Fits).
    # S3 (4) pushes it to 11. Chunk 1 is sealed with [S1, S2].
    # Chunk 2 overlaps by taking S2 (3). S2 (3) + S3 (4) + S4 (1) = 8 (Fits).
    chunks = chunk_sentences_by_exact_tokens(sentences, dummy_tokenizer, max_tokens_per_chunk=10, overlap_sentences=1)

    assert len(chunks) == 2

    # Verify Chunk 1
    assert len(chunks[0]) == 2
    assert chunks[0][0].text == "One two three four"
    assert chunks[0][1].text == "Five six seven"

    # Verify Chunk 2 (Must start with the overlapping S2)
    assert len(chunks[1]) == 3
    assert chunks[1][0].text == "Five six seven"
    assert chunks[1][1].text == "Eight nine ten eleven"
    assert chunks[1][2].text == "Twelve"