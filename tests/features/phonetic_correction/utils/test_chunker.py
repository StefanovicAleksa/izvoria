# Canvas: test_chunker_v3
# Path: tests/features/phonetic_correction/utils/test_chunker.py

import uuid
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.phonetic_correction.utils.chunker import sliding_window_chunker


def test_sliding_window_chunker_basic():
    # Create 10 dummy sentences
    sentences = [
        ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=i, end_sec=i + 1, text=f"Sentence {i}",
                              speaker_id="S0", words=[])
        for i in range(10)
    ]

    # Chunk size 4, overlap 2 -> Step size is 2
    chunks = sliding_window_chunker(sentences, chunk_size=4, overlap=2)

    assert len(chunks) == 4

    # Chunk 1: Sentences 0, 1, 2, 3
    assert len(chunks[0]) == 4
    assert chunks[0][0].text == "Sentence 0"
    assert chunks[0][-1].text == "Sentence 3"

    # Chunk 2: Sentences 2, 3, 4, 5 (Overlap of 2, 3)
    assert len(chunks[1]) == 4
    assert chunks[1][0].text == "Sentence 2"
    assert chunks[1][-1].text == "Sentence 5"


def test_sliding_window_chunker_empty_list():
    chunks = sliding_window_chunker([], chunk_size=40, overlap=5)
    assert len(chunks) == 0