# Canvas: test_contextual_chunker_v2
# Path: tests/features/contextual_retrieval/utils/test_chunker.py

import uuid
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.transcription.domain.entities import TranscriptWord
from app.features.contextual_retrieval.utils.chunker import build_rag_chunks


def _make_sentence(text: str, num_words: int) -> ReconstructedSentence:
    # Helper to generate dummy sentences with a specific word count
    words = [TranscriptWord(start_sec=0, end_sec=1, word="w", probability=1.0) for _ in range(num_words)]
    return ReconstructedSentence(
        sentence_id=uuid.uuid4(),
        text=text,
        speaker_id="S0",
        start_sec=0.0,
        end_sec=1.0,
        words=words
    )


def test_build_rag_chunks_respects_max_words_and_overlap():
    sentences = [
        _make_sentence("Sentence 1 has ten words.", 10),
        _make_sentence("Sentence 2 has ten words.", 10),
        _make_sentence("Sentence 3 has ten words.", 10),
        _make_sentence("Sentence 4 has ten words.", 10)
    ]

    # Target 25 words max. Should fit Sentences 1 & 2 (20 words).
    # Sentence 3 bumps it to 30, so it breaks.
    # Overlap of 1 means Chunk 2 should start with Sentence 2.
    chunks = build_rag_chunks(sentences, max_words=25, overlap_sentences=1)

    assert len(chunks) == 3

    # Chunk 1: [S1, S2]
    assert len(chunks[0]) == 2
    assert chunks[0][0].text == "Sentence 1 has ten words."
    assert chunks[0][1].text == "Sentence 2 has ten words."

    # Chunk 2: [S2, S3] (Because of 1 sentence overlap)
    assert len(chunks[1]) == 2
    assert chunks[1][0].text == "Sentence 2 has ten words."

    # Chunk 3: [S3, S4]
    assert len(chunks[2]) == 2
    assert chunks[2][1].text == "Sentence 4 has ten words."


def test_build_rag_chunks_massive_sentence_handling():
    # If a single sentence exceeds the max word count, it must be the only item in its chunk.
    sentences = [
        _make_sentence("Short sentence.", 5),
        _make_sentence("Massive sentence.", 100),
        _make_sentence("Another short.", 5)
    ]
    chunks = build_rag_chunks(sentences, max_words=50, overlap_sentences=1)

    # Chunk 1: [Short]
    # Chunk 2: [Massive] (A massive sentence prevents grouping with any other sentence)
    # Chunk 3: [Another]
    assert len(chunks) == 3
    assert len(chunks[0]) == 1 and "Short sentence." in chunks[0][0].text
    assert len(chunks[1]) == 1 and "Massive sentence." in chunks[1][0].text
    assert len(chunks[2]) == 1 and "Another short." in chunks[2][0].text


def test_build_rag_chunks_empty():
    assert build_rag_chunks([]) == []