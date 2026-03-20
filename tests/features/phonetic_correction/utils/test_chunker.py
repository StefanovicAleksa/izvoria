import uuid
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.phonetic_correction.utils.chunker import chunk_sentences_by_exact_tokens


def dummy_tokenizer(text: str) -> list:
    """Mock tokenizer that treats every word as 1 token."""
    return text.split()


def test_chunk_sentences_respects_token_limits():
    # FIXED: Replaced segments=[] with words=[]
    sentences = [
        ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=0, end_sec=1, text="One two three four five",
                              speaker_id="S0", words=[]),
        ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=1, end_sec=2, text="Six seven", speaker_id="S0",
                              words=[]),
        ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=2, end_sec=3, text="Eight nine ten", speaker_id="S0",
                              words=[])
    ]

    # Max 6 tokens per chunk.
    # S1 (5) fits. S2 (2) pushes it to 7, so it goes to chunk 2. S3 (3) goes to chunk 2.
    chunks = chunk_sentences_by_exact_tokens(sentences, dummy_tokenizer, max_tokens_per_chunk=6)

    assert len(chunks) == 2
    assert len(chunks[0]) == 1
    assert chunks[0][0].text == "One two three four five"
    assert len(chunks[1]) == 2