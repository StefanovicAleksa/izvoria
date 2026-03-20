import pytest
import uuid
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.legacy_features.asr_error_correction.utils.chunker import chunk_sentences_by_exact_tokens
from app.legacy_features.asr_error_correction.exceptions.pipeline import AsrCorrectionError


@pytest.fixture
def sample_sentences():
    # FIXED: Replaced segments=[] with words=[]
    return [
        ReconstructedSentence(sentence_id=uuid.uuid4(), text="Sentence one.", speaker_id="S1", start_sec=0.0,
                              end_sec=1.0, words=[]),
        ReconstructedSentence(sentence_id=uuid.uuid4(), text="Sentence two.", speaker_id="S1", start_sec=1.0,
                              end_sec=2.0, words=[]),
        ReconstructedSentence(sentence_id=uuid.uuid4(), text="Sentence three.", speaker_id="S1", start_sec=2.0,
                              end_sec=3.0, words=[])
    ]


def test_chunk_sentences_groups_correctly(sample_sentences):
    # Mock token count: each sentence evaluates to 5 tokens
    def mock_tokenize(text):
        return [1, 2, 3, 4, 5]

    # Max 10 tokens per chunk means max 2 sentences per chunk
    chunks = chunk_sentences_by_exact_tokens(sample_sentences, tokenize_func=mock_tokenize, max_tokens_per_chunk=10)

    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 1
    assert chunks[0][0].text == "Sentence one."
    assert chunks[1][0].text == "Sentence three."


def test_chunk_sentences_tokenizer_failure(sample_sentences):
    def failing_tokenize(text):
        raise Exception("llama.cpp crashed")

    with pytest.raises(AsrCorrectionError, match="Failed to tokenize sentence"):
        chunk_sentences_by_exact_tokens(sample_sentences, tokenize_func=failing_tokenize)


def test_chunk_sentences_empty_input():
    assert chunk_sentences_by_exact_tokens([], tokenize_func=lambda x: []) == []