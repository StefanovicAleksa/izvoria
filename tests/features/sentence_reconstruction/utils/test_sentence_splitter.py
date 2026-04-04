# Canvas: test_sentence_splitter_v1
# Path: tests/features/phonetic_correction/utils/test_sentence_splitter.py

import uuid
from app.features.transcription.domain.entities import TranscriptWord
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.domain_phonetic_correction.utils.sentence_splitter import split_massive_sentences


def test_split_massive_sentences_leaves_normal_sentences_alone():
    sentences = [
        ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=0, end_sec=1, text="Normal.", speaker_id="S0",
                              words=[])
    ]

    result = split_massive_sentences(sentences, max_chars=100)

    assert len(result) == 1
    assert result[0].text == "Normal."


def test_split_massive_sentences_interpolates_timestamps_and_words():
    # 60 characters total, duration 10 seconds, 4 words
    text = "A" * 60
    words = [
        TranscriptWord(start_sec=0, end_sec=2, word="1", probability=0.9),
        TranscriptWord(start_sec=2, end_sec=5, word="2", probability=0.9),
        TranscriptWord(start_sec=5, end_sec=8, word="3", probability=0.9),
        TranscriptWord(start_sec=8, end_sec=10, word="4", probability=0.9)
    ]

    sentence = ReconstructedSentence(
        sentence_id=uuid.uuid4(),
        start_sec=0.0,
        end_sec=10.0,
        text=text,
        speaker_id="S0",
        words=words
    )

    # Max chars 25 means 60 // 25 + 1 = 3 chunks (approx 20 chars each)
    result = split_massive_sentences([sentence], max_chars=25)

    assert len(result) == 3

    # Chunk 1
    assert len(result[0].text) == 20
    assert result[0].start_sec == 0.0
    assert result[0].end_sec == 3.3333333333333335
    assert len(result[0].words) == 1
    assert result[0].words[0].word == "1"

    # Chunk 2
    assert len(result[1].text) == 20
    assert result[1].start_sec == 3.3333333333333335
    assert result[1].end_sec == 6.666666666666667
    assert len(result[1].words) == 1
    assert result[1].words[0].word == "2"

    # Chunk 3 (gets the remainder of the words)
    assert len(result[2].text) == 20
    assert result[2].start_sec == 6.666666666666667
    assert result[2].end_sec == 10.0
    assert len(result[2].words) == 2
    assert result[2].words[0].word == "3"
    assert result[2].words[1].word == "4"