# Canvas: test_sentence_splitter_v2
# Path: tests/features/domain_phonetic_correction/utils/test_sentence_splitter.py

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

    assert result == sentences


def test_split_massive_sentences_splits_at_whitespace_and_keeps_words_aligned():
    words = [
        TranscriptWord(start_sec=float(i), end_sec=float(i) + 0.5, word=w, probability=0.9)
        for i, w in enumerate(["one", "two", "three", "four", "five"])
    ]
    sentence = ReconstructedSentence(
        sentence_id=uuid.uuid4(), start_sec=0.0, end_sec=4.5, text="one two three four five", speaker_id="S0",
        words=words
    )

    result = split_massive_sentences([sentence], max_chars=10)

    assert [r.text for r in result] == ["one two", "three four", "five"]
    assert [[w.word for w in r.words] for r in result] == [["one", "two"], ["three", "four"], ["five"]]
    assert [(r.start_sec, r.end_sec) for r in result] == [(0.0, 1.5), (2.0, 3.5), (4.0, 4.5)]
    assert all(r.speaker_id == "S0" for r in result)


def test_split_massive_sentences_hard_splits_a_single_oversized_token():
    text = "A" * 60
    words = [
        TranscriptWord(start_sec=0, end_sec=2, word="1", probability=0.9),
        TranscriptWord(start_sec=2, end_sec=5, word="2", probability=0.9),
        TranscriptWord(start_sec=5, end_sec=8, word="3", probability=0.9),
        TranscriptWord(start_sec=8, end_sec=10, word="4", probability=0.9)
    ]
    sentence = ReconstructedSentence(
        sentence_id=uuid.uuid4(), start_sec=0.0, end_sec=10.0, text=text, speaker_id="S0", words=words
    )

    result = split_massive_sentences([sentence], max_chars=25)

    assert [len(r.text) for r in result] == [25, 25, 10]
    assert [[w.word for w in r.words] for r in result] == [["1"], ["2", "3"], ["4"]]
    assert [(r.start_sec, r.end_sec) for r in result] == [(0, 2), (2, 8), (8, 10)]


def test_split_without_words_interpolates_timestamps():
    sentence = ReconstructedSentence(
        sentence_id=uuid.uuid4(), start_sec=10.0, end_sec=20.0, text="aa bb cc dd", speaker_id="S0", words=[]
    )

    result = split_massive_sentences([sentence], max_chars=5)

    assert [r.text for r in result] == ["aa bb", "cc dd"]
    assert [(r.start_sec, r.end_sec) for r in result] == [(10.0, 15.0), (15.0, 20.0)]
