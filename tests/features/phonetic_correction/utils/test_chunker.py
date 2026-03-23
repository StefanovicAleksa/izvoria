# Canvas: test_chunker_v8
# Path: tests/features/phonetic_correction/utils/test_chunker.py

import uuid
from app.features.sentence_reconstruction.domain.entities import ReconstructedSentence
from app.features.phonetic_correction.utils.chunker import sliding_window_chunker


def test_sliding_window_chunker_dynamic_chars_and_overlap():
    sentences = [
        ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=1, end_sec=2, text="A short sentence.",
                              speaker_id="S0", words=[]),
        ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=2, end_sec=3, text="Another short sentence.",
                              speaker_id="S0", words=[]),
        ReconstructedSentence(sentence_id=uuid.uuid4(), start_sec=3, end_sec=4, text="A third short sentence.",
                              speaker_id="S0", words=[])
    ]

    chunks = sliding_window_chunker(
        items=sentences,
        text_extractor=lambda s: s.text,
        max_chars=30,
        overlap_items=1
    )

    assert len(chunks[0]) == 1
    assert chunks[0][0].text == "A short sentence."

    assert len(chunks[1]) == 1
    assert chunks[1][0].text == "Another short sentence."

    assert len(chunks[2]) == 1
    assert chunks[2][0].text == "A third short sentence."


def test_sliding_window_chunker_empty_list():
    chunks = sliding_window_chunker([], text_extractor=lambda x: x.text, max_chars=40, overlap_items=1)
    assert len(chunks) == 0