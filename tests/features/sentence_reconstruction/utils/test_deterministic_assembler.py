# Canvas: test_deterministic_assembler_v2
# Path: tests/features/sentence_reconstruction/utils/test_deterministic_assembler.py

from app.features.transcription.domain.entities import TranscriptWord
from app.features.sentence_reconstruction.domain.entities import PunctuatedWord
from app.features.sentence_reconstruction.utils.deterministic_assembler import assemble_sentences


def test_assemble_sentences_clean_split():
    """Proves it cleanly splits two sentences when a terminal tag is hit."""

    # Input: 4 words, middle one has a period, last one has a period.
    tagged_words = [
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=0.0, end_sec=0.5, word="this"), punctuation_tag="NONE"),
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=0.5, end_sec=1.0, word="one"), punctuation_tag="."),
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=1.0, end_sec=1.5, word="is"), punctuation_tag="NONE"),
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=1.5, end_sec=2.0, word="two"), punctuation_tag=".")
    ]

    sentences = assemble_sentences("Speaker_1", tagged_words)

    assert len(sentences) == 2
    assert sentences[0].text == "This one."
    assert sentences[0].speaker_id == "Speaker_1"
    assert sentences[0].start_sec == 0.0
    assert sentences[0].end_sec == 1.0

    assert sentences[1].text == "Is two."
    assert sentences[1].start_sec == 1.0
    assert sentences[1].end_sec == 2.0


def test_assemble_sentences_sanitizes_whisper_formatting():
    """Proves it strips leading/trailing spaces and duplicate punctuation from acoustic words."""

    # Whisper gives us leading spaces and duplicate question marks natively
    tagged_words = [
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=0.0, end_sec=0.5, word=" wait"), punctuation_tag=","),
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=0.5, end_sec=1.0, word=" what?"), punctuation_tag="?")
    ]

    sentences = assemble_sentences("Speaker_0", tagged_words)

    assert len(sentences) == 1
    # Should be perfectly capitalized, single spaced, and single punctuated
    assert sentences[0].text == "Wait, what?"


def test_assemble_sentences_handles_orphaned_fragments():
    """Proves it safely wraps up trailing words if the audio cuts off before a period."""

    tagged_words = [
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=0.0, end_sec=0.5, word="unfinished"), punctuation_tag="NONE"),
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=0.5, end_sec=1.0, word="thought"), punctuation_tag="NONE")
    ]

    sentences = assemble_sentences("Speaker_0", tagged_words)

    assert len(sentences) == 1
    # It must inject a period to enforce grammatical correctness for downstream RAG
    assert sentences[0].text == "Unfinished thought."
    assert sentences[0].start_sec == 0.0
    assert sentences[0].end_sec == 1.0


def test_assemble_sentences_keeps_acronym_and_camel_case_capitalization():
    """str.capitalize() turned "MRI" into "Mri" and "iPhone" into "Iphone"."""
    tagged_words = [
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=0.0, end_sec=0.5, word=" MRI"), punctuation_tag="NONE"),
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=0.5, end_sec=1.0, word=" shows"), punctuation_tag="NONE"),
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=1.0, end_sec=1.5, word=" damage"), punctuation_tag="."),
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=1.5, end_sec=2.0, word=" iPhone"), punctuation_tag="NONE"),
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=2.0, end_sec=2.5, word=" video"), punctuation_tag="."),
    ]

    sentences = assemble_sentences("Speaker_0", tagged_words)

    assert [s.text for s in sentences] == ["MRI shows damage.", "IPhone video."]


def test_assemble_sentences_never_writes_non_punctuation_tags_into_text():
    """A leaked "0" tag used to produce text like "So0" and "drill0"."""
    tagged_words = [
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=0.0, end_sec=0.5, word="so"), punctuation_tag="0"),
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=0.5, end_sec=1.0, word="we"), punctuation_tag="LABEL_7"),
        PunctuatedWord(acoustic_word=TranscriptWord(start_sec=1.0, end_sec=1.5, word="drill"), punctuation_tag="."),
    ]

    sentences = assemble_sentences("Speaker_0", tagged_words)

    assert [s.text for s in sentences] == ["So we drill."]
