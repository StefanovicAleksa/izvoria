# Path: tests/features/transcription/utils/test_prompt_echo_filter.py

from app.features.transcription.domain.entities import TranscriptSegment, TranscriptWord
from app.features.transcription.utils.prompt_echo_filter import remove_prompt_echoes

# The strings injected into Whisper on the first real ingestion, which it then repeated into the transcript
STORED_HOTWORDS = "affidavit, plaintiff, diagnosis, cardiology"
STORED_PROMPT = "This recording discusses Legal, Medical, including terms such as affidavit, plaintiff, diagnosis, cardiology."


def _segment(*words: str, start: float = 0.0) -> TranscriptSegment:
    transcript_words = [
        TranscriptWord(start_sec=start + i, end_sec=start + i + 0.5, word=w) for i, w in enumerate(words)
    ]
    return TranscriptSegment(
        start_sec=transcript_words[0].start_sec,
        end_sec=transcript_words[-1].end_sec,
        text="".join(words).strip(),
        words=transcript_words
    )


class TestRemovePromptEchoes:
    def test_drops_a_segment_that_only_repeats_the_prompt(self):
        leaked = _segment(" This", " recording", " discusses", " Legal,", " Medical,", " including", " terms",
                          " such", " as", " affidavit,")

        cleaned, removed = remove_prompt_echoes([leaked], [STORED_HOTWORDS, STORED_PROMPT], 5)

        assert cleaned == []
        assert removed == 10

    def test_keeps_real_speech_around_an_echo_and_rebuilds_the_segment(self):
        segment = _segment(" So", " watch", " the", " Bund.", " Legal,", " Medical,", " including", " terms", " such",
                           " as", " affidavit.", " Then", " sell.", start=100.0)

        cleaned, removed = remove_prompt_echoes([segment], [STORED_HOTWORDS, STORED_PROMPT], 5)

        assert removed == 7
        assert [(s.text, s.start_sec, s.end_sec) for s in cleaned] == [("So watch the Bund. Then sell.", 100.0, 112.5)]
        assert [w.word for w in cleaned[0].words] == [" So", " watch", " the", " Bund.", " Then", " sell."]

    def test_single_glossary_terms_in_real_speech_are_kept(self):
        prompt = "This recording discusses European Central Bank, Bank Of England, including terms such as ECB, Lagarde."
        segment = _segment(" The", " Bank", " of", " England", " decision", " and", " the", " ECB.")

        cleaned, removed = remove_prompt_echoes([segment], ["ECB, Lagarde, Frankfurt, Bunds", prompt], 5)

        assert cleaned == [segment]
        assert removed == 0

    def test_runs_shorter_than_the_threshold_are_kept(self):
        segment = _segment(" affidavit,", " plaintiff,", " diagnosis.")

        assert remove_prompt_echoes([segment], [STORED_HOTWORDS], 5) == ([segment], 0)

    def test_punctuation_only_words_inside_an_echo_are_removed_with_it(self):
        segment = _segment(" We", " trade.", " ECB", ",", " Lagarde", ",", " Frankfurt", ",", " Bunds", ",", " BOE")

        cleaned, removed = remove_prompt_echoes([segment], ["ECB, Lagarde, Frankfurt, Bunds, BOE"], 5)

        assert [s.text for s in cleaned] == ["We trade."]
        assert removed == 9

    def test_segment_without_words_is_dropped_only_when_fully_echoed(self):
        echoed = TranscriptSegment(start_sec=0.0, end_sec=2.0, text="Legal, Medical, including terms such as affidavit.")
        mixed = TranscriptSegment(start_sec=3.0, end_sec=6.0,
                                  text="Legal, Medical, including terms such as affidavit and then we trade.")

        cleaned, removed = remove_prompt_echoes([echoed, mixed], [STORED_PROMPT], 5)

        assert cleaned == [mixed]
        assert removed == 7

    def test_nothing_injected_changes_nothing(self):
        segment = _segment(" This", " recording", " discusses", " rates", " today", " with", " you.")

        assert remove_prompt_echoes([segment], [None, "", "   "], 5) == ([segment], 0)
