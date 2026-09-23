# Path: tests/features/sentence_reconstruction/data/test_whisper_punctuation_adapter.py

import pytest

from app.core.model_manager.domain.entities import ModelManifest
from app.core.model_manager.domain.enums import ModelType
from app.features.transcription.domain.entities import TranscriptWord
from app.features.sentence_reconstruction.data.whisper_punctuation_adapter import WhisperPunctuationAdapter


def _word(text: str, start: float = 0.0) -> TranscriptWord:
    return TranscriptWord(start_sec=start, end_sec=start + 0.3, word=text)


class TestWhisperPunctuationAdapter:
    def test_requires_no_model(self):
        assert WhisperPunctuationAdapter().requires_model is False

    def test_load_and_unload_touch_nothing(self):
        manifest = ModelManifest(model_type=ModelType.SENTENCE_RECONSTRUCTION, model_path="unused", required_vram_mb=0)
        adapter = WhisperPunctuationAdapter()

        loaded = adapter.load(manifest)

        assert loaded.model_instance is None
        assert adapter.unload(loaded) is True

    @pytest.mark.parametrize("text, tag", [
        (" rekao,", ","), (" dan.", "."), (" ne?", "?"), (" stani!", "!"), (" sledeće:", ":"),
        (" i", "NONE"), (" 160", "NONE"), (" čekaj...", "."), ("", "NONE"), (" (npr)", "NONE"),
    ])
    def test_tags_each_word_with_whispers_own_trailing_mark(self, text, tag):
        assert WhisperPunctuationAdapter().predict_punctuation(None, [_word(text)])[0].punctuation_tag == tag

    def test_one_tag_per_word_with_words_untouched(self):
        words = [_word(" Pacijent", 0.0), _word(" je", 0.3), _word(" stabilan.", 0.6)]

        tagged = WhisperPunctuationAdapter().predict_punctuation(None, words)

        assert [t.acoustic_word for t in tagged] == words
        assert [t.punctuation_tag for t in tagged] == ["NONE", "NONE", "."]
