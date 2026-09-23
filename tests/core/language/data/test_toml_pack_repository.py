# Path: tests/core/language/data/test_toml_pack_repository.py

from pathlib import Path
from unittest.mock import patch

import pytest

from app.core.config import settings
from app.core.language.data.toml_pack_repository import TomlLanguagePackRepository
from app.core.language.exceptions import InvalidLanguagePackError, LanguagePackNotFoundError

SHIPPED = settings.BASE_DIR / "data" / "languages"


class TestShippedPacks:
    """The packs checked in beside the code must all load and validate."""

    def test_every_supported_language_ships_a_pack(self):
        assert TomlLanguagePackRepository(SHIPPED).available_codes() == ["en", "sl", "sr"]

    @pytest.mark.parametrize("code", ["en", "sr", "sl"])
    def test_pack_loads_and_validates(self, code):
        pack = TomlLanguagePackRepository(SHIPPED).load(code)

        assert pack.code == code
        assert pack.transcription.whisper_language == code
        assert pack.transcription.whisper_language in pack.transcription.accepted_detections

    def test_english_pack_resolves_nltk_stopwords(self):
        pack = TomlLanguagePackRepository(SHIPPED).load("en")

        assert {"the", "and", "of"} <= set(pack.text.stopwords)

    def test_inline_stopwords_are_lowercased_deduplicated_and_sorted(self):
        stopwords = TomlLanguagePackRepository(SHIPPED).load("sl").text.stopwords

        assert list(stopwords) == sorted(set(stopwords))
        assert stopwords.count("bi") == 1


class TestTomlLanguagePackRepository:
    @staticmethod
    def _write(tmp_path: Path, code: str, text: str) -> TomlLanguagePackRepository:
        (tmp_path / f"{code}.toml").write_text(text, encoding="utf-8")
        return TomlLanguagePackRepository(tmp_path)

    def test_missing_pack_names_the_available_ones(self, tmp_path):
        (tmp_path / "en.toml").write_text("", encoding="utf-8")

        with pytest.raises(LanguagePackNotFoundError, match="Available: en"):
            TomlLanguagePackRepository(tmp_path).load("de")

    def test_missing_directory_has_no_codes(self, tmp_path):
        assert TomlLanguagePackRepository(tmp_path / "absent").available_codes() == []

    def test_malformed_toml_is_invalid(self, tmp_path):
        with pytest.raises(InvalidLanguagePackError):
            self._write(tmp_path, "xx", "code = ").load("xx")

    def test_pack_missing_tables_is_invalid(self, tmp_path):
        with pytest.raises(InvalidLanguagePackError, match="xx.toml"):
            self._write(tmp_path, "xx", 'code = "xx"\nname = "X"\n').load("xx")

    def test_regex_that_does_not_compile_is_invalid(self, tmp_path):
        text = (SHIPPED / "sr.toml").read_text(encoding="utf-8").replace(
            "hallucination_patterns = [", "hallucination_patterns = ['(unclosed',", 1
        )
        with pytest.raises(InvalidLanguagePackError, match="invalid regular expression"):
            self._write(tmp_path, "sr", text).load("sr")

    def test_unknown_key_is_invalid(self, tmp_path):
        text = (SHIPPED / "sl.toml").read_text(encoding="utf-8").replace(
            "[search]", "[search]\nstemmer = \"snowball\"", 1
        )
        with pytest.raises(InvalidLanguagePackError):
            self._write(tmp_path, "sl", text).load("sl")

    def test_nltk_corpus_is_merged_with_inline_stopwords(self, tmp_path):
        text = (SHIPPED / "sl.toml").read_text(encoding="utf-8").replace(
            "[text]", '[text]\nnltk_stopwords = "fake"', 1
        )
        with patch("nltk.corpus.stopwords.words", return_value=["Zzz"]) as words:
            pack = self._write(tmp_path, "sl", text).load("sl")

        words.assert_called_once_with("fake")
        assert "zzz" in pack.text.stopwords and "in" in pack.text.stopwords
