# Path: tests/core/test_config.py

import pytest
from pydantic import ValidationError

from app.core.config import Settings


def _pack(tmp_path, code: str, body: str) -> str:
    (tmp_path / f"{code}.toml").write_text(body, encoding="utf-8")
    return str(tmp_path)


class TestLanguagePackSettings:
    def test_english_keeps_every_default(self):
        settings = Settings(APPLIANCE_LANGUAGE="en")

        assert settings.FTS_LANGUAGE_CONFIG == "english"
        assert settings.CONTEXTUAL_RETRIEVAL_MODEL_PATH == "models/llama-3.2-3b-instruct-q8_0.gguf"
        assert settings.DISCOVERY_SCOUT_MODEL_PATH == "models/whisper-tiny-en"

    @pytest.mark.parametrize("code, fts", [("sr", "serbian")])
    def test_shipped_pack_swaps_small_models_and_fts(self, code, fts):
        settings = Settings(APPLIANCE_LANGUAGE=code)

        assert settings.FTS_LANGUAGE_CONFIG == fts
        gemma = "models/google_gemma-4-E4B-it-Q4_K_M.gguf"
        assert (settings.DOMAIN_CLASSIFIER_MODEL_PATH, settings.CONTEXTUAL_RETRIEVAL_MODEL_PATH,
                settings.VISUAL_CONTEXT_SYNTHESIS_MODEL_PATH, settings.DOMAIN_PHONETIC_CORRECTION_MODEL_PATH,
                settings.VISUAL_HALLUCINATION_CORRECTION_MODEL_PATH) == (gemma,) * 5
        assert settings.DISCOVERY_SCOUT_MODEL_PATH == "models/faster-whisper-large-v3"
        # The answer models serve every language unchanged
        assert settings.QA_GENERATION_MODEL_PATH == "models/Qwen3.5-9B-Q4_K_M.gguf"

    def test_explicit_setting_beats_the_pack(self):
        settings = Settings(APPLIANCE_LANGUAGE="sr", CONTEXTUAL_RETRIEVAL_MODEL_PATH="models/mine.gguf",
                            FTS_LANGUAGE_CONFIG="simple")

        assert settings.CONTEXTUAL_RETRIEVAL_MODEL_PATH == "models/mine.gguf"
        assert settings.FTS_LANGUAGE_CONFIG == "simple"
        assert settings.DOMAIN_CLASSIFIER_MODEL_PATH == "models/google_gemma-4-E4B-it-Q4_K_M.gguf"

    def test_override_values_are_coerced_to_the_field_type(self, tmp_path):
        packs = _pack(tmp_path, "xx", "[models]\nCONTEXTUAL_RETRIEVAL_N_CTX = 8192.0\n")

        settings = Settings(APPLIANCE_LANGUAGE="xx", LANGUAGE_PACKS_PATH=packs)

        assert settings.CONTEXTUAL_RETRIEVAL_N_CTX == 8192
        assert isinstance(settings.CONTEXTUAL_RETRIEVAL_N_CTX, int)

    def test_language_without_a_pack_is_rejected(self, tmp_path):
        with pytest.raises(ValidationError, match="has no language pack"):
            Settings(APPLIANCE_LANGUAGE="de", LANGUAGE_PACKS_PATH=str(tmp_path))

    def test_pack_overriding_an_unknown_setting_is_rejected(self, tmp_path):
        packs = _pack(tmp_path, "xx", '[models]\nWHISPER_MODLE_PATH = "typo"\n')

        with pytest.raises(ValidationError, match="unknown setting 'WHISPER_MODLE_PATH'"):
            Settings(APPLIANCE_LANGUAGE="xx", LANGUAGE_PACKS_PATH=packs)

    def test_pack_fts_config_must_be_an_identifier(self, tmp_path):
        packs = _pack(tmp_path, "xx", "[search]\nfts_config = \"english'); DROP TABLE x; --\"\n")

        with pytest.raises(ValidationError):
            Settings(APPLIANCE_LANGUAGE="xx", LANGUAGE_PACKS_PATH=packs)

    def test_glossaries_live_in_one_directory_per_language(self, tmp_path):
        settings = Settings(APPLIANCE_LANGUAGE="sr", GLOSSARY_PATH=str(tmp_path))

        assert settings.GLOSSARY_DIR == tmp_path / "sr"
        assert settings.GLOSSARY_DIR.is_dir()


GEMMA = "models/google_gemma-4-E4B-it-Q4_K_M.gguf"


class TestRuntimeLanguageSwitch:
    def test_switching_applies_the_new_pack(self):
        settings = Settings(APPLIANCE_LANGUAGE="en")

        settings.use_language("sr")

        assert settings.APPLIANCE_LANGUAGE == "sr"
        assert settings.FTS_LANGUAGE_CONFIG == "serbian"
        assert settings.CONTEXTUAL_RETRIEVAL_MODEL_PATH == GEMMA

    def test_switching_back_restores_the_defaults_the_pack_replaced(self):
        settings = Settings(APPLIANCE_LANGUAGE="sr")

        settings.use_language("en")

        assert settings.FTS_LANGUAGE_CONFIG == "english"
        assert settings.CONTEXTUAL_RETRIEVAL_MODEL_PATH == "models/llama-3.2-3b-instruct-q8_0.gguf"
        assert settings.DISCOVERY_SCOUT_MODEL_PATH == "models/whisper-tiny-en"
        assert settings.DISCOVERY_SCOUT_VRAM_REQUIREMENT_MB == 200.0

    def test_explicit_setting_survives_every_switch(self):
        settings = Settings(APPLIANCE_LANGUAGE="en", CONTEXTUAL_RETRIEVAL_MODEL_PATH="models/mine.gguf")

        settings.use_language("sr")
        assert settings.CONTEXTUAL_RETRIEVAL_MODEL_PATH == "models/mine.gguf"
        settings.use_language("en")
        assert settings.CONTEXTUAL_RETRIEVAL_MODEL_PATH == "models/mine.gguf"

    def test_unsupported_language_is_refused_and_nothing_changes(self):
        settings = Settings(APPLIANCE_LANGUAGE="sr")

        with pytest.raises(ValueError, match="'de' is not a supported language"):
            settings.use_language("de")

        assert settings.APPLIANCE_LANGUAGE == "sr"
        assert settings.FTS_LANGUAGE_CONFIG == "serbian"

    def test_supported_languages_are_the_shipped_packs(self):
        assert Settings().supported_languages() == ["en", "sr"]

    def test_a_legacy_language_is_refused_by_name(self):
        with pytest.raises(ValidationError, match="'sl' is a legacy language"):
            Settings(APPLIANCE_LANGUAGE="sl")

    def test_switching_to_a_legacy_language_is_refused(self):
        settings = Settings(APPLIANCE_LANGUAGE="en")

        with pytest.raises(ValueError, match="'sl' is not a supported language"):
            settings.use_language("sl")

    def test_search_covers_every_packs_fts_config(self):
        assert Settings(APPLIANCE_LANGUAGE="sr").SEARCH_FTS_CONFIGS == ["english", "serbian"]

    def test_search_also_covers_an_explicit_fts_config(self):
        settings = Settings(APPLIANCE_LANGUAGE="en", FTS_LANGUAGE_CONFIG="german")

        assert settings.SEARCH_FTS_CONFIGS == ["english", "serbian", "german"]


class TestRememberedLanguage:
    def test_starts_in_the_remembered_language(self, tmp_path):
        Settings(DATA_DIR=str(tmp_path), ENV="development").remember_language("sr")

        settings = Settings(DATA_DIR=str(tmp_path), ENV="development", REMEMBER_LANGUAGE=True)

        assert settings.APPLIANCE_LANGUAGE == "sr"
        assert settings.FTS_LANGUAGE_CONFIG == "serbian"
        assert (tmp_path / "development" / "active_language").read_text() == "sr"

    def test_explicit_language_beats_the_remembered_one(self, tmp_path):
        Settings(DATA_DIR=str(tmp_path), ENV="development").remember_language("sr")

        settings = Settings(DATA_DIR=str(tmp_path), ENV="development", REMEMBER_LANGUAGE=True, APPLIANCE_LANGUAGE="en")

        assert settings.APPLIANCE_LANGUAGE == "en"

    def test_ignored_when_turned_off(self, tmp_path):
        Settings(DATA_DIR=str(tmp_path), ENV="development").remember_language("sr")

        assert Settings(DATA_DIR=str(tmp_path), ENV="development", REMEMBER_LANGUAGE=False).APPLIANCE_LANGUAGE == "en"

    def test_the_test_suite_never_remembers(self):
        # The root conftest.py sets REMEMBER_LANGUAGE=false before anything imports app.core.config
        assert Settings().REMEMBER_LANGUAGE is False

    def test_a_remembered_legacy_language_is_ignored(self, tmp_path):
        (tmp_path / "development").mkdir()
        (tmp_path / "development" / "active_language").write_text("sl")

        assert Settings(DATA_DIR=str(tmp_path), ENV="development", REMEMBER_LANGUAGE=True).APPLIANCE_LANGUAGE == "en"

    def test_a_remembered_language_without_a_pack_is_ignored(self, tmp_path):
        (tmp_path / "development").mkdir()
        (tmp_path / "development" / "active_language").write_text("de")

        assert Settings(DATA_DIR=str(tmp_path), ENV="development", REMEMBER_LANGUAGE=True).APPLIANCE_LANGUAGE == "en"


class TestDatabaseUrl:
    def test_names_the_psycopg2_driver(self):
        # SQLAlchemy 2.1 made psycopg (3) the default for a bare postgresql://, which the appliance doesn't install
        from app.core.config import settings
        assert settings.DATABASE_URL.startswith("postgresql+psycopg2://")
