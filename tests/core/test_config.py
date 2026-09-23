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

    @pytest.mark.parametrize("code, fts", [("sr", "serbian"), ("sl", "simple")])
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
        settings = Settings(APPLIANCE_LANGUAGE="sl", GLOSSARY_PATH=str(tmp_path))

        assert settings.GLOSSARY_DIR == tmp_path / "sl"
        assert settings.GLOSSARY_DIR.is_dir()
