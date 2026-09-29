# Path: tests/core/language/service/test_language_api.py

from unittest.mock import patch

from app.core.config import settings
from app.core.language.service import language_api


class TestGetLanguagePack:
    def test_serves_the_configured_language(self, appliance_language):
        appliance_language("sr")

        assert language_api.get_language_pack().code == "sr"

    def test_loads_each_language_once(self, appliance_language):
        language_api.clear_language_pack_cache()
        with patch.object(language_api.TomlLanguagePackRepository, "load",
                          side_effect=lambda code: code) as load:
            appliance_language("sr")
            first = language_api.get_language_pack()
            second = language_api.get_language_pack()
        language_api.clear_language_pack_cache()

        assert first == second == "sr"
        load.assert_called_once_with("sr")

    def test_reads_packs_from_the_configured_directory(self, monkeypatch):
        language_api.clear_language_pack_cache()
        monkeypatch.setattr(settings, "LANGUAGE_PACKS_PATH", "/opt/onyx/languages")
        with patch.object(language_api, "TomlLanguagePackRepository") as repository:
            language_api.get_language_pack()
        language_api.clear_language_pack_cache()

        repository.assert_called_once_with("/opt/onyx/languages")

    def test_serves_a_named_language_regardless_of_the_active_one(self, appliance_language):
        appliance_language("en")

        assert language_api.get_language_pack("sr").code == "sr"

    def test_follows_a_runtime_switch(self, appliance_language):
        appliance_language("en")
        assert language_api.get_language_pack().code == "en"

        appliance_language("sr")
        assert language_api.get_language_pack().code == "sr"


class TestSupportedLanguagePacks:
    def test_one_pack_per_supported_language_with_its_own_name(self):
        packs = language_api.get_supported_language_packs()

        assert [(p.code, p.native_name) for p in packs] == [
            ("en", "English"), ("sr", "Srpski")
        ]
