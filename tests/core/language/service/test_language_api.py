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


class TestLanguageScope:
    """The hosted API's per-request answer language: a context-local override of the appliance's language."""

    def test_without_a_scope_the_appliance_language_is_active(self, appliance_language):
        from app.core.language.service.language_api import active_language
        appliance_language("en")

        assert active_language() == "en"

    def test_a_scope_overrides_the_language_and_the_pack_and_is_undone_after(self, appliance_language):
        from app.core.language.service.language_api import active_language, get_language_pack, language_scope
        appliance_language("en")

        with language_scope("sr"):
            assert active_language() == "sr"
            assert get_language_pack().code == "sr"

        assert active_language() == "en"
        assert get_language_pack().code == "en"

    def test_scopes_nest(self, appliance_language):
        from app.core.language.service.language_api import active_language, language_scope
        appliance_language("en")

        with language_scope("sr"):
            with language_scope("en"):
                assert active_language() == "en"
            assert active_language() == "sr"

    def test_an_unsupported_language_is_refused(self):
        import pytest
        from app.core.language.service.language_api import language_scope

        with pytest.raises(ValueError, match="'sl' is not a supported language"):
            with language_scope("sl"):
                pass

    def test_another_thread_does_not_see_the_override(self, appliance_language):
        import threading
        from app.core.language.service.language_api import active_language, language_scope
        appliance_language("en")
        seen = []

        with language_scope("sr"):
            thread = threading.Thread(target=lambda: seen.append(active_language()))
            thread.start()
            thread.join()

        assert seen == ["en"]

    def test_submit_in_context_carries_the_override_to_a_pool(self, appliance_language):
        from concurrent.futures import ThreadPoolExecutor
        from app.core.language.service.language_api import active_language, language_scope, submit_in_context
        appliance_language("en")

        with ThreadPoolExecutor(1) as pool, language_scope("sr"):
            assert submit_in_context(pool, active_language).result() == "sr"
            assert pool.submit(active_language).result() == "en"
