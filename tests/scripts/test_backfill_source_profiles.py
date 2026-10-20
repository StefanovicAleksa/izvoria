# Path: tests/scripts/test_backfill_source_profiles.py

import uuid
from unittest.mock import MagicMock

from app.core.config import settings
from app.features.source_profile.domain.entities import SourceIdentity
from scripts.backfill_source_profiles import _by_language, _profile_in


def _identity(language):
    return SourceIdentity(source_id=uuid.uuid4(), name="lecture.mp4", language=language)


class TestByLanguage:
    def test_sources_are_grouped_by_their_language_and_unknown_ones_take_the_active(self, appliance_language):
        appliance_language("en")
        english, serbian, unknown = _identity("en"), _identity("sr"), _identity(None)

        groups = _by_language([english, serbian, unknown])

        assert groups == {"en": [english, unknown], "sr": [serbian]}


class TestProfileIn:
    def test_serbian_sources_are_profiled_by_the_serbian_packs_model_and_the_language_is_restored(
            self, appliance_language):
        appliance_language("en")
        english_model = settings.CONTEXTUAL_RETRIEVAL_MODEL_PATH
        seen = {}
        contextualizer = MagicMock()
        contextualizer.profile_sources.return_value = 2

        def contextualizer_for():
            seen["model"], seen["language"] = settings.CONTEXTUAL_RETRIEVAL_MODEL_PATH, settings.APPLIANCE_LANGUAGE
            return contextualizer

        work = [(uuid.uuid4(), ["tekst"], "sr")]
        assert _profile_in("sr", work, contextualizer_for) == 2

        assert seen["language"] == "sr"
        assert "gemma" in seen["model"].lower() and seen["model"] != english_model
        contextualizer.profile_sources.assert_called_once_with(work)
        assert settings.APPLIANCE_LANGUAGE == "en"
        assert settings.CONTEXTUAL_RETRIEVAL_MODEL_PATH == english_model

    def test_the_language_is_restored_when_profiling_fails(self, appliance_language):
        appliance_language("en")
        contextualizer = MagicMock()
        contextualizer.profile_sources.side_effect = RuntimeError("VRAM")

        try:
            _profile_in("sr", [], lambda: contextualizer)
        except RuntimeError:
            pass

        assert settings.APPLIANCE_LANGUAGE == "en"

    def test_a_language_without_a_served_pack_is_skipped(self, appliance_language):
        appliance_language("en")
        contextualizer_for = MagicMock()

        assert _profile_in("sl", [(uuid.uuid4(), ["besedilo"], "sl")], contextualizer_for) == 0

        contextualizer_for.assert_not_called()
        assert settings.APPLIANCE_LANGUAGE == "en"
