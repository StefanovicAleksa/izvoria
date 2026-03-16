# Canvas: test_glossary_builder_v1
# Path: tests/features/domain_adaptation/test_glossary_builder.py

import pytest
from app.features.domain_adaptation.utils.glossary_builder import GlossaryBuilder


class TestGlossaryBuilder:

    @pytest.fixture
    def mock_manifest(self, mocker):
        """Mocks the physical JSON file read to provide controlled test data."""
        manifest_data = {
            "base_terms": ["EBITDA"],
            "sub_domains": {
                "OPTIONS_TRADING": {
                    "anchor_words": ["strike", "theta"],
                    "terms": ["Gamma"]
                }
            }
        }
        mocker.patch.object(GlossaryBuilder, "_load_master_glossary", return_value=manifest_data)

    def test_get_candidate_labels(self, mock_manifest):
        labels = GlossaryBuilder.get_candidate_labels("FINANCE")
        assert labels == ["OPTIONS_TRADING"]

    def test_validate_and_build_success(self, mock_manifest):
        """If anchor words are found, it should merge base, domain, and anchor words."""
        subdomain, hotwords = GlossaryBuilder.validate_and_build(
            industry="FINANCE",
            guessed_subdomain="OPTIONS_TRADING",
            scouted_text="We are looking at the strike price today."
        )

        assert subdomain == "OPTIONS_TRADING"
        # Verify all three tiers made it into the hotwords
        assert "EBITDA:15.0" in hotwords
        assert "Gamma:15.0" in hotwords
        assert "strike:15.0" in hotwords

    def test_validate_and_build_anchor_failure(self, mock_manifest):
        """If no anchor words are found, it should fall back to BASE."""
        subdomain, hotwords = GlossaryBuilder.validate_and_build(
            industry="FINANCE",
            guessed_subdomain="OPTIONS_TRADING",
            scouted_text="Just some normal financial conversation."
        )

        assert subdomain == "BASE_FALLBACK"
        # Only base terms should be injected
        assert "EBITDA:15.0" in hotwords
        assert "Gamma:15.0" not in hotwords
        assert "strike:15.0" not in hotwords

    def test_validate_and_build_general_domain(self, mock_manifest):
        """If the classifier guessed GENERAL (or scout failed), it just returns base terms."""
        subdomain, hotwords = GlossaryBuilder.validate_and_build(
            industry="FINANCE",
            guessed_subdomain="GENERAL",
            scouted_text=""
        )

        assert subdomain == "GENERAL"
        assert "EBITDA:15.0" in hotwords