# Canvas: test_glossary_builder_v4
# Path: tests/features/domain_adaptation/utils/test_glossary_builder.py

import pytest
from unittest.mock import patch

from app.features.domain_adaptation.utils.glossary_builder import GlossaryBuilder


class TestGlossaryBuilder:
    @pytest.fixture
    def mock_manifest_data(self):
        return {
            "base_terms": ["EBITDA", "CapEx"],
            "sub_domains": {
                "OPTIONS_TRADING": {"anchor_words": ["strike", "theta"], "terms": ["Gamma"]},
                "M_AND_A": {"anchor_words": ["merger", "acquisition"]}
            }
        }

    @patch.object(GlossaryBuilder, '_load_master_glossary')
    def test_get_candidate_labels(self, mock_load_glossary, mock_manifest_data):
        mock_load_glossary.return_value = mock_manifest_data

        labels = GlossaryBuilder.get_candidate_labels("FINANCE")

        assert labels == ["OPTIONS_TRADING", "M_AND_A"]

    @patch.object(GlossaryBuilder, '_load_master_glossary')
    def test_get_candidate_descriptions_uses_anchor_words_then_terms(self, mock_load_glossary, mock_manifest_data):
        mock_manifest_data["sub_domains"]["LEGACY_LIST"] = ["alpha", "beta"]
        mock_load_glossary.return_value = mock_manifest_data

        descriptions = GlossaryBuilder.get_candidate_descriptions("FINANCE")

        assert descriptions == {
            "OPTIONS_TRADING": "strike, theta, Gamma",
            "M_AND_A": "merger, acquisition",
            "LEGACY_LIST": "alpha, beta",
        }
        mock_load_glossary.assert_called_once_with("FINANCE")

    @patch.object(GlossaryBuilder, '_load_master_glossary')
    def test_candidate_descriptions_are_capped(self, mock_load_glossary):
        mock_load_glossary.return_value = {"sub_domains": {"BIG": {"anchor_words": [f"a{i}" for i in range(8)],
                                                                   "terms": [f"t{i}" for i in range(8)]}}}

        descriptions = GlossaryBuilder.get_candidate_descriptions("X")

        assert descriptions["BIG"] == ", ".join([f"a{i}" for i in range(8)] + ["t0", "t1"])

    @patch.object(GlossaryBuilder, '_load_master_glossary')
    def test_build_injection_payloads_multi_domain(self, mock_load_glossary, mock_manifest_data):
        mock_load_glossary.return_value = mock_manifest_data

        hotwords, initial_prompt = GlossaryBuilder.build_injection_payloads(
            industry="FINANCE",
            predicted_domains=["OPTIONS_TRADING", "M_AND_A"]
        )

        # Plain comma-separated terms: faster-whisper has no "term:weight" syntax
        assert hotwords == "strike, theta, merger, acquisition"
        # No sentence prompt: Whisper copied "This recording discusses ..." into transcripts as if it had been said
        assert initial_prompt == ""

    @patch.object(GlossaryBuilder, '_load_master_glossary')
    def test_build_injection_payloads_fallback_to_base(self, mock_load_glossary, mock_manifest_data):
        mock_load_glossary.return_value = mock_manifest_data

        # LLM hallucinated a domain
        hotwords, initial_prompt = GlossaryBuilder.build_injection_payloads(
            industry="FINANCE",
            predicted_domains=["CRYPTO_TRADING"]
        )

        assert hotwords == "EBITDA, CapEx"
        assert initial_prompt == ""

    @patch.object(GlossaryBuilder, '_load_master_glossary')
    def test_build_injection_payloads_without_terms(self, mock_load_glossary):
        mock_load_glossary.return_value = {"base_terms": [], "sub_domains": {}}

        assert GlossaryBuilder.build_injection_payloads(industry="GENERAL", predicted_domains=[]) == ("", "")

    @patch.object(GlossaryBuilder, '_load_master_glossary')
    def test_hotwords_are_capped(self, mock_load_glossary):
        terms = [f"term{i}" for i in range(60)]
        mock_load_glossary.return_value = {"base_terms": terms, "sub_domains": {}}

        hotwords, _ = GlossaryBuilder.build_injection_payloads(industry="X", predicted_domains=[])

        assert hotwords == ", ".join(terms[:GlossaryBuilder.MAX_ANCHOR_WORDS])
