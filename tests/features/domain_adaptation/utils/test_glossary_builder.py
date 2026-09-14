# Canvas: test_glossary_builder_v3
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
                "OPTIONS_TRADING": {"anchor_words": ["strike", "theta"]},
                "M_AND_A": {"anchor_words": ["merger", "acquisition"]}
            }
        }

    @patch.object(GlossaryBuilder, '_load_master_glossary')
    def test_get_candidate_labels(self, mock_load_glossary, mock_manifest_data):
        mock_load_glossary.return_value = mock_manifest_data

        labels = GlossaryBuilder.get_candidate_labels("FINANCE")

        assert labels == ["OPTIONS_TRADING", "M_AND_A"]

    @patch.object(GlossaryBuilder, '_load_master_glossary')
    def test_build_injection_payloads_multi_domain(self, mock_load_glossary, mock_manifest_data):
        mock_load_glossary.return_value = mock_manifest_data

        hotwords, initial_prompt = GlossaryBuilder.build_injection_payloads(
            industry="FINANCE",
            predicted_domains=["OPTIONS_TRADING", "M_AND_A"]
        )

        # Plain comma-separated terms: faster-whisper has no "term:weight" syntax
        assert hotwords == "strike, theta, merger, acquisition"
        assert initial_prompt == (
            "This recording discusses Options Trading, M And A, including terms such as "
            "strike, theta, merger, acquisition."
        )

    @patch.object(GlossaryBuilder, '_load_master_glossary')
    def test_build_injection_payloads_fallback_to_base(self, mock_load_glossary, mock_manifest_data):
        mock_load_glossary.return_value = mock_manifest_data

        # LLM hallucinated a domain
        hotwords, initial_prompt = GlossaryBuilder.build_injection_payloads(
            industry="FINANCE",
            predicted_domains=["CRYPTO_TRADING"]
        )

        assert hotwords == "EBITDA, CapEx"
        assert initial_prompt == "This recording discusses Crypto Trading, including terms such as EBITDA, CapEx."

    @patch.object(GlossaryBuilder, '_load_master_glossary')
    def test_build_injection_payloads_without_terms(self, mock_load_glossary):
        mock_load_glossary.return_value = {"base_terms": [], "sub_domains": {}}

        hotwords, initial_prompt = GlossaryBuilder.build_injection_payloads(industry="GENERAL", predicted_domains=[])

        assert hotwords == ""
        assert initial_prompt == "This recording discusses General."

    @patch.object(GlossaryBuilder, '_load_master_glossary')
    def test_prompt_terms_are_capped(self, mock_load_glossary):
        terms = [f"term{i}" for i in range(30)]
        mock_load_glossary.return_value = {"base_terms": terms, "sub_domains": {}}

        hotwords, initial_prompt = GlossaryBuilder.build_injection_payloads(industry="X", predicted_domains=[])

        assert hotwords == ", ".join(terms)
        assert initial_prompt.endswith("term11.")
        assert "term12" not in initial_prompt
