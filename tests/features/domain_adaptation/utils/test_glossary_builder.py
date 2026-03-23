# Canvas: test_glossary_builder_v2
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

        assert len(labels) == 2
        assert "OPTIONS_TRADING" in labels
        assert "M_AND_A" in labels

    @patch.object(GlossaryBuilder, '_load_master_glossary')
    def test_build_injection_payloads_multi_domain(self, mock_load_glossary, mock_manifest_data):
        mock_load_glossary.return_value = mock_manifest_data

        hotwords, initial_prompt = GlossaryBuilder.build_injection_payloads(
            industry="FINANCE",
            predicted_domains=["OPTIONS_TRADING", "M_AND_A"]
        )

        assert "strike:15.0" in hotwords
        assert "merger:15.0" in hotwords

        # Verify the actual template string from prompts.py was formatted correctly
        assert "The following dictation is a formal, professional transcript covering topics related to Options Trading, M And A" in initial_prompt
        assert "Use proper capitalization and punctuation." in initial_prompt

    @patch.object(GlossaryBuilder, '_load_master_glossary')
    def test_build_injection_payloads_fallback_to_base(self, mock_load_glossary, mock_manifest_data):
        mock_load_glossary.return_value = mock_manifest_data

        # LLM hallucinated a domain
        hotwords, initial_prompt = GlossaryBuilder.build_injection_payloads(
            industry="FINANCE",
            predicted_domains=["CRYPTO_TRADING"]
        )

        assert "EBITDA:15.0" in hotwords
        assert "The following dictation is a formal, professional transcript covering topics related to Crypto Trading" in initial_prompt
        assert "Use proper capitalization and punctuation." in initial_prompt