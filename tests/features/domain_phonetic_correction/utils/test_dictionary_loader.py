# File: tests/features/domain_phonetic_correction/utils/test_dictionary_loader.py
# Canvas: test_dictionary_loader_v6
# Path: tests/features/domain_phonetic_correction/utils/test_dictionary_loader.py

import json
import pytest
from unittest.mock import patch, mock_open
from pathlib import Path
from app.features.domain_phonetic_correction.utils.dictionary_loader import DictionaryLoader


class TestDictionaryLoader:

    @pytest.fixture
    def mock_manifest_data(self):
        """Mock JSON data replicating the structure of our master glossaries."""
        return {
            "base_terms": ["EBITDA", "Revenue"],
            "sub_domains": {
                "OPTIONS_TRADING": {
                    "anchor_words": ["strike", "theta"],
                    "terms": ["Gamma"]
                },
                "M_AND_A": {
                    "anchor_words": ["merger"],
                    "terms": ["synergy"]
                }
            }
        }

    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.settings")
    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.Path.exists")
    @patch("builtins.open")
    def test_build_prompt_dictionary_success_single(self, mock_file_open, mock_path_exists, mock_settings,
                                                    mock_manifest_data):
        """Verifies it correctly merges and formats base terms with a single sub-domain."""
        mock_settings.GLOSSARY_DIR = Path("/mock/glossaries")
        mock_path_exists.return_value = True
        mock_file_open.return_value = mock_open(read_data=json.dumps(mock_manifest_data)).return_value

        result = DictionaryLoader.build_prompt_dictionary(industry="FINANCE", sub_domains=["OPTIONS_TRADING"])

        expected_terms = sorted(["EBITDA", "Revenue", "strike", "theta", "Gamma"])
        expected_output = "\n".join([f"- {term}" for term in expected_terms])

        assert result == expected_output

    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.settings")
    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.Path.exists")
    @patch("builtins.open")
    def test_build_prompt_dictionary_multiple_subdomains(self, mock_file_open, mock_path_exists, mock_settings,
                                                         mock_manifest_data):
        """Verifies aggregation across multiple sub-domains works correctly."""
        mock_settings.GLOSSARY_DIR = Path("/mock/glossaries")
        mock_path_exists.return_value = True
        mock_file_open.return_value = mock_open(read_data=json.dumps(mock_manifest_data)).return_value

        result = DictionaryLoader.build_prompt_dictionary(industry="FINANCE",
                                                          sub_domains=["OPTIONS_TRADING", "M_AND_A"])

        all_terms = ["EBITDA", "Revenue", "strike", "theta", "Gamma", "merger", "synergy"]
        expected_output = "\n".join([f"- {term}" for term in sorted(all_terms)])

        assert result == expected_output

    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.settings")
    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.Path.exists")
    @patch("builtins.open")
    def test_build_prompt_dictionary_unknown_subdomain(self, mock_file_open, mock_path_exists, mock_settings,
                                                       mock_manifest_data):
        """If a requested sub-domain isn't in the JSON, it returns base terms."""
        mock_settings.GLOSSARY_DIR = Path("/mock/glossaries")
        mock_path_exists.return_value = True
        mock_file_open.return_value = mock_open(read_data=json.dumps(mock_manifest_data)).return_value

        result = DictionaryLoader.build_prompt_dictionary(industry="FINANCE", sub_domains=["UNKNOWN"])

        assert result == "- EBITDA\n- Revenue"

    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.settings")
    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.Path.exists")
    def test_build_prompt_dictionary_file_not_found(self, mock_path_exists, mock_settings):
        """If the industry file doesn't exist, it returns an empty string."""
        mock_settings.GLOSSARY_DIR = Path("/mock/glossaries")
        mock_path_exists.return_value = False

        result = DictionaryLoader.build_prompt_dictionary(industry="GHOST", sub_domains=["GENERAL"])

        assert result == ""

    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.settings")
    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.Path.exists")
    @patch("builtins.open")
    def test_build_prompt_dictionary_corrupted_json(self, mock_file_open, mock_path_exists, mock_settings):
        """If the JSON is corrupted, it handles the exception."""
        mock_settings.GLOSSARY_DIR = Path("/mock/glossaries")
        mock_path_exists.return_value = True
        mock_file_open.return_value = mock_open(read_data="{ bad json {").return_value

        result = DictionaryLoader.build_prompt_dictionary(industry="FINANCE", sub_domains=["GENERAL"])

        assert result == ""