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
                    "anchor_words": ["merger", " strike "],
                    "terms": ["synergy"]
                }
            }
        }

    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.settings")
    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.Path.exists")
    @patch("builtins.open")
    def test_load_terms_single_subdomain(self, mock_file_open, mock_path_exists, mock_settings, mock_manifest_data):
        """Base terms plus one sub-domain's anchor words and terms, sorted."""
        mock_settings.GLOSSARY_DIR = Path("/mock/glossaries")
        mock_path_exists.return_value = True
        mock_file_open.return_value = mock_open(read_data=json.dumps(mock_manifest_data)).return_value

        result = DictionaryLoader.load_terms(industry="FINANCE", sub_domains=["OPTIONS_TRADING"])

        assert result == sorted(["EBITDA", "Revenue", "strike", "theta", "Gamma"])

    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.settings")
    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.Path.exists")
    @patch("builtins.open")
    def test_load_terms_multiple_subdomains_are_stripped_and_deduplicated(
            self, mock_file_open, mock_path_exists, mock_settings, mock_manifest_data):
        mock_settings.GLOSSARY_DIR = Path("/mock/glossaries")
        mock_path_exists.return_value = True
        mock_file_open.return_value = mock_open(read_data=json.dumps(mock_manifest_data)).return_value

        result = DictionaryLoader.load_terms(industry="FINANCE", sub_domains=["OPTIONS_TRADING", "M_AND_A"])

        assert result == sorted(["EBITDA", "Revenue", "strike", "theta", "Gamma", "merger", "synergy"])

    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.settings")
    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.Path.exists")
    @patch("builtins.open")
    def test_load_terms_unknown_subdomain_returns_base_terms(self, mock_file_open, mock_path_exists, mock_settings,
                                                             mock_manifest_data):
        mock_settings.GLOSSARY_DIR = Path("/mock/glossaries")
        mock_path_exists.return_value = True
        mock_file_open.return_value = mock_open(read_data=json.dumps(mock_manifest_data)).return_value

        result = DictionaryLoader.load_terms(industry="FINANCE", sub_domains=["UNKNOWN"])

        assert result == ["EBITDA", "Revenue"]

    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.settings")
    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.Path.exists")
    def test_load_terms_file_not_found(self, mock_path_exists, mock_settings):
        mock_settings.GLOSSARY_DIR = Path("/mock/glossaries")
        mock_path_exists.return_value = False

        assert DictionaryLoader.load_terms(industry="GHOST", sub_domains=["GENERAL"]) == []

    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.settings")
    @patch("app.features.domain_phonetic_correction.utils.dictionary_loader.Path.exists")
    @patch("builtins.open")
    def test_load_terms_corrupted_json(self, mock_file_open, mock_path_exists, mock_settings):
        mock_settings.GLOSSARY_DIR = Path("/mock/glossaries")
        mock_path_exists.return_value = True
        mock_file_open.return_value = mock_open(read_data="{ bad json {").return_value

        assert DictionaryLoader.load_terms(industry="FINANCE", sub_domains=["GENERAL"]) == []
