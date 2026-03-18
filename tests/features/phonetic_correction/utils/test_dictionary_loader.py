# Canvas: test_dictionary_loader_v1
# Path: tests/features/phonetic_correction/utils/test_dictionary_loader.py

import json
import pytest
from unittest.mock import patch, mock_open

from app.features.phonetic_correction.utils.dictionary_loader import DictionaryLoader


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
                }
            }
        }

    @patch("app.features.phonetic_correction.utils.dictionary_loader.Path.exists")
    @patch("builtins.open")
    def test_build_prompt_dictionary_success(self, mock_file_open, mock_path_exists, mock_manifest_data):
        """Verifies it correctly merges and formats base terms with sub-domain terms."""
        mock_path_exists.return_value = True
        mock_file_open.return_value = mock_open(read_data=json.dumps(mock_manifest_data)).return_value

        result = DictionaryLoader.build_prompt_dictionary(industry="FINANCE", sub_domain="OPTIONS_TRADING")

        # It should deduplicate, sort alphabetically, and format as bullet points
        expected_output = (
            "- EBITDA\n"
            "- Gamma\n"
            "- Revenue\n"
            "- strike\n"
            "- theta"
        )
        assert result == expected_output

    @patch("app.features.phonetic_correction.utils.dictionary_loader.Path.exists")
    @patch("builtins.open")
    def test_build_prompt_dictionary_fallback_to_base_only(self, mock_file_open, mock_path_exists, mock_manifest_data):
        """If the sub-domain isn't in the JSON, it should gracefully just return the base terms."""
        mock_path_exists.return_value = True
        mock_file_open.return_value = mock_open(read_data=json.dumps(mock_manifest_data)).return_value

        result = DictionaryLoader.build_prompt_dictionary(industry="FINANCE", sub_domain="UNKNOWN_SUBDOMAIN")

        expected_output = (
            "- EBITDA\n"
            "- Revenue"
        )
        assert result == expected_output

    @patch("app.features.phonetic_correction.utils.dictionary_loader.Path.exists")
    def test_build_prompt_dictionary_file_not_found(self, mock_path_exists):
        """If the industry file doesn't exist, it should catch it and return an empty string."""
        mock_path_exists.return_value = False

        result = DictionaryLoader.build_prompt_dictionary(industry="GHOST_INDUSTRY", sub_domain="GENERAL")

        assert result == ""

    @patch("app.features.phonetic_correction.utils.dictionary_loader.Path.exists")
    @patch("builtins.open")
    def test_build_prompt_dictionary_corrupted_json(self, mock_file_open, mock_path_exists):
        """If the JSON is corrupted, the error mapper should catch it and return an empty string."""
        mock_path_exists.return_value = True
        mock_file_open.return_value = mock_open(read_data="{ bad json {").return_value

        result = DictionaryLoader.build_prompt_dictionary(industry="FINANCE", sub_domain="GENERAL")

        assert result == ""