# Canvas: test_glossary_miner_api_v1
# Path: tests/features/glossary_miner/service/test_miner_api.py

import pytest
from unittest.mock import MagicMock

from app.features.glossary_miner.service.miner_api import GlossaryMinerAPI
from app.features.glossary_miner.domain.interfaces import IWikidataAdapter, IGlossaryStorageAdapter
from app.features.glossary_miner.domain.entities import WikidataItem
from app.features.glossary_miner.exceptions import DataProcessingError


@pytest.fixture
def mock_wikidata_adapter():
    return MagicMock(spec=IWikidataAdapter)


@pytest.fixture
def mock_storage_adapter():
    return MagicMock(spec=IGlossaryStorageAdapter)


@pytest.fixture
def miner_api(mock_wikidata_adapter, mock_storage_adapter):
    return GlossaryMinerAPI(
        wikidata_adapter=mock_wikidata_adapter,
        storage_adapter=mock_storage_adapter
    )


class TestGlossaryMinerAPI:
    def test_mine_and_package_subdomain_success(self, miner_api, mock_wikidata_adapter, mock_storage_adapter):
        # Arrange
        mock_wikidata_adapter.fetch_domain_terms.return_value = [
            WikidataItem(item_id="Q1", label="Infarction", aliases_by_lang={"en": ["Heart Attack"]})
        ]
        mock_storage_adapter.save_manifest.return_value = "/mock/path/medical.json"

        # Act
        result = miner_api.mine_and_package_subdomain(
            wikidata_parent_id="Q12136",
            sub_domain_name="MEDICAL",
            file_name="medical.json",
            anchor_words=["oncology"]
        )

        # Assert
        mock_wikidata_adapter.fetch_domain_terms.assert_called_once_with("Q12136", limit=5000)
        mock_storage_adapter.save_manifest.assert_called_once()

        # Verify the manifest payload built by the API
        saved_file_name, saved_manifest = mock_storage_adapter.save_manifest.call_args[0]
        assert saved_file_name == "medical.json"
        assert "MEDICAL" in saved_manifest.sub_domains
        assert "Heart Attack" in saved_manifest.sub_domains["MEDICAL"].terms
        assert "oncology" in saved_manifest.sub_domains["MEDICAL"].anchor_words

        assert result == "/mock/path/medical.json"

    def test_mine_and_package_empty_results_raises_error(self, miner_api, mock_wikidata_adapter):
        # Arrange
        mock_wikidata_adapter.fetch_domain_terms.return_value = []

        # Act & Assert
        with pytest.raises(DataProcessingError, match="returned 0 items"):
            miner_api.mine_and_package_subdomain(
                wikidata_parent_id="Q12136",
                sub_domain_name="MEDICAL",
                file_name="medical.json"
            )