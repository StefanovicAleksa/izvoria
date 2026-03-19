# Canvas: test_miner_api_v2
# Path: tests/features/glossary_miner/service/test_miner_api.py

import pytest
from unittest.mock import MagicMock, call

from app.features.glossary_miner.service.miner_api import GlossaryMinerAPI
from app.features.glossary_miner.domain.interfaces import IWikidataAdapter, IGlossaryStorageAdapter
from app.features.glossary_miner.domain.entities import WikidataItem
from app.features.glossary_miner.exceptions import DataProcessingError


@pytest.fixture
def mock_wikidata_adapter() -> MagicMock:
    return MagicMock(spec=IWikidataAdapter)


@pytest.fixture
def mock_storage_adapter() -> MagicMock:
    return MagicMock(spec=IGlossaryStorageAdapter)


@pytest.fixture
def miner_api(mock_wikidata_adapter: MagicMock, mock_storage_adapter: MagicMock) -> GlossaryMinerAPI:
    return GlossaryMinerAPI(
        wikidata_adapter=mock_wikidata_adapter,
        storage_adapter=mock_storage_adapter
    )


class TestGlossaryMinerAPI:

    # ==========================================
    # TESTS FOR build_industry_manifest
    # ==========================================

    def test_build_industry_manifest_success(self, miner_api: GlossaryMinerAPI, mock_wikidata_adapter: MagicMock,
                                             mock_storage_adapter: MagicMock):
        # Arrange
        def mock_fetch_domain_terms(q_id, limit):
            if q_id == "Q1":
                return [WikidataItem(item_id="Q100", label="Quantitative Easing", aliases_by_lang={"en": ["QE"]})]
            elif q_id == "Q2":
                return [WikidataItem(item_id="Q200", label="Contango", aliases_by_lang={})]
            return []

        mock_wikidata_adapter.fetch_domain_terms.side_effect = mock_fetch_domain_terms
        mock_storage_adapter.save_manifest.return_value = "/mock/path/finance.json"

        targets = {
            "MONETARY_POLICY": {"q_id": "Q1", "anchors": ["ecb"]},
            "FUTURES_MARKET": {"q_id": "Q2", "anchors": ["margin"]}
        }
        base_terms = ["liquidity", "volatility"]

        # Act
        result = miner_api.build_industry_manifest(
            file_name="finance.json",
            base_terms=base_terms,
            sub_domain_targets=targets,
            limit=5000
        )

        # Assert
        assert mock_wikidata_adapter.fetch_domain_terms.call_count == 2
        mock_wikidata_adapter.fetch_domain_terms.assert_has_calls([
            call("Q1", limit=5000),
            call("Q2", limit=5000)
        ], any_order=True)

        mock_storage_adapter.save_manifest.assert_called_once()

        saved_file_name, saved_manifest = mock_storage_adapter.save_manifest.call_args[0]
        assert saved_file_name == "finance.json"

        # Verify Manifest Structure
        assert saved_manifest.base_terms == base_terms
        assert "MONETARY_POLICY" in saved_manifest.sub_domains
        assert "FUTURES_MARKET" in saved_manifest.sub_domains

        assert "ecb" in saved_manifest.sub_domains["MONETARY_POLICY"].anchor_words
        assert "Quantitative Easing" in saved_manifest.sub_domains["MONETARY_POLICY"].terms
        assert "Contango" in saved_manifest.sub_domains["FUTURES_MARKET"].terms

        assert result == "/mock/path/finance.json"

    def test_build_industry_manifest_skips_missing_qid(self, miner_api: GlossaryMinerAPI,
                                                       mock_wikidata_adapter: MagicMock,
                                                       mock_storage_adapter: MagicMock):
        # Arrange
        mock_wikidata_adapter.fetch_domain_terms.return_value = [
            WikidataItem(item_id="Q1", label="Valid Term", aliases_by_lang={})
        ]
        targets = {
            "INVALID_DOMAIN": {"anchors": ["broken"]},  # Missing q_id
            "VALID_DOMAIN": {"q_id": "Q1", "anchors": ["works"]}
        }

        # Act
        miner_api.build_industry_manifest("test.json", [], targets)

        # Assert
        # Should only call wikidata once for the valid domain
        mock_wikidata_adapter.fetch_domain_terms.assert_called_once_with("Q1", limit=5000)

        saved_file_name, saved_manifest = mock_storage_adapter.save_manifest.call_args[0]
        assert "VALID_DOMAIN" in saved_manifest.sub_domains
        assert "INVALID_DOMAIN" not in saved_manifest.sub_domains

    def test_build_industry_manifest_empty_results_raises_error(self, miner_api: GlossaryMinerAPI,
                                                                mock_wikidata_adapter: MagicMock):
        # Arrange
        mock_wikidata_adapter.fetch_domain_terms.return_value = []
        targets = {
            "EMPTY_DOMAIN": {"q_id": "Q999", "anchors": ["nothing"]}
        }

        # Act & Assert
        with pytest.raises(DataProcessingError, match="Failed to extract any valid sub-domains"):
            miner_api.build_industry_manifest("test.json", [], targets)

    # ==========================================
    # TESTS FOR LEGACY mine_and_package_subdomain
    # ==========================================

    def test_mine_and_package_subdomain_success(self, miner_api: GlossaryMinerAPI, mock_wikidata_adapter: MagicMock,
                                                mock_storage_adapter: MagicMock):
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

    def test_mine_and_package_empty_results_raises_error(self, miner_api: GlossaryMinerAPI,
                                                         mock_wikidata_adapter: MagicMock):
        # Arrange
        mock_wikidata_adapter.fetch_domain_terms.return_value = []

        # Act & Assert
        with pytest.raises(DataProcessingError, match="returned 0 items"):
            miner_api.mine_and_package_subdomain(
                wikidata_parent_id="Q12136",
                sub_domain_name="MEDICAL",
                file_name="medical.json"
            )