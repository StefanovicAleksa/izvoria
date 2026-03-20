# Canvas: test_glossary_miner_sparql_client_v4
# Path: tests/features/glossary_miner/data/test_sparql_client.py

import pytest
import urllib.error
from email.message import Message
from unittest.mock import patch, MagicMock

from app.tools.glossary_miner.data.sparql_client import WikidataSparqlClient
from app.tools.glossary_miner.exceptions import SparqlServerTimeoutError, TermExtractionError


class TestWikidataSparqlClient:

    @patch("app.tools.glossary_miner.data.sparql_client.SPARQLWrapper")
    def test_fetch_domain_terms_success(self, mock_sparql_class):
        mock_client_instance = MagicMock()
        mock_sparql_class.return_value = mock_client_instance

        mock_client_instance.queryAndConvert.return_value = {
            "results": {
                "bindings": [
                    {
                        "item": {"value": "https://www.wikidata.org/entity/Q123"},
                        "itemLabel": {"value": "Primary Term"},
                        "alias": {"value": "Alias Term"},
                        "aliasLang": {"value": "en"}
                    }
                ]
            }
        }

        client = WikidataSparqlClient()
        results = client.fetch_domain_terms("Q1")

        assert len(results) == 1
        assert results[0].item_id == "Q123"
        assert results[0].label == "Primary Term"
        assert "Alias Term" in results[0].aliases_by_lang["en"]

    @patch("app.tools.glossary_miner.data.sparql_client.SPARQLWrapper")
    def test_fetch_domain_terms_server_timeout(self, mock_sparql_class):
        mock_client_instance = MagicMock()
        mock_sparql_class.return_value = mock_client_instance

        # HTTP 500 on Wikidata usually means Blazegraph timed out computing the query
        mock_client_instance.queryAndConvert.side_effect = urllib.error.HTTPError(
            "https://test.local", 500, "Internal Error", Message(), None
        )

        client = WikidataSparqlClient()

        # We now correctly expect the mapped timeout error, not a generic network error
        with pytest.raises(SparqlServerTimeoutError):
            client.fetch_domain_terms("Q1")

    @patch("app.tools.glossary_miner.data.sparql_client.SPARQLWrapper")
    def test_fetch_domain_terms_malformed_json(self, mock_sparql_class):
        mock_client_instance = MagicMock()
        mock_sparql_class.return_value = mock_client_instance

        mock_client_instance.queryAndConvert.return_value = {"garbage_data": True}

        client = WikidataSparqlClient()

        with pytest.raises(TermExtractionError, match="missing 'results'"):
            client.fetch_domain_terms("Q1")