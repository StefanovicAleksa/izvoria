# Canvas: glossary_miner_sparql_client_v1
# Path: app/features/glossary_miner/data/sparql_client.py

from typing import List, Dict, Any, Optional
from SPARQLWrapper import SPARQLWrapper, JSON
from pydantic import ValidationError

from app.core.language.service.language_api import get_language_pack
from app.tools.glossary_miner.domain.interfaces import IWikidataAdapter
from app.tools.glossary_miner.domain.entities import WikidataItem
from app.tools.glossary_miner.data.queries import GET_DOMAIN_TERMS_QUERY
from app.tools.glossary_miner.utils.error_mapper import map_wikidata_error
from app.tools.glossary_miner.exceptions import TermExtractionError, ManifestFormattingError


class WikidataSparqlClient(IWikidataAdapter):
    """
    Concrete adapter for querying the Wikidata knowledge graph.
    Uses SPARQLWrapper to execute queries and parses the flat tabular results
    into strictly nested WikidataItem domain entities.
    """

    def __init__(self, endpoint_url: str = "https://query.wikidata.org/sparql", languages: Optional[List[str]] = None):
        self.endpoint_url = endpoint_url
        # Wikidata label languages (the pack's [glossary] table). Latin only ever supplies aliases: a Latin label
        # would put a Latin word where the appliance's own language should be.
        self.languages = list(languages or get_language_pack().glossary.wikidata_languages)
        self.label_languages = [lang for lang in self.languages if lang != "la"] or self.languages
        # Wikidata strictly enforces User-Agents to prevent server abuse.
        self.user_agent = "OnyxGlossaryMiner/1.0 (Enterprise RAG Appliance DevTool; Contact: dev@onyx.local)"

        self.client = SPARQLWrapper(self.endpoint_url, agent=self.user_agent)
        self.client.setReturnFormat(JSON)

    def fetch_domain_terms(self, wikidata_parent_id: str, limit: int = 2000) -> List[WikidataItem]:
        """
        Executes the query and catches/maps all network/query exceptions.
        """
        query_str = GET_DOMAIN_TERMS_QUERY.format(
            parent_id=wikidata_parent_id,
            limit=limit,
            label_languages=_sparql_list(self.label_languages),
            alias_languages=_sparql_list(self.languages),
        )
        self.client.setQuery(query_str)

        try:
            # Query the endpoint and get the raw JSON dict
            response: Dict[str, Any] = self.client.queryAndConvert()
        except Exception as e:
            # Trap any urllib, socket, or SPARQLWrapper exceptions and map them to our Domain
            raise map_wikidata_error(e) from e

        # Translate flat dictionary to strictly typed Domain Entities
        return self._parse_bindings(response)

    def _parse_bindings(self, response: Dict[str, Any]) -> List[WikidataItem]:
        """
        Internal mapping method. Aggregates multiple flat rows containing aliases
        into a single nested entity per unique Item ID.
        """
        try:
            bindings = response["results"]["bindings"]
        except KeyError as e:
            raise TermExtractionError(
                message="Malformed JSON from Wikidata: missing 'results' or 'bindings' keys.",
                original_exception=e
            ) from e

        items_map: Dict[str, Dict[str, Any]] = {}

        for row in bindings:
            # Wikidata URIs look like "http://www.wikidata.org/entity/Q123". We just want "Q123".
            raw_item_uri = row.get("item", {}).get("value", "")
            item_id = raw_item_uri.split("/")[-1] if "/" in raw_item_uri else raw_item_uri
            label = row.get("itemLabel", {}).get("value", "")

            if not item_id or not label:
                continue

            # Initialize the dict entry if it's the first time we see this item_id
            if item_id not in items_map:
                items_map[item_id] = {
                    "item_id": item_id,
                    "label": label,
                    "aliases_by_lang": {lang: [] for lang in self.languages}
                }

            # If this specific row contains an alias, append it to the correct language array
            alias = row.get("alias", {}).get("value")
            alias_lang = row.get("aliasLang", {}).get("value")

            if alias and alias_lang in self.languages:
                # Prevent duplicate alias entries
                if alias not in items_map[item_id]["aliases_by_lang"][alias_lang]:
                    items_map[item_id]["aliases_by_lang"][alias_lang].append(alias)

        try:
            # Dump the mapped dictionaries into our immutable Pydantic entities
            return [WikidataItem(**data) for data in items_map.values()]
        except ValidationError as e:
            raise ManifestFormattingError(
                message="Failed to instantiate WikidataItem entities. Validation constraint broken.",
                original_exception=e
            ) from e

def _sparql_list(languages: List[str]) -> str:
    """Language codes as a SPARQL IN list: '"sr", "sr-el"'."""
    return ", ".join(f'"{lang}"' for lang in languages)
