# Canvas: glossary_miner_interfaces_v2
# Path: app/features/glossary_miner/domain/interfaces.py

from abc import ABC, abstractmethod
from typing import List

from app.tools.glossary_miner.domain.entities import WikidataItem, IzvoriaGlossaryManifest


class IWikidataAdapter(ABC):
    """
    Contract for fetching raw terminology from the Wikidata SPARQL endpoint.
    The implementation will be responsible for rate-limiting and HTTP connection management.
    """

    @abstractmethod
    def fetch_domain_terms(self, wikidata_parent_id: str, limit: int = 2000) -> List[WikidataItem]:
        """
        Executes a SPARQL query to retrieve all child entities of a specific domain.
        Must filter for the appliance language's labels and aliases natively at the database level.
        """
        pass


class IGlossaryStorageAdapter(ABC):
    """
    Contract for persisting the finalized JSON manifest to the Izvoria Appliance's data directory.
    """

    @abstractmethod
    def save_manifest(self, file_name: str, manifest: IzvoriaGlossaryManifest) -> str:
        """
        Saves the Pydantic model as a strict JSON file.
        Returns the absolute path to the saved file.
        """
        pass