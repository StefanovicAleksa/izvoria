# Canvas: glossary_miner_api_v1
# Path: app/features/glossary_miner/service/miner_api.py

import logging
from typing import List, Optional

from app.features.glossary_miner.domain.interfaces import IWikidataAdapter, IGlossaryStorageAdapter
from app.features.glossary_miner.domain.entities import OnyxGlossaryManifest, SubDomainGlossary
from app.features.glossary_miner.utils.term_cleaner import clean_and_extract_terms
from app.features.glossary_miner.exceptions import DataProcessingError

logger = logging.getLogger(__name__)


class GlossaryMinerAPI:
    """
    The Orchestrator for the Glossary Miner Developer Tool.
    Automates the ETL pipeline: Wikidata -> Deterministic Cleanup -> JSON Manifest.
    """

    def __init__(
            self,
            wikidata_adapter: IWikidataAdapter,
            storage_adapter: IGlossaryStorageAdapter
    ):
        self._wikidata = wikidata_adapter
        self._storage = storage_adapter

    def mine_and_package_subdomain(
            self,
            wikidata_parent_id: str,
            sub_domain_name: str,
            file_name: str,
            anchor_words: Optional[List[str]] = None,
            limit: int = 5000
    ) -> str:
        """
        Executes the full pipeline to generate an Onyx-ready JSON glossary.

        Args:
            wikidata_parent_id: The Q-ID of the domain (e.g., 'Q12136' for Disease).
            sub_domain_name: The internal Onyx label (e.g., 'MEDICAL_ONCOLOGY').
            file_name: The desired output file name (e.g., 'medical_oncology.json').
            anchor_words: Highly specific keywords to deterministically trigger this glossary.
            limit: Maximum number of terms to fetch from Wikidata.

        Returns:
            str: The absolute path to the saved JSON file.
        """
        logger.info(f"Starting extraction for Sub-Domain: {sub_domain_name} (Q-ID: {wikidata_parent_id})")

        # 1. EXTRACT: Fetch raw entities from the Wikidata Graph
        # Automatically filters for 'en' and 'la' at the database level.
        raw_items = self._wikidata.fetch_domain_terms(wikidata_parent_id, limit=limit)

        if not raw_items:
            raise DataProcessingError(f"Wikidata returned 0 items for Q-ID {wikidata_parent_id}.")

        logger.info(f"Successfully fetched {len(raw_items)} raw items from Wikidata.")

        # 2. TRANSFORM: Deterministic Heuristic Cleanup
        # Strips brackets, drops long phrases, and formats for Whisper hotwords.
        cleaned_terms = clean_and_extract_terms(raw_items)

        logger.info(f"Heuristic cleanup resulted in {len(cleaned_terms)} Whisper-ready terms.")

        # 3. PACKAGE: Build the Aggregate Root directly (No external builder needed)
        sub_domain_glossary = SubDomainGlossary(
            anchor_words=anchor_words or [],
            terms=cleaned_terms
        )

        manifest = OnyxGlossaryManifest(
            base_terms=[],  # Can be appended manually or by another script later
            sub_domains={
                sub_domain_name.upper(): sub_domain_glossary
            }
        )

        # 4. LOAD: Persist strictly to disk
        saved_path = self._storage.save_manifest(file_name, manifest)

        logger.info(f"Glossary successfully packaged and saved to: {saved_path}")
        return saved_path