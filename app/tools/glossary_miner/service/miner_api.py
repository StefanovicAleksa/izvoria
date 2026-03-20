# Canvas: glossary_miner_api_v2
# Path: app/features/glossary_miner/service/miner_api.py

import logging
from typing import List, Optional, Dict, Any

from app.tools.glossary_miner.domain.interfaces import IWikidataAdapter, IGlossaryStorageAdapter
from app.tools.glossary_miner.domain.entities import OnyxGlossaryManifest, SubDomainGlossary
from app.tools.glossary_miner.utils.term_cleaner import clean_and_extract_terms
from app.tools.glossary_miner.exceptions import DataProcessingError

logger = logging.getLogger(__name__)


class GlossaryMinerAPI:
    """
    The Orchestrator for the Glossary Miner Developer Tool.
    Automates the ETL pipeline: Wikidata -> Deterministic Cleanup -> JSON Manifest.
    Updated to support aggregating multiple Sub-Domains into a single Master Industry Manifest.
    """

    def __init__(
            self,
            wikidata_adapter: IWikidataAdapter,
            storage_adapter: IGlossaryStorageAdapter
    ):
        self._wikidata = wikidata_adapter
        self._storage = storage_adapter

    def build_industry_manifest(
            self,
            file_name: str,
            base_terms: List[str],
            sub_domain_targets: Dict[str, Dict[str, Any]],
            limit: int = 5000
    ) -> str:
        """
        Executes the full pipeline to generate an aggregated Onyx-ready JSON glossary.

        Args:
            file_name: The desired output file name (e.g., 'finance.json').
            base_terms: General industry terminology applicable across all sub-domains.
            sub_domain_targets: A dictionary mapping the Sub-Domain name (e.g., 'MONETARY_POLICY')
                                to its config: {"q_id": "Q178564", "anchors": ["ecb", "fomc"]}.
            limit: Maximum number of terms to fetch from Wikidata per sub-domain.

        Returns:
            str: The absolute path to the saved JSON file.
        """
        logger.info(f"Starting aggregated extraction for Master Glossary: {file_name}")

        sub_domains_payload: Dict[str, SubDomainGlossary] = {}

        for sd_name, config in sub_domain_targets.items():
            q_id = config.get("q_id")
            anchors = config.get("anchors", [])

            if not q_id:
                logger.error(f"Skipping {sd_name}: Missing 'q_id' in config.")
                continue

            logger.info(f"Mining Wikidata for {sd_name} (Parent: {q_id})...")

            # 1. EXTRACT
            raw_items = self._wikidata.fetch_domain_terms(q_id, limit=limit)
            if not raw_items:
                logger.warning(f"Wikidata returned 0 items for {sd_name} (Q-ID {q_id}). Skipping.")
                continue

            # 2. TRANSFORM
            cleaned_terms = clean_and_extract_terms(raw_items)
            logger.info(f"Heuristic cleanup resulted in {len(cleaned_terms)} Whisper-ready terms for {sd_name}.")

            # 3. BUILD SUB-DOMAIN
            sub_domains_payload[sd_name.upper()] = SubDomainGlossary(
                anchor_words=anchors,
                terms=cleaned_terms
            )

        if not sub_domains_payload:
            raise DataProcessingError(f"Failed to extract any valid sub-domains for {file_name}.")

        # 4. PACKAGE
        manifest = OnyxGlossaryManifest(
            base_terms=base_terms,
            sub_domains=sub_domains_payload
        )

        # 5. LOAD
        saved_path = self._storage.save_manifest(file_name, manifest)
        logger.info(f"Master Glossary successfully packaged and saved to: {saved_path}")

        return saved_path

    def mine_and_package_subdomain(
            self,
            wikidata_parent_id: str,
            sub_domain_name: str,
            file_name: str,
            anchor_words: Optional[List[str]] = None,
            limit: int = 5000
    ) -> str:
        """
        Legacy Method: Executes the full pipeline to generate an Onyx-ready JSON glossary for a single subdomain.
        """
        logger.info(f"Starting extraction for Sub-Domain: {sub_domain_name} (Q-ID: {wikidata_parent_id})")

        raw_items = self._wikidata.fetch_domain_terms(wikidata_parent_id, limit=limit)

        if not raw_items:
            raise DataProcessingError(f"Wikidata returned 0 items for Q-ID {wikidata_parent_id}.")

        logger.info(f"Successfully fetched {len(raw_items)} raw items from Wikidata.")

        cleaned_terms = clean_and_extract_terms(raw_items)

        logger.info(f"Heuristic cleanup resulted in {len(cleaned_terms)} Whisper-ready terms.")

        sub_domain_glossary = SubDomainGlossary(
            anchor_words=anchor_words or [],
            terms=cleaned_terms
        )

        manifest = OnyxGlossaryManifest(
            base_terms=[],
            sub_domains={
                sub_domain_name.upper(): sub_domain_glossary
            }
        )

        saved_path = self._storage.save_manifest(file_name, manifest)

        logger.info(f"Glossary successfully packaged and saved to: {saved_path}")
        return saved_path