# Path: scripts/run_miner.py

import logging
from app.core.config import settings
from app.tools.glossary_miner.data.sparql_client import WikidataSparqlClient
from app.tools.glossary_miner.data.file_storage_adapter import LocalFileStorageAdapter
from app.tools.glossary_miner.service.miner_api import GlossaryMinerAPI

# Setup basic terminal logging so you can see the progress
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def main():
    # 1. Initialize Adapters
    wikidata_client = WikidataSparqlClient()
    # Terms are mined in the appliance's language (APPLIANCE_LANGUAGE) into that language's glossary directory
    storage_client = LocalFileStorageAdapter(base_dir=str(settings.GLOSSARY_DIR))

    # 2. Initialize Orchestrator
    api = GlossaryMinerAPI(wikidata_client, storage_client)

    # 3. Execute for Finance
    api.mine_and_package_subdomain(
        wikidata_parent_id="Q247506",  # The correct ID for Financial Instrument
        sub_domain_name="CORPORATE_FINANCE",
        file_name="finance.json",
        anchor_words=["ebitda", "amortization", "liquidity", "dividend", "equity"],
        limit=2000
    )

if __name__ == "__main__":
    main()