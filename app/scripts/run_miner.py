# Quick script you run manually from your terminal:
from app.features.glossary_miner.data.sparql_client import WikidataSparqlClient
from app.features.glossary_miner.data.file_storage_adapter import LocalFileStorageAdapter
from app.features.glossary_miner.service.miner_api import GlossaryMinerAPI

# Dependency Injection
wikidata_client = WikidataSparqlClient()
storage_client = LocalFileStorageAdapter(base_dir="data/glossaries")

api = GlossaryMinerAPI(wikidata_client, storage_client)

# Example: Generate the Medical Oncology Dictionary
api.mine_and_package_subdomain(
    wikidata_parent_id="Q12136", # Disease
    sub_domain_name="MEDICAL_ONCOLOGY",
    file_name="medical_oncology.json",
    anchor_words=["oncology", "carcinoma", "chemotherapy", "tumor", "malignant"]
)