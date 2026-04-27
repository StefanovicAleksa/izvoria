# Canvas: cli_search_v3
# Path: cli_search.py

import uuid
import logging
from typing import Optional  # Ensure Optional is imported
from sqlalchemy import text

# Core Infrastructure
from app.core.database.service.manager import db_manager
from app.core.model_manager.data.vram_monitor import PyTorchVramMonitor
from app.core.model_manager.service.orchestrator import ModelManagerOrchestrator

# Feature Components
from app.features.rag_retrieval.data.embedding_adapter import HuggingFacePromptAdapter
from app.features.rag_retrieval.data.postgres_search_repo import PostgresSearchRepository
from app.features.rag_retrieval.service.retrieval_api import RetrievalAPI
from app.features.rag_retrieval.domain.entities import SearchQuery

# Setup minimal logging for clean CLI output
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("cli_search")
logger.setLevel(logging.INFO)


# FIX: Changed type hint to Optional[uuid.UUID]
def get_vault_path(file_id: Optional[uuid.UUID]) -> str:
    """Queries the central files table to find where the clip is actually stored."""
    if not file_id:
        return "No Clip Linkage Found"
    try:
        with db_manager.session() as session:
            # Reaching into the core storage 'files' table
            result = session.execute(
                text("SELECT file_path FROM files WHERE id = :fid"),
                {"fid": str(file_id)}
            ).scalar_one_or_none()
            return result if result else "File ID not found in storage."
    except Exception as e:
        return f"Error resolving path: {str(e)}"


def run_manual_test():
    print("--- Onyx RAG Retrieval: Direct Feature Test ---")

    # 1. Bootstrapping minimum infrastructure
    vram_monitor = PyTorchVramMonitor()
    orchestrator = ModelManagerOrchestrator(vram_monitor)

    # 2. Instantiating Feature Layer
    adapter = HuggingFacePromptAdapter()
    repo = PostgresSearchRepository()
    api = RetrievalAPI(
        orchestrator=orchestrator,
        embedding_adapter=adapter,
        search_repo=repo
    )

    print("[SYSTEM] Ready. VRAM Monitor Active. DB Connected.")

    while True:
        prompt = input("\n[Enter Search Query] > ").strip()
        if prompt.lower() in ['exit', 'quit']:
            break
        if not prompt:
            continue

        try:
            query = SearchQuery(prompt=prompt, limit=5)

            print(f"Vectorizing and searching for: '{prompt}'...")
            search_result = api.search(query)

            print(f"\nFound {len(search_result.results)} relevant segments in {search_result.execution_time_ms}ms:")
            print("=" * 80)

            for i, chunk in enumerate(search_result.results, 1):
                # This call now passes type validation
                physical_path = get_vault_path(chunk.extracted_file_id)

                print(f"RESULT #{i} | RRF Score: {chunk.relevance_score:.5f}")
                print(f"Timestamp: {chunk.start_sec}s - {chunk.end_sec}s")
                print(f"Context: {chunk.text_payload}")
                print(f"Disk Path: {physical_path}")
                print("-" * 80)

        except Exception as e:
            print(f"Search failed: {e}")


if __name__ == "__main__":
    run_manual_test()