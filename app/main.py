# Canvas: main_v5
# Path: app/main.py

import os
import sys
import logging
from pathlib import Path

# CRITICAL FIX: Ensure PyTorch uses expandable segments to prevent fragmentation crashes.
# This MUST be declared at the absolute top before any PyTorch-dependent modules are imported.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from app.engine.service.bootstrap import OnyxBootstrapper
from app.engine.exceptions.base import EngineError
from app.features.rag_retrieval.domain.entities import SearchQuery

# Setup minimal logging for clean CLI output
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("onyx_cli")
logger.setLevel(logging.INFO)


def print_header():
    print("==================================================")
    print("             ONYX ENTERPRISE APPLIANCE            ")
    print("==================================================")


def do_ingest(engine):
    print("\n--- INGESTION MODE ---")
    raw_path = input("[Enter Absolute Path to Media Folder] > ").strip()

    if not raw_path:
        return

    target_path = Path(raw_path)
    if not target_path.exists():
        print(f"[ERROR] Path does not exist: {target_path}")
        return

    print(f"\n[SYSTEM] Locking hardware and starting batch pipeline for: {target_path}")
    print("[SYSTEM] This may take a while depending on file size and hardware...")

    try:
        # Engine expects a string path, so we cast the validated Path object back to string
        report = engine.ingest(str(target_path))

        print("\n==================================================")
        print("             INGESTION REPORT                     ")
        print("==================================================")
        print(f"Target Path: {report.target_path}")
        print(f"Duration:    {report.total_duration_seconds} seconds")
        print(f"Scanned:     {report.total_files_scanned} files")
        print(f"Success:     {report.successful_ingestions} files")
        print(f"Failed:      {report.failed_ingestions} files")
        print(f"Vectors:     {report.vectors_created}")
        print(f"Clips:       {report.clips_extracted}")

        if report.errors:
            print("\n[ERRORS RECORDED]")
            for key, err_msg in report.errors.items():
                print(f" - {key}: {err_msg}")

    except EngineError as e:
        print(f"\n[FATAL ERROR] Pipeline aborted: {e}")


def do_search(engine):
    print("\n--- SEARCH MODE ---")
    while True:
        prompt = input("\n[Enter Search Query] (or 'back' to return) > ").strip()
        if prompt.lower() in ['back', 'exit', 'quit', 'b']:
            break
        if not prompt:
            continue

        try:
            query = SearchQuery(prompt=prompt, limit=5)
            print(f"Vectorizing and executing hybrid search for: '{prompt}'...\n")

            search_result = engine.search(query)

            print(f"Found {len(search_result.results)} relevant segments in {search_result.execution_time_ms}ms:")
            print("-" * 80)

            for i, chunk in enumerate(search_result.results, 1):
                physical_path = getattr(chunk, 'physical_path', 'No physical file resolved')

                print(f"RESULT #{i} | RRF Score: {chunk.relevance_score:.5f}")
                print(f"Timestamp: {chunk.start_sec}s - {chunk.end_sec}s")
                print(f"Context: {chunk.text_payload}")
                print(f"Disk Path: {physical_path}")
                print("-" * 80)

        except EngineError as e:
            print(f"\n[ERROR] Search execution failed: {e}")


def main():
    print_header()
    print("[SYSTEM] Initializing background workers and allocating VRAM...")

    try:
        engine = OnyxBootstrapper.build()
        engine.start()
    except Exception as e:
        print(f"[FATAL STARTUP ERROR] {e}")
        sys.exit(1)

    try:
        while True:
            print("\nMAIN MENU:")
            print("  1. Ingest Media Folder")
            print("  2. Search Vault")
            print("  3. Shutdown Appliance")

            choice = input("\n[Select Option 1-3] > ").strip()

            if choice == '1':
                do_ingest(engine)
            elif choice == '2':
                do_search(engine)
            elif choice == '3' or choice.lower() in ['exit', 'quit']:
                print("\n[SYSTEM] Initiating graceful shutdown sequence...")
                break
            else:
                print("[ERROR] Invalid selection.")

    except KeyboardInterrupt:
        print("\n[SYSTEM] Emergency interrupt detected. Releasing hardware locks...")
    finally:
        engine.stop()
        print("[SYSTEM] Onyx Appliance is Offline.")


if __name__ == "__main__":
    main()