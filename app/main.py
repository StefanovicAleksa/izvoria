# Canvas: main_v7
# Path: app/main.py

import os
import sys
import logging

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
    print("\n" + "=" * 60)
    print("          O N Y X   E N T E R P R I S E   A I          ")
    print("=" * 60)
    print("   AIR-GAPPED AI APPLIANCE | VRAM-MANAGED PIPELINES    ")
    print("=" * 60 + "\n")


def print_report(report):
    print("\n" + "-" * 20 + " INGESTION REPORT " + "-" * 20)
    print(f"Target Path: {report.target_path}")
    print(f"Duration:    {report.total_duration_seconds} seconds")
    print(f"Scanned:     {report.total_files_scanned} files")
    print(f"Success:     {report.successful_ingestions} files")
    print(f"Failed:      {report.failed_ingestions} files")
    print(f"Vectors:     {report.vectors_created}")
    print(f"Clips:       {report.clips_extracted}")
    print("-" * 58)

    if report.errors:
        print("\n[ERRORS RECORDED]")
        for key, err_msg in report.errors.items():
            print(f" ! {key}: {err_msg}")


def print_search_results(results, mode_name: str):
    print(f"\n[{mode_name}] Found {len(results.results)} relevant segments:")
    print("=" * 80)

    for i, chunk in enumerate(results.results, 1):
        physical_path = getattr(chunk, 'physical_path', 'No physical file resolved')

        print(f"RESULT #{i}")
        print(f"  Source:    {physical_path}")
        print(f"  Timestamp: {chunk.start_sec}s - {chunk.end_sec}s")
        print(f"  Content:   {chunk.text_payload}")
        print("-" * 80)


def main():
    print_header()
    print("[SYSTEM] Initializing hardware state and preloading UI models...")

    try:
        engine = OnyxBootstrapper.build()
        engine.start()
    except Exception as e:
        print(f"\n[FATAL STARTUP ERROR] Appliance failed to boot: {e}")
        sys.exit(1)

    try:
        while True:
            print("\nAVAILABLE OPERATIONS:")
            print("  1. Ingest Folders [Video Pipeline - FULL]")
            print("  2. Ingest Folders [Audio Pipeline - FAST]")
            print("  3. Standard Search [Fast RAG]")
            print("  4. Verified Search [SLM Hallucination Scrubber]")
            print("  5. Chat with Data [RAG -> LLM Synthesis]")
            print("  6. Shutdown Appliance")

            choice = input("\n[Select Option 1-6] > ").strip()

            try:
                if choice == '1':
                    raw_path = input("[Enter Absolute Path to Video Folder] > ").strip()
                    if raw_path:
                        print("\n[SYSTEM] Evicting UI models... Initializing Video Pipeline.")
                        report = engine.ingest_videos(raw_path)
                        print_report(report)

                elif choice == '2':
                    raw_path = input("[Enter Absolute Path to Audio Folder] > ").strip()
                    if raw_path:
                        print("\n[SYSTEM] Evicting UI models... Initializing Audio Pipeline.")
                        report = engine.ingest_audios(raw_path)
                        print_report(report)

                elif choice == '3':
                    prompt = input("[Enter Search Query] > ").strip()
                    if prompt:
                        query = SearchQuery(prompt=prompt, limit=5)
                        results = engine.search_standard(query)
                        print_search_results(results, "STANDARD RAG")

                elif choice == '4':
                    prompt = input("[Enter High-Precision Query] > ").strip()
                    if prompt:
                        print("\n[SYSTEM] Running SLM Scrubber verification...")
                        query = SearchQuery(prompt=prompt, limit=5)
                        results = engine.search_verified(query)
                        print_search_results(results, "VERIFIED RETRIEVAL")

                elif choice == '5':
                    prompt = input("[Enter your question] > ").strip()
                    if prompt:
                        print("\n[SYSTEM] Running RAG + QA Generation Pipeline...")
                        query = SearchQuery(prompt=prompt, limit=5)
                        response = engine.chat(query)

                        print(f"\n[ONYX AI]\n{response.generated_answer}")
                        print(f"\n[CITATIONS USED: {len(response.citations)}]")
                        for i, cit in enumerate(response.citations, 1):
                            path = getattr(cit, 'physical_path', 'No physical file resolved')
                            print(f"  {i}. {path} (Doc {cit.doc_id})")

                elif choice == '6' or choice.lower() in ['exit', 'quit']:
                    print("\n[SYSTEM] Initiating graceful shutdown sequence...")
                    break

                else:
                    print("[ERROR] Invalid selection.")

            except EngineError as e:
                print(f"\n[ENGINE ERROR] Task failed: {e}")
            except Exception as e:
                print(f"\n[CRITICAL FAILURE] Unexpected crash: {e}")

    except KeyboardInterrupt:
        print("\n[SYSTEM] Emergency interrupt detected.")
    finally:
        engine.stop()
        print("[SYSTEM] Onyx Appliance is Offline.")


if __name__ == "__main__":
    main()