# Canvas: main_v7
# Path: app/main.py

import os
import sys
import uuid
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


def print_deep_research_result(result):
    if result.qa_response:
        print(f"\n[ONYX AI - DEEP RESEARCH]\n{result.qa_response.generated_answer}")
        print(f"\n[CITATIONS USED: {len(result.qa_response.citations)}]")
        for cit in result.qa_response.citations:
            print(f"  {cit.doc_id} {cit.physical_path or 'No physical file resolved'}")
            print(f"         Timestamp: {cit.start_sec}s - {cit.end_sec}s")
        if result.critique_unresolved:
            print("\n[NOTE] Self-critique had unresolved concerns about this answer's grounding.")
    else:
        print(f"\n[ONYX AI - CLARIFICATION NEEDED]\n{result.clarification_question}")

    print(f"\n[REASONING TRACE: {result.total_iterations} iteration(s)]")
    for step in result.iteration_trace:
        print(f"  Iter {step.iteration}: {step.decision} - {step.interim_analysis[:150]}")
        if step.refined_query:
            print(f"    -> refined query: {step.refined_query}")


def print_chat_response(response):
    print(f"\n[ONYX AI]\n{response.generated_answer}")
    print(f"\n[CITATIONS USED: {len(response.citations)}]")
    for cit in response.citations:
        print(f"  {cit.doc_id} {cit.physical_path or 'No physical file resolved'}")
        print(f"         Timestamp: {cit.start_sec}s - {cit.end_sec}s")


def run_conversation(engine, session_id):
    """
    Multi-turn chat: each question is a turn of the session's conversation, so follow-ups can refer to earlier
    questions, answers and cited excerpts. Returns the session id to keep using (/new starts a fresh one).
    """
    print("\n[SYSTEM] Conversation mode. Blank line returns to the menu; /new starts a new conversation.")
    while True:
        prompt = input("\n[You] > ").strip()
        if not prompt:
            return session_id
        if prompt.lower() == "/new":
            session_id = uuid.uuid4()
            print("[SYSTEM] Started a new conversation.")
            continue

        print("\n[SYSTEM] Running RAG + QA Generation Pipeline...")
        try:
            response = engine.chat(SearchQuery(prompt=prompt, limit=5), session_id=session_id)
        except EngineError as e:
            print(f"\n[ENGINE ERROR] Task failed: {e}")
            continue
        print_chat_response(response)


def run_deep_research_with_clarifications(engine, session_id, original_question, max_clarification_rounds=3):
    """
    chat_deep_research()'s intra-question state (evidence pool, decision history) is ephemeral
    and discarded once the call returns, so a clarification round can't resume mid-loop — it
    re-asks with the original question plus every clarification given so far appended, and lets
    the Reasoner start its reasoning fresh with that fuller picture.
    """
    clarifications = []
    for round_num in range(max_clarification_rounds + 1):
        conversation = original_question
        for i, clarification in enumerate(clarifications, 1):
            conversation += f"\n\n[Clarification {i}] {clarification}"

        print("\n[SYSTEM] Running Deep Research Reasoning Loop (this may take a while)...")
        query = SearchQuery(prompt=conversation, limit=5)
        result = engine.chat_deep_research(session_id, query)
        print_deep_research_result(result)

        if result.qa_response is not None:
            return

        if round_num >= max_clarification_rounds:
            print(f"\n[SYSTEM] Reached the clarification limit ({max_clarification_rounds} rounds) without a grounded answer.")
            return

        answer = input("\n[Your clarification] > (leave blank to cancel) ").strip()
        if not answer:
            print("\n[SYSTEM] Cancelled.")
            return
        clarifications.append(answer)


def main():
    print_header()
    print("[SYSTEM] Initializing hardware state and preloading UI models...")

    try:
        engine = OnyxBootstrapper.build()
        engine.start()
    except Exception as e:
        print(f"\n[FATAL STARTUP ERROR] Appliance failed to boot: {e}")
        sys.exit(1)

    # Chat (5) and Deep Research (7) share one conversation until /new is typed in chat
    conversation_session_id = uuid.uuid4()

    try:
        while True:
            print("\nAVAILABLE OPERATIONS:")
            print("  1. Ingest Folders [Video Pipeline - FULL]")
            print("  2. Ingest Folders [Audio Pipeline - FAST]")
            print("  3. Standard Search [Fast RAG]")
            print("  4. Verified Search [SLM Hallucination Scrubber]")
            print("  5. Chat with Data [Multi-Turn Conversation]")
            print("  6. Shutdown Appliance")
            print("  7. Deep Research Chat [Multi-Pass Reasoning]")

            choice = input("\n[Select Option 1-7] > ").strip()

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
                    conversation_session_id = run_conversation(engine, conversation_session_id)

                elif choice == '7':
                    prompt = input("[Enter your research question] > ").strip()
                    if prompt:
                        run_deep_research_with_clarifications(engine, conversation_session_id, prompt)

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